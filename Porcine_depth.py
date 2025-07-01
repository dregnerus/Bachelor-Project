import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

# Parameters
video_path       = r"C:\Users\Lenovo X1 Carbon\OneDrive\Documents\Doutsen\Bachelor project\Results Porcine\Video\Porcine_22 ‐ Gemaakt met Clipchamp.mp4"
start_frame_idx  = 442
needle_length_mm = 80.0
zoom_factor      = 2
min_duration_s   = 14.0
pixel_error      = 3

# STEP 1) Read starting frame
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
for _ in range(start_frame_idx + 1):
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Video ended before start frame")
cap.release()

# STEP 2) Zoom and click 3 points
h, w = frame.shape[:2]
disp = cv2.resize(frame, (int(w*zoom_factor), int(h*zoom_factor)), interpolation=cv2.INTER_NEAREST)
pts = []
def on_click(evt, x, y, flags, param):
    if evt == cv2.EVENT_LBUTTONDOWN and len(pts) < 3:
        pts.append((x//zoom_factor, y//zoom_factor))
        print("Clicked:", pts[-1])
cv2.namedWindow("Select", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Select", on_click)
while len(pts) < 3:
    vis = disp.copy()
    for i, (xx, yy) in enumerate(pts):
        cv2.circle(vis, (xx*zoom_factor, yy*zoom_factor), 6, (0,255,0), -1)
        cv2.putText(vis, str(i+1), (xx*zoom_factor+8, yy*zoom_factor-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    cv2.imshow("Select", vis)
    if cv2.waitKey(1) == 27:
        pts.clear()
cv2.destroyWindow("Select")
hub_pt, tip_pt, surf_pt = pts
init_surf_y = surf_pt[1]

# STEP 3) Calibration
px_vert = abs(tip_pt[1] - hub_pt[1])
mm_per_pixel = needle_length_mm / px_vert
error_mm = pixel_error * mm_per_pixel
print(f"Calibration (vertical only): {px_vert:.1f}px → {needle_length_mm:.1f}mm = {mm_per_pixel:.6f} mm/px")
print(f"Depth‐error bands = ±{error_mm:.3f} mm\n")

# STEP 4) Reopen video and track
cap = cv2.VideoCapture(video_path)
for _ in range(start_frame_idx + 1):
    cap.read()
old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
p0 = np.array([hub_pt, tip_pt, surf_pt], dtype=np.float32).reshape(-1,1,2)
lk = dict(winSize=(15,15), maxLevel=2,
          criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,10,0.03))

times_seconds, measured_depths, indentations = [], [], []
frame_counter = start_frame_idx

while True:
    ret2, frame_next = cap.read()
    if not ret2:
        break

    gray = cv2.cvtColor(frame_next, cv2.COLOR_BGR2GRAY)
    p1, st, _ = cv2.calcOpticalFlowPyrLK(old_gray, gray, p0, None, **lk)
    st = st.reshape(-1) if st is not None else [1,1,1]

    h_t = tuple(p1[0].ravel()) if st[0] else hub_pt
    t_t = tuple(p1[1].ravel()) if st[1] else tip_pt
    s_t = tuple(p1[2].ravel()) if st[2] else surf_pt

    visible_length_mm = abs(h_t[1] - t_t[1]) * mm_per_pixel
    measured_depths.append(max(0, needle_length_mm - visible_length_mm))

    indentation = max(0, abs(s_t[1] - init_surf_y) * mm_per_pixel)
    indentations.append(indentation)

    elapsed = (frame_counter - start_frame_idx) / fps
    times_seconds.append(elapsed)

    vis = frame_next.copy()
    cv2.circle(vis, (int(h_t[0]),int(h_t[1])), 6, (255,0,0), -1)
    cv2.circle(vis, (int(t_t[0]),int(t_t[1])), 6, (0,0,255), -1)
    cv2.circle(vis, (int(s_t[0]),int(s_t[1])), 6, (0,255,255), -1)
    cv2.putText(vis, f"t={elapsed:.2f}s", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
    cv2.imshow("Tracking (press ESC to stop early)", vis)
    if cv2.waitKey(30) == 27:
        break
    if elapsed >= min_duration_s and len(measured_depths) > 1 and measured_depths[-1] < measured_depths[-2]:
        break

    old_gray = gray.copy()
    p0 = np.array([h_t, t_t, s_t], dtype=np.float32).reshape(-1,1,2)
    frame_counter += 1

cap.release()
cv2.destroyAllWindows()

# STEP 5) Convert and compute depths
t_arr    = np.array([0.0] + times_seconds)
md_arr   = np.maximum(np.array([measured_depths[0]] + measured_depths), 0)
ind_arr  = np.maximum(np.array([indentations[0]] + indentations), 0)
real_arr = np.maximum(md_arr - ind_arr, 0)

# ───── SAVE RAW REAL DEPTH ──────────────────────────────
out_csv = os.path.join(os.path.dirname(video_path), "Porcine_depth.csv")
pd.DataFrame({"real_depth_mm": real_arr}).to_csv(out_csv, index=False)
print(f"Saved raw real depth data to {out_csv}")

# ───── INTERPOLATE TO EVEN TIME STEPS (0.2s) ─────────────────────
max_time = np.floor(t_arr[-1] * 5) / 5.0
even_times = np.arange(0, max_time + 0.001, 0.2)
even_real_depths = np.interp(even_times, t_arr, real_arr)

df_even = pd.DataFrame({
    "time_s": even_times,
    "real_depth_mm": even_real_depths
})

out_csv_even = os.path.splitext(out_csv)[0] + '_even.csv'
df_even.to_csv(out_csv_even, index=False)
print(f"Saved evenly sampled depth data to {out_csv_even}")

# ───── INTERPOLATE TO 1 ms TIME STEPS (0.001s) ─────────────────────
even_times_ms = np.arange(0, t_arr[-1] + 0.0005, 0.001)
even_real_depths_ms = np.interp(even_times_ms, t_arr, real_arr)

df_even_ms = pd.DataFrame({
    "time_s": even_times_ms,
    "real_depth_mm": even_real_depths_ms
})

out_csv_ms = os.path.splitext(out_csv)[0] + '_1ms.csv'
df_even_ms.to_csv(out_csv_ms, index=False)
print(f"Saved 1 ms-sampled depth data to {out_csv_ms}")

# STEP 6) Velocity & Plot (limit to t ≤ 6 s)
t_for_vel = t_arr[1:]
rd_for_vel = real_arr[1:]
velocities = np.diff(rd_for_vel) / np.diff(t_for_vel)
mean_vel = np.mean(velocities)
std_vel = np.std(velocities)
print(f"Mean velocity = {mean_vel:.3f} mm/s (±{std_vel:.3f} mm/s)\n")

# Mask data to only show up to 6 seconds
plot_mask = t_arr <= 6.0
t_plot = t_arr[plot_mask]
md_plot = md_arr[plot_mask]
ind_plot = ind_arr[plot_mask]
real_plot = real_arr[plot_mask]

real_err = np.sqrt(2) * error_mm
plt.figure(figsize=(10,6))
l1, = plt.plot(t_plot, md_plot, label="Measured Depth", linewidth=2, color="tab:blue")
plt.fill_between(t_plot, md_plot-error_mm, md_plot+error_mm, color="tab:blue", alpha=0.3)
l2, = plt.plot(t_plot, ind_plot, label="Indentation", linewidth=2, color="tab:orange")
plt.fill_between(t_plot, ind_plot-error_mm, ind_plot+error_mm, color="tab:orange", alpha=0.3)
l3, = plt.plot(t_plot, real_plot, label="Real Depth", linewidth=2, color="tab:green")
plt.fill_between(t_plot, real_plot-real_err, real_plot+real_err, color="tab:green", alpha=0.3)

plt.legend(handles=[l1, l2, l3], loc="upper left", fontsize=10)
txt = f"V_avg = {mean_vel:.3f} mm/s\nσ = {std_vel:.3f} mm/s"
plt.text(0.95, 0.95, txt, transform=plt.gca().transAxes, va="top", ha="right",
         bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))

plt.xlabel("Time (s)")
plt.ylabel("Depth (mm)")
plt.title("Needle Depth vs. Time - Porcine Material")
plt.grid(True)
plt.tight_layout()
plt.show()
