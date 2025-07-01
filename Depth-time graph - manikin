import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

# Parameters (may change for each trial)
video_path       = r"C:\Users\Lenovo X1 Carbon\OneDrive\Documents\Doutsen\Bachelor project\Results experiment\Video\Moderate_6 - Cropped.mp4" # path to desired video
start_frame_idx  = 82       # starting frame from where tracking should begin
needle_length_mm = 80.0     # full needle length in mm (hub→tip)
zoom_factor      = 2        
min_duration_s   = 14.0      # ensuring the graph runs at least 14 seconds, may change for different trials
pixel_error      = 3        # ± pixels reading uncertainty

# Reading starting frame
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
for _ in range(start_frame_idx + 1):
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Video ended before start frame")
cap.release()

# Showing the selected frame for clicking hub, tip & surface
h, w = frame.shape[:2]
disp = cv2.resize(frame, (int(w*zoom_factor), int(h*zoom_factor)),
                  interpolation=cv2.INTER_NEAREST)
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

# Calibration & initial indent
px_ht      = np.hypot(tip_pt[0]-hub_pt[0], tip_pt[1]-hub_pt[1])
mm_per_pixel = needle_length_mm / px_ht
error_mm   = pixel_error * mm_per_pixel
init_indent_mm = abs(tip_pt[1]-surf_pt[1]) * mm_per_pixel

print(f"Calibration: {px_ht:.1f}px → {needle_length_mm:.1f}mm = {mm_per_pixel:.6f} mm/px")
print(f"Depth‐error bands = ±{error_mm:.3f} mm\n")

# Visualizing the video and track the selected points
cap = cv2.VideoCapture(video_path)
for _ in range(start_frame_idx + 1):
    cap.read()
old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
p0 = np.array([hub_pt, tip_pt, surf_pt], dtype=np.float32).reshape(-1,1,2)
lk = dict(winSize=(15,15), maxLevel=2,
          criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,10,0.03))

times_seconds, measured_depths, indentations = [], [], []
frame_counter = start_frame_idx

# Main tracking loop
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

    # Measured depth
    pixel_dist = np.hypot(h_t[0]-t_t[0], h_t[1]-t_t[1]) * mm_per_pixel
    measured_depths.append(needle_length_mm - pixel_dist)

    # Indentation: vertical distance
    pv = abs(t_t[1] - s_t[1])                    # pixel‐difference in y
    indentation = pv * mm_per_pixel - init_indent_mm
    indentations.append(indentation)


    elapsed = (frame_counter - start_frame_idx) / fps
    times_seconds.append(elapsed)

    # Displaying full frame with dots
    vis = frame_next.copy()
    cv2.circle(vis, (int(h_t[0]),int(h_t[1])), 6, (255,0,0), -1)
    cv2.circle(vis, (int(t_t[0]),int(t_t[1])), 6, (0,0,255), -1)
    cv2.circle(vis, (int(s_t[0]),int(s_t[1])), 6, (0,255,255), -1)
    cv2.putText(vis, f"t={elapsed:.2f}s", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
    cv2.imshow("Tracking (press ESC to stop early)", vis)
    if cv2.waitKey(30) == 27: # ASCII code for Esc key
        break
    if elapsed >= min_duration_s and len(measured_depths)>1 \
       and measured_depths[-1] < measured_depths[-2]:
        break

    old_gray = gray.copy()
    p0 = np.array([h_t, t_t, s_t], dtype=np.float32).reshape(-1,1,2)
    frame_counter += 1

cap.release()
cv2.destroyAllWindows()

# Converting lists to arrays
t_arr    = np.array([0.0] + times_seconds)

# Building individual depth arrays
md_arr  = np.array([measured_depths[0]] + measured_depths)
ind_arr = np.array([indentations[0]]   + indentations)
real_arr= md_arr - ind_arr

# Saving current trial depth
df_out = pd.DataFrame({
    "time_s":        t_arr,
    "real_depth_mm": real_arr
})

# Storing data in a csv file
video_name = os.path.splitext(os.path.basename(video_path))[0]
out_csv = os.path.join(os.path.dirname(video_path), f"Depth_time - {video_name}.csv"
df_out.to_csv(out_csv, index=False)
print(f"Saved depth data to {out_csv}")

# Dropping the first zero‐time entry so dt are all > 0 for velocity calculation
t_for_vel = t_arr[1:]
rd_for_vel = real_arr[1:]

# Printing velocity & std
velocities = np.diff(rd_for_vel) / np.diff(t_for_vel)
mean_vel   = np.mean(velocities)
std_vel    = np.std(velocities)
print(f"Mean velocity = {mean_vel:.3f} mm/s (±{std_vel:.3f} mm/s)\n")

real_err = np.sqrt(2) * error_mm

# Plotting depth vs. time
plt.figure(figsize=(10,6))
l1, = plt.plot(t_arr, md_arr,   label="Measured Depth", linewidth=2, color="tab:blue")
plt.fill_between(t_arr, md_arr-error_mm, md_arr+error_mm, color="tab:blue", alpha=0.3)
l2, = plt.plot(t_arr, ind_arr,  label="Indentation",    linewidth=2, color="tab:orange")
plt.fill_between(t_arr, ind_arr-error_mm, ind_arr+error_mm, color="tab:orange", alpha=0.3)
l3, = plt.plot(t_arr, real_arr, label="Real Depth",     linewidth=2, color="tab:green")
plt.fill_between(t_arr, real_arr-real_err, real_arr+real_err, color="tab:green", alpha=0.3)
plt.legend(handles=[l1, l2, l3], loc="upper left", fontsize=10)
txt = f"V_avg = {mean_vel:.3f} mm/s\nσ = {std_vel:.3f} mm/s"
plt.text(0.95, 0.95, txt,
         transform=plt.gca().transAxes,
         va="top", ha="right",
         bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
plt.xlabel("Time (s)")
plt.ylabel("Depth (mm)")
plt.title("Needle Depth vs. Time - Moderate_6")
plt.grid(True)
plt.tight_layout()
plt.show()
