import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ─── CONFIG ─────────────────────────────────────────────────────────────
FILE_PATH = r"C:\Users\Lenovo X1 Carbon\OneDrive\Documents\Doutsen\Bachelor project\Results Porcine\Porcine_force_force_depth.csv"
STEP_PLOT = 0.1   # mm for plotting
STEP_CSV  = 0.2   # mm for saved output
MAX_D     = 65.0
LABEL     = "Porcine Material"
COLOR     = "tab:blue"

# ─── Load and prepare data ──────────────────────────────────────────────
df = pd.read_csv(FILE_PATH)

if not {"real_depth_mm", "time_s", "force"}.issubset(df.columns):
    raise ValueError("CSV must contain 'real_depth_mm', 'time_s', and 'force'.")

depth_raw = df["real_depth_mm"].values
time_raw  = df["time_s"].values
force_raw = df["force"].values

# Compute velocity
velocity_raw = np.zeros_like(depth_raw)
velocity_raw[1:] = np.diff(depth_raw) / np.diff(time_raw)
velocity_raw[0] = velocity_raw[1]  # pad first

# Pad if needed
if depth_raw[0] > 0:
    depth_raw = np.insert(depth_raw, 0, 0.0)
    time_raw  = np.insert(time_raw, 0, time_raw[0])
    force_raw = np.insert(force_raw, 0, force_raw[0])
    velocity_raw = np.insert(velocity_raw, 0, 0.0)
if depth_raw[-1] < MAX_D:
    depth_raw = np.append(depth_raw, MAX_D)
    time_raw  = np.append(time_raw, time_raw[-1])
    force_raw = np.append(force_raw, 0.0)
    velocity_raw = np.append(velocity_raw, 0.0)

# ─── Interpolation for plotting (0.1 mm) ────────────────────────────────
depth_grid_plot = np.arange(0, MAX_D + STEP_PLOT/2, STEP_PLOT)
v_interp_plot = np.interp(depth_grid_plot, depth_raw, velocity_raw, left=np.nan, right=np.nan)

# ─── Plot ───────────────────────────────────────────────────────────────
plt.figure(figsize=(8,6))
plt.plot(depth_grid_plot, v_interp_plot, color=COLOR, lw=2, label=LABEL)
plt.xlim(0, MAX_D)
plt.ylim(0, None)
plt.xlabel("Depth (mm)")
plt.ylabel("Velocity (mm/s)")
plt.title("Velocity vs. Depth – Porcine Material")
plt.grid(True, ls="--", alpha=0.4)
plt.tight_layout()
plt.show()

# ─── Save interpolated CSV (0.2 mm steps) ───────────────────────────────
depth_grid_csv = np.arange(0, MAX_D + STEP_CSV/2, STEP_CSV)
out_df = pd.DataFrame({
    "depth_mm": depth_grid_csv,
    "time_s":   np.interp(depth_grid_csv, depth_raw, time_raw),
    "force":    np.interp(depth_grid_csv, depth_raw, force_raw),
    "velocity": np.interp(depth_grid_csv, depth_raw, velocity_raw)
})

out_csv_path = os.path.splitext(FILE_PATH)[0] + "_interpolated.csv"
out_df.to_csv(out_csv_path, index=False)
print(f"✅ Saved interpolated file to:\n{out_csv_path}")
