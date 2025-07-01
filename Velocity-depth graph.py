import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Parameters (may change per dataset)
csv_path     = r"C:\Users\Lenovo X1 Carbon\OneDrive\Documents\Doutsen\Bachelor project\Results Porcine\Porcine_force_depth.csv"
label        = "Porcine Material"
color        = "tab:blue"
depth_step   = 0.1    # mm step size for both plotting and saved CSV
max_depth    = 65.0   # limit depth-axis

# Reading CSV file
df = pd.read_csv(csv_path)
if not {"real_depth_mm", "time_s", "force"}.issubset(df.columns):
    raise ValueError("CSV must contain 'real_depth_mm', 'time_s', and 'force' columns.")

depth_raw = df["real_depth_mm"].values
time_raw  = df["time_s"].values
force_raw = df["force"].values

# Computing velocity
velocity_raw = np.zeros_like(depth_raw)
velocity_raw[1:] = np.diff(depth_raw) / np.diff(time_raw)
velocity_raw[0] = velocity_raw[1]

# Padding beginning if depth does not start at zero
if depth_raw[0] > 0:
    depth_raw    = np.insert(depth_raw, 0, 0.0)
    time_raw     = np.insert(time_raw, 0, time_raw[0])
    force_raw    = np.insert(force_raw, 0, force_raw[0])
    velocity_raw = np.insert(velocity_raw, 0, 0.0)

# Padding end if depth does not reach max
if depth_raw[-1] < max_depth:
    depth_raw    = np.append(depth_raw, max_depth)
    time_raw     = np.append(time_raw, time_raw[-1])
    force_raw    = np.append(force_raw, 0.0)
    velocity_raw = np.append(velocity_raw, 0.0)

# Interpolation grid
depth_grid = np.arange(0, max_depth + depth_step/2, depth_step)
time_interp     = np.interp(depth_grid, depth_raw, time_raw)
force_interp    = np.interp(depth_grid, depth_raw, force_raw)
velocity_interp = np.interp(depth_grid, depth_raw, velocity_raw)

# Plotting velocity vs. depth
plt.figure(figsize=(8, 6))
plt.plot(depth_grid, velocity_interp, linewidth=2, color=color, label=label)
plt.xlabel("Depth (mm)")
plt.ylabel("Velocity (mm/s)")
plt.title("Velocity vs. Depth – Porcine Material")
plt.xlim(0, max_depth)
plt.ylim(0, None)
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.show()

# Saving interpolated data to CSV
df_out = pd.DataFrame({
    "depth_mm": depth_grid,
    "time_s":   time_interp,
    "force":    force_interp,
    "velocity": velocity_interp
})
out_path = os.path.splitext(csv_path)[0] + "_velocity.csv"
df_out.to_csv(out_path, index=False)
print(f"✅ Saved interpolated file to:\n{out_path}")
