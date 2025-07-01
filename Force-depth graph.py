import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Loading CSVs to combine
force_csv = r"C:\Users\Lenovo X1 Carbon\OneDrive\Documents\Doutsen\Bachelor project\Results Porcine\Porcine_force-time.csv"
depth_csv = r"C:\Users\Lenovo X1 Carbon\OneDrive\Documents\Doutsen\Bachelor project\Results Porcine\Video\Porcine_depth_time.csv"

df_f = pd.read_csv(force_csv)    # ['time_s','force']
df_d = pd.read_csv(depth_csv)    # ['time_s','real_depth_mm']

# Dropping duplicate timestamps
df_f = df_f.drop_duplicates('time_s').set_index('time_s')
df_d = df_d.drop_duplicates('time_s').set_index('time_s')

# Defining common time grid
t_min = max(df_f.index.min(), df_d.index.min())
t_max = min(df_f.index.max(), df_d.index.max())
time_grid = np.arange(t_min, t_max, 0.02)

# Interpolating both to common time grid
force_i = df_f.reindex(time_grid).interpolate(method='index')
depth_i = df_d.reindex(time_grid).interpolate(method='index')

# Enforcing increasing depth
depth_vals = depth_i['real_depth_mm'].values
for i in range(1, len(depth_vals)):
    if depth_vals[i] < depth_vals[i - 1]:
        depth_vals[i] = depth_vals[i - 1]
depth_i['real_depth_mm'] = depth_vals

# Merging into one dataframe
df_fd = pd.concat([force_i, depth_i], axis=1)
df_fd.index.name = 'time_s'
df_fd = df_fd.reset_index()

# Computing velocity (from final depth + time)
df_fd['velocity_mm_per_s'] = np.gradient(df_fd['real_depth_mm'], df_fd['time_s'])
df_fd['velocity_mm_per_s'] = df_fd['velocity_mm_per_s'].clip(lower=0)

# Saving merged CSV
out_csv = force_csv.replace('.csv', '_force_depth_velocity.csv')
df_fd.to_csv(out_csv, index=False)
print("✅ Saved merged force–depth–velocity data to:", out_csv)

# Plotting force vs. depth
plt.figure(figsize=(8, 5))
plt.plot(df_fd['real_depth_mm'], df_fd['force'], '-', lw=1)
plt.xlabel("Depth (mm)")
plt.ylabel("Force (N)")
plt.title("Force vs. Depth - Porcine Material")
plt.grid(True)
plt.tight_layout()
plt.show()
