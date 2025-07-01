import pandas as pd
import numpy as np
import os

# 1) point this at your merged file
IN_CSV = r"C:\Users\Lenovo X1 Carbon\OneDrive\Documents\Doutsen\Bachelor project\Results experiment\Force-depth_data\Slow\Slow_10_Force_depth.csv"

# 2) load and keep only depth & force
df = pd.read_csv(IN_CSV)[['real_depth_mm','force']].dropna()

# 3) prepend the (0 mm, 0 N) anchor
df0 = pd.DataFrame({'real_depth_mm':[0.0],'force':[0.0]})
df = pd.concat([df0, df], ignore_index=True).sort_values('real_depth_mm')

# 4) build your depth grid at 0.1 mm steps
d_max = df['real_depth_mm'].max()
depth_grid = np.arange(0, d_max + 1e-6, 0.1)

# 5) interpolate force onto that grid
force_on_d = np.interp(depth_grid,
                       df['real_depth_mm'],
                       df['force'],
                       left=np.nan, right=np.nan)

# 6) save out a new CSV
out = pd.DataFrame({'depth_mm': depth_grid, 'force_N': force_on_d})
OUT_CSV = os.path.splitext(IN_CSV)[0] + '_force_per_depth.csv'
out.to_csv(OUT_CSV, index=False)
print("Wrote:", OUT_CSV)
