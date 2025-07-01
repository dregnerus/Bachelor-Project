import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# Loading the csv path
csv_path = r"C:\Users\Lenovo X1 Carbon\OneDrive\Documents\Doutsen\Bachelor project\Results Porcine\Force\Trial_23.csv"

# 1) Read, skipping any junk first line
df = pd.read_csv(csv_path, header=None, skiprows=1)

# 2) Drop everything before the 259th data‐row (zero‐based index 258)
df_tail = df.iloc[103:].copy().reset_index(drop=True)

# 3) Take that very first timestamp as t₀
t0 = df_tail.iat[0, 0]

# 4) Compute time in seconds
df_tail['time_s'] = (df_tail[0] - t0) / 1000.0

df_tail = df_tail[df_tail['time_s'] <= 10.0].reset_index(drop=True)

# 5) Force as float
df_tail['force'] = df_tail[1].astype(float)

# 6) median‐filter the force to remove single‐frame spikes
df_tail['force_filt'] = (
    df_tail['force']
      .rolling(window=11, center=True)
      .median()
      .fillna(method='bfill')
      .fillna(method='ffill')
)

# ───── INTERPOLATE FORCE TO EVEN 1 ms TIME STEPS ─────
max_time = df_tail['time_s'].iloc[-1]
even_times = np.arange(0, max_time + 0.0005, 0.001)  # step = 1 ms

# Interpolate using filtered force values
even_force = np.interp(even_times, df_tail['time_s'], df_tail['force_filt'])

df_even = pd.DataFrame({
    "time_s": even_times,
    "force": even_force
})

# Save interpolated data
even_out_path = os.path.splitext(csv_path)[0] + '_Force_1ms.csv'
df_even.to_csv(even_out_path, index=False)
print(f"Saved 1ms-sampled force data to {even_out_path}")

# ───── SAVE original force–time data ─────────────────
out_df = df_tail[['time_s', 'force']]
out_path = os.path.splitext(csv_path)[0] + '_Force_raw.csv'
out_df.to_csv(out_path, index=False)
print(f"Saved raw force–time data to {out_path}")

# ───── PLOT force vs. time ────────────────────────────
plt.figure(figsize=(8, 4))
plt.plot(df_tail['time_s'], df_tail['force_filt'], linewidth=1)
plt.xlabel('Time (s)')
plt.ylabel('Force (N)')
plt.title('Force vs. Time - Porcine Material')
plt.grid(True)
plt.tight_layout()
plt.show()
