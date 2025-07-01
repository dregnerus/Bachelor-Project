import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# Load the CSV file, skipping any junk first line
csv_path = r"C:\Users\Lenovo X1 Carbon\OneDrive\Documents\Doutsen\Bachelor project\Results Porcine\Force\Trial_23.csv"
df = pd.read_csv(csv_path, header=None, skiprows=1)

# Keep data starting from the 103rd row (index 102)
df_tail = df.iloc[103:].copy().reset_index(drop=True)

# Take first timestamp as t₀ and compute relative time in seconds
t0 = df_tail.iat[0, 0]
df_tail['time_s'] = (df_tail[0] - t0) / 1000.0

# Keep only first 10 seconds of data
df_tail = df_tail[df_tail['time_s'] <= 10.0].reset_index(drop=True)

# Convert force to float (no filtering applied)
df_tail['force'] = df_tail[1].astype(float)

# Interpolate raw force to even 1 ms time steps
max_time = df_tail['time_s'].iloc[-1]
even_times = np.arange(0, max_time + 0.0005, 0.001)  # step = 1 ms
even_force = np.interp(even_times, df_tail['time_s'], df_tail['force'])

df_even = pd.DataFrame({'time_s': even_times, 'force': even_force})

# Save interpolated force data
even_out_path = os.path.splitext(csv_path)[0] + '_Force_1ms_raw.csv'
df_even.to_csv(even_out_path, index=False)
print(f"Saved 1ms-sampled raw force data to {even_out_path}")

# Save raw force–time data (uninterpolated)
out_df = df_tail[['time_s', 'force']]
out_path = os.path.splitext(csv_path)[0] + '_Force_raw.csv'
out_df.to_csv(out_path, index=False)
print(f"Saved raw force–time data to {out_path}")

# Plot raw force vs. time
plt.figure(figsize=(8, 4))
plt.plot(df_tail['time_s'], df_tail['force'], linewidth=1)
plt.xlabel('Time (s)')
plt.ylabel('Force (N)')
plt.title('Force vs. Time - Porcine Material (Raw)')
plt.grid(True)
plt.tight_layout()
plt.show()
