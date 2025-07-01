import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# Loading the csv path
csv_path = r"C:\Users\Lenovo X1 Carbon\OneDrive\Documents\Doutsen\Bachelor project\Results Porcine\Force\Trial_23.csv"

# Skipping the first line, in case of an empty line
df = pd.read_csv(csv_path, header=None, skiprows=1)

# Only keeping data starting from needle insertion (row 103 onward, differs per trial)
df_tail = df.iloc[103:].copy().reset_index(drop=True)

# Taking that row as t0 and converting to seconds
t0 = df_tail.iat[0, 0]
df_tail['time_s'] = (df_tail[0] - t0) / 1000.0

# Keeping data within the estimated duration for plotting
df_tail = df_tail[df_tail['time_s'] <= 10.0].reset_index(drop=True)

# Rolling median filter to remove error spikes
df_tail['force'] = df_tail[1].astype(float)
df_tail['force_filt'] = (
    df_tail['force']
    .rolling(window=5, center=True)
    .median()
    .bfill()
    .ffill()
)

# Interpolating to 1 ms time steps
even_times = np.arange(0, df_tail['time_s'].iloc[-1] + 0.001, 0.001)
even_force = np.interp(even_times, df_tail['time_s'], df_tail['force_filt'])

df_even = pd.DataFrame({'time_s': even_times, 'force': even_force})

# Storing data interpolated data in a csv file
base_path = os.path.splitext(csv_path)[0]
df_even.to_csv(f"{base_path}_Force_time-filtered.csv", index=False)
print(f"Saved interpolated data to {base_path}_Force_time-filtered.csv")

# Plotting filtered force vs. time
plt.figure(figsize=(8, 4))
plt.plot(df_tail['time_s'], df_tail['force_filt'], linewidth=1)
plt.xlabel('Time (s)')
plt.ylabel('Force (N)')
plt.title('Force vs. Time - Porcine Material')  # Update per trial name if needed
plt.grid(True)
plt.tight_layout()
plt.show()
