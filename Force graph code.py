import pandas as pd
import matplotlib.pyplot as plt

# 1) Path to your Fast_4.csv (raw‐string)
csv_path = r"C:\Users\Lenovo X1 Carbon\OneDrive\Documents\Doutsen\Bachelor project\Results experiment\Fast\Force\Fast_10.csv"

# 2) Read the CSV, skipping the first “00” line:
df = pd.read_csv(csv_path,
                 header=None,
                 sep=',',
                 engine='python',
                 skiprows=1)   # ← drops that lone “00” line

# 3) Drop all rows BEFORE the 238th original (so original row 238 → index 237 becomes df_tail.index 0)
df_tail = df.iloc[248:].copy().reset_index(drop=True)

# 4) Take that “238th‐row” trial value as t₀ (in ms)
t0 = df_tail.loc[0, 0]

# 5) Convert raw “trial” (ms) to time in seconds
df_tail['time_s'] = (df_tail[0] - t0) / 1000.0

# 6) Extract force values (second column)
df_tail['force'] = df_tail[1]

df_tail = df_tail[df_tail['time_s'] <= 2.0].reset_index(drop=True)

# 7) Plot force vs. time
plt.figure(figsize=(8, 4))
plt.plot(df_tail['time_s'], df_tail['force'], linewidth=1)
plt.xlabel('Time (s)')
plt.ylabel('Force (N)')
plt.title('Force vs. Time - Fast_10')
plt.grid(True)
plt.tight_layout()
plt.show()
