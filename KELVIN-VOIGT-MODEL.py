import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# --- Kelvin–Voigt model ---
def kelvin_voigt(x, k, c):
    depth, velocity = x
    return k * depth + c * velocity

# --- Fit function for one file ---
def fit_kv_model(file_path, label, color, depth_range=(27, 32)):
    df = pd.read_csv(file_path)
    df_layer = df[(df["depth_mm"] >= depth_range[0]) & (df["depth_mm"] <= depth_range[1])]

    depth = df_layer["depth_mm"].values
    velocity = df_layer["velocity_mm_per_s"].values
    force = df_layer["force"].values

    popt, _ = curve_fit(kelvin_voigt, (depth, velocity), force, p0=(1.0, 0.0))
    force_pred = kelvin_voigt((depth, velocity), *popt)

    return {
        "depth": depth,
        "force": force,
        "force_pred": force_pred,
        "label": label,
        "k": popt[0],
        "c": popt[1],
        "color": color
    }

# --- File paths for all groups ---
fast_path = r"C:\Users\Lenovo X1 Carbon\OneDrive\Documents\Doutsen\Bachelor project\Results experiment\All data\Fast_group_avg.csv"
moderate_path = r"C:\Users\Lenovo X1 Carbon\OneDrive\Documents\Doutsen\Bachelor project\Results experiment\All data\Moderate_group_avg.csv"
slow_path = r"C:\Users\Lenovo X1 Carbon\OneDrive\Documents\Doutsen\Bachelor project\Results experiment\All data\Slow_group_avg.csv"

# --- Fit all models ---
# --- Fit all models for interspinous ligament (6–27 mm) ---
fast_result = fit_kv_model(fast_path, "Fast", "green")
moderate_result = fit_kv_model(moderate_path, "Moderate", "red")
slow_result = fit_kv_model(slow_path, "Slow", "blue")

# --- Plot all together ---
plt.figure(figsize=(10, 5))
for result in [fast_result, moderate_result, slow_result]:
    plt.plot(result["depth"], result["force"], '.', color=result["color"], alpha=0.5, label=f"{result['label']} Data")
    plt.plot(result["depth"], result["force_pred"], '-', color=result["color"],
             label=f"{result['label']} KV Fit: k={result['k']:.3f} N/mm, c={result['c']:.3f} N·s/mm")

plt.title("Kelvin–Voigt Model Fit: Ligamentum Flavum (27–32 mm)")
plt.xlabel("Depth (mm)")
plt.ylabel("Force (N)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
