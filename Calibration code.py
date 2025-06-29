import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

# Load the calibration data (first row is header)
df = pd.read_csv("Calibration.csv", header=0)

# Calculate average output for each weight
df["Average"] = df.iloc[:, 1:21].mean(axis=1)

# Fit a linear regression model: RawOutput → Weight
X = df["Average"].values.reshape(-1, 1)  # predictor
y = df["Weight"].values  # target
model = LinearRegression()
model.fit(X, y)

a = model.coef_[0]
b = model.intercept_

# Convert regression to output force (in Newtons)
k = a * 0.00980665  # multiply slope by g
c = b * 0.00980665  # multiply intercept by g

print(f"\nCalibration model:")
print(f"Weight (g) = {a:.6f} * RawOutput + {b:.2f}")
print(f"Force (N)  = {k:.8f} * RawOutput + {c:.4f}")

# Plot the calibration plot
plt.figure(figsize=(14,7))    # wider and shorter
plt.plot(df["Weight"], df["Average"], marker='o')
plt.xlabel("Weight (g)")
plt.ylabel("Average Raw Output")
plt.title("Calibration Plot")
plt.grid(True)
plt.show()
