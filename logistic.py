import matplotlib.pyplot as plt
import pandas as pd
import pysindy as ps

# Step 1: Load the dataset
df = pd.read_csv("JapanPopulation.csv")
df = df[["Year", "Population"]]
df = df.dropna()

# Step 2: Extract and scale population
df["Population"] = df["Population"] / 1_000_000

years = df["Year"].values
t = years - 1950  # years since 1950
japan_data = df["Population"].values.reshape(-1, 1)

# Step 3: Build SINDy model
optimizer = ps.STLSQ(threshold=0.0, alpha=0, unbias=False)
library = ps.PolynomialLibrary(degree=4, include_bias=False)
model = ps.SINDy(feature_library=library, optimizer=optimizer)

# Step 4: Fit using shifted time
model.fit(japan_data, t=t, feature_names = ["X"])
model.print()

# Step 5: Simulate using same shifted time
sim = model.simulate(x0=[japan_data[0, 0]], t=t)

plt.figure(dpi=600)
plt.plot(t, japan_data, "o", markersize=2, label="True Values")
plt.ylabel("Population (Millions)")
plt.xlabel("Years Since 1950")
plt.grid(True)
plt.tight_layout()
plt.savefig("japanData.png")
plt.show()

# Step 6: Plot
plt.figure(dpi=600)
plt.plot(t, japan_data, "o", markersize=2, label="True Values")
plt.plot(t, sim, "-", markersize=2, label="SINDy Simulation")
plt.legend()
plt.ylabel("Population (Millions)")
plt.xlabel("Years Since 1950")
plt.grid(True)
plt.tight_layout()
plt.savefig("logisticUntruncated.png")
plt.show()

# Calculate RMSE of the simulation to the real data
import numpy as np

# Flatten arrays so shapes match
true_vals = japan_data.flatten()
pred_vals = sim.flatten()

# RMSE calculation
rmse = np.sqrt(np.mean((true_vals - pred_vals) ** 2))

print(f"RMSE: {rmse:.4f} million people")

# R^2 calculation
ss_res = np.sum((true_vals - pred_vals) ** 2)
ss_tot = np.sum((true_vals - np.mean(true_vals)) ** 2)
r2 = 1 - (ss_res / ss_tot)

print(f"R^2: {r2:.4f}")

# -------------------- Truncate data to t <= 60 --------------------
mask = t <= 60

t_trunc = t[mask]
data_trunc = japan_data[mask]

# -------------------- Refit SINDy on truncated data --------------------
model_trunc = ps.SINDy(feature_library=library, optimizer=optimizer)


model_trunc.fit(data_trunc, t=t_trunc, feature_names = ["X"])
model_trunc.print()
print(model_trunc.coefficients())

# -------------------- Simulate --------------------
sim_trunc = model_trunc.simulate(x0=[data_trunc[0, 0]], t=t_trunc)

# -------------------- Plot --------------------
plt.figure(dpi=600)
plt.plot(t_trunc, data_trunc, "o", markersize=2, label="True (Truncated)")
plt.plot(t_trunc, sim_trunc, "-", markersize=2, label="SINDy (Truncated)")
plt.legend()
plt.xlabel("Years Since 1950")
plt.ylabel("Population (Millions)")
plt.grid(True)
plt.tight_layout()
plt.savefig("logisticTruncated.png")
plt.show()

# -------------------- RMSE --------------------
true_trunc = data_trunc.flatten()
pred_trunc = sim_trunc.flatten()

rmse_trunc = np.sqrt(np.mean((true_trunc - pred_trunc) ** 2))
print(f"Truncated RMSE: {rmse_trunc:.4f} million people")

# -------------------- R^2 --------------------
ss_res_trunc = np.sum((true_trunc - pred_trunc) ** 2)
ss_tot_trunc = np.sum((true_trunc - np.mean(true_trunc)) ** 2)
r2_trunc = 1 - (ss_res_trunc / ss_tot_trunc)

print(f"Truncated R^2: {r2_trunc:.4f}")
