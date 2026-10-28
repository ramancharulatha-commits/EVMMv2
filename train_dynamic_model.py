import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

df = pd.read_csv("dynamic_memory_dataset.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
df = df.sort_values("timestamp").reset_index(drop=True)

# Predict the next memory measurement from the previous readings.
df["previous_1"] = df["memory_mb"].shift(1)
df["previous_2"] = df["memory_mb"].shift(2)
df["previous_3"] = df["memory_mb"].shift(3)
df["target"] = df["memory_mb"].shift(-1)
df = df.dropna().reset_index(drop=True)

features = ["memory_mb", "previous_1", "previous_2", "previous_3"]
X = df[features]
y = df["target"]

split = int(len(df) * 0.7)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

if len(X_train) < 5 or len(X_test) < 2:
    raise SystemExit("Not enough measurements. Collect more data first.")

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("\nDYNAMIC MEMORY PREDICTION RESULTS")
print("---------------------------------")
print(f"Total usable samples: {len(df)}")
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")
print(f"MAE:  {mean_absolute_error(y_test, predictions):.3f} MiB")
print(f"RMSE: {mean_squared_error(y_test, predictions) ** 0.5:.3f} MiB")
print(f"R2:   {r2_score(y_test, predictions):.3f}")
print("\nNote: this is a small dataset from one test pod.")
print("Results do not establish general performance.")

joblib.dump(model, "dynamic_memory_predictor.joblib")

plt.figure(figsize=(10, 5))
plt.plot(y_test.to_numpy(), marker="o", label="Actual next memory")
plt.plot(predictions, marker="x", label="Predicted next memory")
plt.xlabel("Test sample")
plt.ylabel("Memory (MiB)")
plt.title("Dynamic Kubernetes Memory: Actual vs Predicted")
plt.legend()
plt.tight_layout()
plt.savefig("dynamic_memory_prediction.png", dpi=150)
print("\nSaved model: dynamic_memory_predictor.joblib")
print("Saved graph: dynamic_memory_prediction.png")
