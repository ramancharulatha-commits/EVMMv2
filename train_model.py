import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pathlib import Path
import joblib

DATA = Path("memory_dataset.csv")
df = pd.read_csv(DATA)

df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
df = df.sort_values(["pod_name", "timestamp"])

# Create past-memory features and a future target for each pod.
df["memory_previous"] = df.groupby("pod_name")["memory_mb"].shift(1)
df["memory_previous_2"] = df.groupby("pod_name")["memory_mb"].shift(2)
df["target_memory"] = df.groupby("pod_name")["memory_mb"].shift(-1)

df = df.dropna().copy()

features = ["memory_mb", "memory_previous", "memory_previous_2"]
X = df[features]
y = df["target_memory"]

# Chronological split: train on earlier observations, test on later ones.
split = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions) ** 0.5
r2 = r2_score(y_test, predictions) if len(y_test) > 1 else float("nan")

print("Evaluation results")
print(f"Test observations: {len(y_test)}")
print(f"MAE:  {mae:.4f} MiB")
print(f"RMSE: {rmse:.4f} MiB")
print(f"R2:   {r2:.4f}")
print("\nNote: near-constant memory can make these results misleading.")

joblib.dump(model, "memory_predictor.joblib")

plt.figure(figsize=(10, 5))
plt.plot(y_test.reset_index(drop=True).iloc[:100], label="Actual memory")
plt.plot(pd.Series(predictions[:100]), label="Predicted memory")
plt.xlabel("Test observation")
plt.ylabel("Memory (MiB)")
plt.title("Memory Prediction: Actual vs Predicted")
plt.legend()
plt.tight_layout()
plt.savefig("memory_prediction.png", dpi=150)
print("\nSaved model: memory_predictor.joblib")
print("Saved graph: memory_prediction.png")
