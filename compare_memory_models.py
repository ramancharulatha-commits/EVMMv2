import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("dynamic_memory_dataset.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
df = df.sort_values("timestamp").reset_index(drop=True)

df["previous_1"] = df["memory_mb"].shift(1)
df["previous_2"] = df["memory_mb"].shift(2)
df["previous_3"] = df["memory_mb"].shift(3)
df["target"] = df["memory_mb"].shift(-1)
df = df.dropna().reset_index(drop=True)

features = ["memory_mb", "previous_1", "previous_2", "previous_3"]
split = int(len(df) * 0.7)
test = df.iloc[split:].copy()

rf = joblib.load("dynamic_memory_predictor.joblib")

predictions = {
    "Persistence": test["memory_mb"].to_numpy(),
    "Random Forest": rf.predict(test[features]),
    "Moving average": test[
        ["memory_mb", "previous_1", "previous_2", "previous_3"]
    ].mean(axis=1).to_numpy()
}

print("\nMODEL COMPARISON")
print("----------------")
for name, pred in predictions.items():
    actual = test["target"].to_numpy()
    mae = mean_absolute_error(actual, pred)
    rmse = mean_squared_error(actual, pred) ** 0.5
    print(f"{name:16s} MAE={mae:.3f} MiB  RMSE={rmse:.3f} MiB")

print("\nLower MAE and RMSE indicate smaller prediction errors.")
print("Evaluation uses the same chronological test split.")
print("Results apply only to this test pod and dataset.")
