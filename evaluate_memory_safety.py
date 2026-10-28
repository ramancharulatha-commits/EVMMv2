import math
import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error

DATA_FILE = "dynamic_memory_dataset.csv"
MODEL_FILE = "dynamic_memory_predictor.joblib"
SAFETY_MARGIN = 1.20

df = pd.read_csv(DATA_FILE)
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
df = df.sort_values("timestamp").reset_index(drop=True)

df["previous_1"] = df["memory_mb"].shift(1)
df["previous_2"] = df["memory_mb"].shift(2)
df["previous_3"] = df["memory_mb"].shift(3)
df["target"] = df["memory_mb"].shift(-1)
df = df.dropna().reset_index(drop=True)

features = [
    "memory_mb", "previous_1", "previous_2", "previous_3"
]

split = int(len(df) * 0.7)
test = df.iloc[split:].copy()

model = joblib.load(MODEL_FILE)
predictions = model.predict(test[features])

test["prediction"] = predictions
test["prediction"] = test["prediction"].clip(lower=0)
test["buffered_prediction"] = (
    test["prediction"] * SAFETY_MARGIN
)
test["actual_next"] = test["target"]

covered = (
    test["buffered_prediction"] >= test["actual_next"]
)

# Compare the model with a simple persistence baseline:
# predict that the next reading equals the current reading.
baseline = test["memory_mb"]

print("\nMEMORY SAFETY EVALUATION")
print("------------------------")
print(f"Test observations: {len(test)}")
print(
    "Model MAE: "
    f"{mean_absolute_error(test['actual_next'], test['prediction']):.3f} MiB"
)
print(
    "Persistence baseline MAE: "
    f"{mean_absolute_error(test['actual_next'], baseline):.3f} MiB"
)
print(
    "Actual next readings covered by 20% margin: "
    f"{covered.sum()}/{len(test)} ({covered.mean() * 100:.1f}%)"
)
print(
    "Largest actual next reading: "
    f"{test['actual_next'].max():.1f} MiB"
)
print(
    "Largest buffered prediction: "
    f"{test['buffered_prediction'].max():.1f} MiB"
)

print("\nFirst 10 test observations:")
print(
    test[["memory_mb", "prediction", "buffered_prediction", "actual_next"]]
    .head(10).round(1).to_string(index=False)
)

print("\nThis is a one-step-ahead evaluation on one pod.")
print("It does not prove that reducing the live memory limit is safe.")
