import math
import subprocess
from pathlib import Path

import joblib
import pandas as pd

MODEL_FILE = "dynamic_memory_predictor.joblib"
DATA_FILE = "dynamic_memory_dataset.csv"
POD_NAME = "dynamic-memory-test"

SAFETY_MARGIN = 1.20
MINIMUM_LIMIT_MB = 64


def run_command(command):
    result = subprocess.run(
        command, capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "Command failed")
    return result.stdout.strip()


def parse_memory(value):
    value = value.strip()

    if value.endswith("Ki"):
        return float(value[:-2]) / 1024
    if value.endswith("Mi"):
        return float(value[:-2])
    if value.endswith("Gi"):
        return float(value[:-2]) * 1024
    if value.endswith("Ti"):
        return float(value[:-2]) * 1024 * 1024

    raise ValueError(f"Unsupported Kubernetes memory unit: {value}")


def get_live_memory():
    output = run_command([
        "kubectl", "top", "pod", POD_NAME, "--no-headers"
    ])
    fields = output.split()

    if len(fields) < 3:
        raise RuntimeError(f"Unexpected kubectl top output: {output}")

    return parse_memory(fields[2])


def get_configured_limit():
    output = run_command([
        "kubectl", "get", "pod", POD_NAME,
        "-o", "jsonpath={.spec.containers[0].resources.limits.memory}"
    ])

    if not output:
        return None

    return parse_memory(output)


def main():
    for filename in (MODEL_FILE, DATA_FILE):
        if not Path(filename).exists():
            raise FileNotFoundError(
                f"{filename} not found. Check your project files."
            )

    model = joblib.load(MODEL_FILE)

    df = pd.read_csv(DATA_FILE)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    if len(df) < 4:
        raise RuntimeError("At least four historical measurements are needed.")

    # Match the training feature order exactly.
    values = df["memory_mb"].tail(4).tolist()[::-1]
    current, previous_1, previous_2, previous_3 = values

    features = pd.DataFrame([{
        "memory_mb": current,
        "previous_1": previous_1,
        "previous_2": previous_2,
        "previous_3": previous_3,
    }])

    prediction = max(0.0, float(model.predict(features)[0]))
    recommended_mb = math.ceil(prediction * SAFETY_MARGIN)

    # Read the actual pod measurements and configuration.
    live_memory_mb = get_live_memory()
    current_limit_mb = get_configured_limit()

    # A recommendation is not allowed to reduce below the chosen
    # demonstration floor. It is not applied automatically.
    recommended_mb = max(MINIMUM_LIMIT_MB, recommended_mb)

    print("\nPREDICTIVE MEMORY RECOMMENDATION")
    print("--------------------------------")
    print(f"Pod:                       {POD_NAME}")
    print(f"Live memory usage:         {live_memory_mb:.1f} MiB")
    print(f"Latest training reading:   {current:.1f} MiB")
    print(f"Predicted next usage:      {prediction:.1f} MiB")
    print(f"Prediction + 20% margin:   {math.ceil(prediction * SAFETY_MARGIN)} MiB")
    print(
        "Configured memory limit:  "
        + (f"{current_limit_mb:.1f} MiB"
           if current_limit_mb is not None else "Not set")
    )
    print(f"Suggested limit:           {recommended_mb} MiB")

    print("\nDECISION")
    print("--------")

    if current_limit_mb is None:
        print("No memory limit is configured; review the workload before setting one.")
    elif prediction * SAFETY_MARGIN > current_limit_mb:
        print("INCREASE / INVESTIGATE: predicted demand plus margin exceeds the limit.")
    elif live_memory_mb >= current_limit_mb * 0.85:
        print("INVESTIGATE: live usage is at least 85% of the configured limit.")
    elif recommended_mb < current_limit_mb:
        print("POTENTIAL REDUCTION: validate peak usage and repeated tests first.")
    else:
        print("RETAIN: the current limit covers the predicted demand plus margin.")

    print("\nRecommendation only: no Kubernetes configuration was changed.")
    print("Validate using repeated measurements and additional workloads.")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"ERROR: {exc}")
