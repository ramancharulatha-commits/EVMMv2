import time
import math
import subprocess
from collections import deque

import joblib
import pandas as pd

MODEL_PATH = "dynamic_memory_predictor.joblib"
DEPLOYMENT = "predictive-memory-demo"
SELECTOR = "app=predictive-memory-demo"

MIN_LIMIT_MIB = 80
MAX_LIMIT_MIB = 128
SAFETY_MARGIN = 0.20
INTERVAL_SECONDS = 30
COOLDOWN_SECONDS = 120

model = joblib.load(MODEL_PATH)
history = deque(maxlen=4)
last_change = 0


def kubectl(*args):
    result = subprocess.run(
        ["kubectl", *args],
        capture_output=True,
        text=True,
        timeout=20
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "kubectl command failed")
    return result.stdout.strip()


def read_memory_mib():
    output = kubectl(
        "top", "pods", "-l", SELECTOR, "--no-headers"
    )
    if not output:
        raise RuntimeError("Metrics are not available yet")

    rows = output.splitlines()
    columns = rows[0].split()
    if len(columns) < 3:
        raise RuntimeError(f"Unexpected metrics output: {output}")

    value = columns[2]
    if value.endswith("Ki"):
        return float(value[:-2]) / 1024
    if value.endswith("Mi"):
        return float(value[:-2])
    if value.endswith("Gi"):
        return float(value[:-2]) * 1024
    raise RuntimeError(f"Unknown memory unit: {value}")


def current_limit_mib():
    value = kubectl(
        "get", "deployment", DEPLOYMENT,
        "-o", "jsonpath={.spec.template.spec.containers[0].resources.limits.memory}"
    )
    if value.endswith("Mi"):
        return int(float(value[:-2]))
    if value.endswith("Gi"):
        return int(float(value[:-2]) * 1024)
    raise RuntimeError(f"Unexpected deployment memory limit: {value}")


def apply_limit(mib):
    kubectl(
        "set", "resources", f"deployment/{DEPLOYMENT}",
        "--containers=memory-generator",
        f"--limits=memory={mib}Mi"
    )


print("Predictive memory controller started.")
print("Target:", DEPLOYMENT)
print(f"Allowed limits: {MIN_LIMIT_MIB}-{MAX_LIMIT_MIB} MiB")
print("Press Ctrl+C to stop the controller.")
print("The demo Deployment may recreate its pod after a limit change.")

try:
    while True:
        try:
            memory = read_memory_mib()
            history.append(memory)

            print(f"\nLive memory usage: {memory:.1f} MiB")

            if len(history) < 4:
                print(f"Collecting history: {len(history)}/4 readings")
                time.sleep(INTERVAL_SECONDS)
                continue

            features = pd.DataFrame([{
                "memory_mb": history[-1],
                "previous_1": history[-2],
                "previous_2": history[-3],
                "previous_3": history[-4],
            }])

            prediction = float(model.predict(features)[0])
            prediction = max(0.0, prediction)
            buffered = prediction * (1 + SAFETY_MARGIN)

            # Round upward to 16 MiB steps, then enforce demo bounds.
            desired = math.ceil(buffered / 16) * 16
            desired = max(MIN_LIMIT_MIB, min(MAX_LIMIT_MIB, desired))

            current = current_limit_mib()

            print(f"Predicted next reading: {prediction:.1f} MiB")
            print(f"Prediction with 20% margin: {buffered:.1f} MiB")
            print(f"Current limit: {current} MiB")
            print(f"Controller target: {desired} MiB")

            if desired == current:
                print("No adjustment needed.")
            elif time.time() - last_change < COOLDOWN_SECONDS:
                print("Change skipped: cooldown is active.")
            else:
                apply_limit(desired)
                last_change = time.time()
                history.clear()
                print(f"Adjustment requested: {current} MiB -> {desired} MiB")
                print("Waiting for Kubernetes to apply the Deployment update.")

        except (RuntimeError, ValueError, subprocess.TimeoutExpired) as exc:
            print("Temporary controller issue:", exc)
        except Exception as exc:
            print("Unexpected controller issue:", repr(exc))

        time.sleep(INTERVAL_SECONDS)

except KeyboardInterrupt:
    print("\nController stopped by user.")
