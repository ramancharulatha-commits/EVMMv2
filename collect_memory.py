import csv
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

OUTPUT = Path("memory_dataset.csv")
INTERVAL_SECONDS = 5

def get_memory_usage():
    result = subprocess.run(
        ["kubectl", "top", "pods", "--no-headers"],
        capture_output=True,
        text=True,
        check=True
    )

    measurements = []

    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) < 3:
            continue

        pod_name = fields[0]
        memory = fields[2]

        if memory.endswith("Ki"):
            memory_mb = float(memory[:-2]) / 1024
        elif memory.endswith("Mi"):
            memory_mb = float(memory[:-2])
        elif memory.endswith("Gi"):
            memory_mb = float(memory[:-2]) * 1024
        else:
            continue

        measurements.append((pod_name, memory_mb))

    return measurements

file_exists = OUTPUT.exists()

with OUTPUT.open("a", newline="") as file:
    writer = csv.writer(file)

    if not file_exists or OUTPUT.stat().st_size == 0:
        writer.writerow(["timestamp", "pod_name", "memory_mb"])

    print("Collecting Kubernetes memory measurements.")
    print("Press Ctrl+C to stop collecting.")

    try:
        while True:
            timestamp = datetime.now(timezone.utc).isoformat()

            try:
                measurements = get_memory_usage()

                for pod_name, memory_mb in measurements:
                    writer.writerow([timestamp, pod_name, memory_mb])
                    print(f"{timestamp} | {pod_name} | {memory_mb:.2f} MiB")

                file.flush()

            except (subprocess.CalledProcessError, FileNotFoundError) as error:
                print(f"Collection failed: {error}")

            time.sleep(INTERVAL_SECONDS)

    except KeyboardInterrupt:
        print("\nCollection stopped.")
        print(f"Data saved to: {OUTPUT.resolve()}")
