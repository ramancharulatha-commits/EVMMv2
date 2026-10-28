import csv
import subprocess
import time
from datetime import datetime, timezone

output = "dynamic_memory_dataset.csv"
duration = 600
interval = 5

print("Collecting for 10 minutes. This will stop automatically.")

with open(output, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["timestamp", "pod_name", "memory_mb"])
    end_time = time.time() + duration

    while time.time() < end_time:
        result = subprocess.run(
            ["kubectl", "top", "pod", "dynamic-memory-test",
             "--no-headers"],
            capture_output=True, text=True
        )

        if result.returncode == 0 and result.stdout.strip():
            fields = result.stdout.split()
            value = fields[2]

            if value.endswith("Mi"):
                memory = float(value[:-2])
            elif value.endswith("Ki"):
                memory = float(value[:-2]) / 1024
            elif value.endswith("Gi"):
                memory = float(value[:-2]) * 1024
            else:
                time.sleep(interval)
                continue

            writer.writerow([
                datetime.now(timezone.utc).isoformat(),
                fields[0],
                memory
            ])
            file.flush()
            print(f"{fields[0]}: {memory:.1f} MiB")

        else:
            print("Metrics unavailable:", result.stderr.strip())

        time.sleep(interval)

print("Collection finished. Saved", output)
