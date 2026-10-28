import subprocess

DEPLOYMENT = "predictive-memory-demo"
CONTAINER = "memory-generator"
TEST_LIMIT_MIB = 96
MIN_LIMIT_MIB = 80
MAX_LIMIT_MIB = 128

def run(*args):
    result = subprocess.run(
        ["kubectl", *args],
        text=True,
        capture_output=True
    )
    if result.returncode != 0:
        raise SystemExit(result.stderr)
    return result.stdout.strip()

current = run(
    "get", "deployment", DEPLOYMENT,
    "-o", "jsonpath={.spec.template.spec.containers[0].resources.limits.memory}"
)
print("Current limit:", current)

# Test a bounded scale-up from 80 MiB to 96 MiB.
if current != "80Mi":
    raise SystemExit("Safety check failed: expected current limit to be 80Mi.")

if not MIN_LIMIT_MIB <= TEST_LIMIT_MIB <= MAX_LIMIT_MIB:
    raise SystemExit("Safety check failed: test limit is outside configured bounds.")

run(
    "set", "resources", f"deployment/{DEPLOYMENT}",
    f"--containers={CONTAINER}",
    f"--limits=memory={TEST_LIMIT_MIB}Mi"
)
print(f"Scale-up update requested: 80Mi -> {TEST_LIMIT_MIB}Mi")

run("rollout", "status", f"deployment/{DEPLOYMENT}", "--timeout=120s")

actual = run(
    "get", "deployment", DEPLOYMENT,
    "-o", "jsonpath={.spec.template.spec.containers[0].resources.limits.memory}"
)
print("Verified configured limit:", actual)

if actual == f"{TEST_LIMIT_MIB}Mi":
    print("PASS: Kubernetes accepted the scale-up configuration.")
else:
    raise SystemExit("FAIL: configured limit does not match expected value.")
