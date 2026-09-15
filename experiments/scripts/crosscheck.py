"""
Meter validation: our RAPL reader vs CodeCarbon, same workload, same window.
Produces the agreement figure quoted in the paper's measurement protocol.
"""
import os, sys, time, logging
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "energy"))
from models import MODELS_DIR
os.environ["HF_HOME"] = os.path.abspath(MODELS_DIR)
logging.getLogger("codecarbon").setLevel(logging.ERROR)

from meter import Meter, measure_idle
from codecarbon import EmissionsTracker

def workload(seconds=25.0):
    """Deterministic CPU-bound work: repeated float matmul via pure python-ish numpy."""
    import numpy as np
    rng = np.random.RandomState(0)
    a = rng.rand(600, 600).astype("float32")
    b = rng.rand(600, 600).astype("float32")
    end = time.perf_counter() + seconds
    k = 0
    while time.perf_counter() < end:
        a @ b; k += 1
    return k

print("[cross] idle baseline (20 s) ...", flush=True)
_, idle_w = measure_idle(20.0)
print(f"[cross] idle = {idle_w:.2f} W", flush=True)

print("[cross] running 25 s workload under BOTH meters simultaneously ...", flush=True)
tracker = EmissionsTracker(measure_power_secs=1, save_to_file=False,
                           log_level="error", tracking_mode="machine")
m = Meter()
tracker.start()
with m:
    n = workload(25.0)
    m.sample()
tracker.stop()
# NOTE: tracker.stop() returns kgCO2eq, NOT energy. Read the energy field.
_d = tracker.final_emissions_data
cc_kwh = _d.cpu_energy          # kWh, CPU package only
cc_co2 = _d.emissions           # kgCO2eq
cc_intensity = (cc_co2 / cc_kwh) if cc_kwh else 0.0

ours_J = m.total_joules
ours_pkg_J = m.joules.get("package-0", 0.0)
cc_J = cc_kwh * 3.6e6
ours_net = ours_J - idle_w * m.seconds

print("\n" + "=" * 62)
print(f"  workload: {n} matmuls in {m.seconds:.2f} s")
print(f"  our meter   (package+dram, gross) : {ours_J:9.2f} J")
print(f"  our meter   (package only, gross) : {ours_pkg_J:9.2f} J")
print(f"  CodeCarbon  (package only, gross) : {cc_J:9.2f} J")
d = 100 * abs(ours_pkg_J - cc_J) / max(cc_J, 1e-9)
print(f"  package-to-package difference     : {d:9.2f} %   {'AGREE' if d < 5 else 'DISAGREE'}")
print(f"  our meter   (net of idle)         : {ours_net:9.2f} J")
print(f"  mean power while working          : {ours_J/m.seconds:9.2f} W  (idle {idle_w:.2f} W)")
print(f"  CodeCarbon grid intensity used    : {cc_intensity:9.4f} kgCO2/kWh")
print(f"  CO2 for this 25 s workload        : {cc_co2*1e6:9.2f} mg")
print("=" * 62)
