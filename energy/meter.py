"""
Energy measurement for LLM inference.

Reads Intel RAPL (CPU + DRAM) and, when available, NVIDIA NVML (GPU).
Handles the things that silently ruin energy numbers: counter wraparound,
idle-power subtraction, and the fact that a single fast query is below
the sensor's noise floor.

Requires read access to RAPL:
    sudo chmod -R a+r /sys/class/powercap/intel-rapl
"""

import os
import time
import glob
import statistics


# ----------------------------------------------------------------- RAPL

class RaplDomain:
    """One RAPL energy counter (package, core, or dram)."""

    def __init__(self, path):
        self.path = path
        self.name = open(os.path.join(path, "name")).read().strip()
        self.max_uj = int(open(os.path.join(path, "max_energy_range_uj")).read())

    def read_uj(self):
        return int(open(os.path.join(self.path, "energy_uj")).read())

    def delta_uj(self, before, after):
        """Energy between two readings, correcting for counter wraparound."""
        if after >= before:
            return after - before
        return after + self.max_uj - before


def find_rapl_domains():
    domains = []
    for path in sorted(glob.glob("/sys/class/powercap/intel-rapl:*")):
        if "mmio" in path:
            continue
        try:
            domains.append(RaplDomain(path))
        except (PermissionError, FileNotFoundError):
            pass
    return domains


# ----------------------------------------------------------------- NVML

class Gpu:
    """GPU power sampling. Pascal cards have no energy counter, so we
    integrate power over time instead."""

    def __init__(self):
        import pynvml
        self.nvml = pynvml
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        self.name = pynvml.nvmlDeviceGetName(self.handle)
        if isinstance(self.name, bytes):
            self.name = self.name.decode()
        try:
            pynvml.nvmlDeviceGetTotalEnergyConsumption(self.handle)
            self.has_energy_counter = True
        except Exception:
            self.has_energy_counter = False

    def watts(self):
        return self.nvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0

    def energy_mj(self):
        return self.nvml.nvmlDeviceGetTotalEnergyConsumption(self.handle)


# ----------------------------------------------------------- measurement

class Meter:
    """Context manager that accumulates energy across a block of work."""

    def __init__(self, gpu_sample_hz=100, use_gpu=True):
        self.domains = find_rapl_domains()
        self.gpu_period = 1.0 / gpu_sample_hz
        if not use_gpu:
            self.gpu = None
            return
        try:
            self.gpu = Gpu()
        except Exception:
            self.gpu = None

    def __enter__(self):
        self._t0 = time.perf_counter()
        self._before = {d.name: d.read_uj() for d in self.domains}
        self._gpu_samples = []
        if self.gpu and not self.gpu.has_energy_counter:
            self._gpu_last = time.perf_counter()
        elif self.gpu:
            self._gpu_before = self.gpu.energy_mj()
        return self

    def sample(self):
        """Call periodically during long work so GPU power gets integrated."""
        if self.gpu and not self.gpu.has_energy_counter:
            now = time.perf_counter()
            self._gpu_samples.append((self.gpu.watts(), now - self._gpu_last))
            self._gpu_last = now

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self._t0
        self.joules = {}
        for d in self.domains:
            uj = d.delta_uj(self._before[d.name], d.read_uj())
            self.joules[d.name] = uj / 1e6
        if self.gpu:
            if self.gpu.has_energy_counter:
                self.joules["gpu"] = (self.gpu.energy_mj() - self._gpu_before) / 1000.0
            else:
                self.sample()
                self.joules["gpu"] = sum(w * dt for w, dt in self._gpu_samples)
        return False

    @property
    def total_joules(self):
        # 'package' already contains 'core'; don't double count.
        return sum(v for k, v in self.joules.items() if k != "core")


def measure_idle(seconds=60.0, use_gpu=True):
    """Baseline power with nothing running. Subtract this from measurements."""
    m = Meter(use_gpu=use_gpu)
    with m:
        end = time.perf_counter() + seconds
        while time.perf_counter() < end:
            time.sleep(0.01)
            m.sample()
    return {k: v / m.seconds for k, v in m.joules.items()}, m.total_joules / m.seconds


def measure_per_query(fn, queries, repeats=1000, idle_watts=0.0):
    """Energy for ONE query.

    A single query finishes faster than the sensors update, so we run the
    work `repeats` times and divide. This is the difference between a real
    measurement and noise.
    """
    m = Meter()
    with m:
        for i in range(repeats):
            fn(queries[i % len(queries)])
            if i % 20 == 0:
                m.sample()

    gross = m.total_joules
    net = gross - idle_watts * m.seconds
    return {
        "joules_per_query": net / repeats,
        "wh_per_query": net / repeats / 3600.0,
        "ms_per_query": m.seconds * 1000.0 / repeats,
        "gross_joules": gross,
        "idle_joules": idle_watts * m.seconds,
        "seconds": m.seconds,
        "breakdown": {k: v / repeats for k, v in m.joules.items()},
    }


def run_trials(fn, queries, repeats=1000, trials=5, idle_watts=0.0, cooldown=60.0):
    """Repeat the whole measurement so you can report mean +/- CI."""
    results = []
    for t in range(trials):
        r = measure_per_query(fn, queries, repeats, idle_watts)
        results.append(r["joules_per_query"])
        print(f"  trial {t+1}/{trials}: {r['joules_per_query']*1000:.4f} mJ/query"
              f"  ({r['ms_per_query']:.2f} ms)")
        if t < trials - 1:
            time.sleep(cooldown)   # let the laptop cool; avoids throttling drift

    mean = statistics.mean(results)
    sd = statistics.stdev(results) if len(results) > 1 else 0.0
    ci95 = 1.96 * sd / (len(results) ** 0.5)
    return {
        "mean_joules": mean,
        "sd_joules": sd,
        "ci95_joules": ci95,
        "mean_wh": mean / 3600.0,
        "trials": results,
    }


if __name__ == "__main__":
    domains = find_rapl_domains()
    if not domains:
        print("No readable RAPL domains.")
        print("Fix with:  sudo chmod -R a+r /sys/class/powercap/intel-rapl")
        raise SystemExit(1)

    print("RAPL domains:", ", ".join(d.name for d in domains))

    m = Meter()
    print("GPU:", m.gpu.name if m.gpu else "not available")

    print("\nMeasuring idle for 5s...")
    per_domain, idle_w = measure_idle(5.0)
    for k, v in per_domain.items():
        print(f"  {k:10s} {v:6.2f} W")
    print(f"  {'TOTAL':10s} {idle_w:6.2f} W")

    print("\nMeasuring a dummy workload (matrix multiply)...")
    import random
    def work(_):
        n = 120
        a = [[random.random() for _ in range(n)] for _ in range(n)]
        b = [[random.random() for _ in range(n)] for _ in range(n)]
        return [[sum(a[i][k] * b[k][j] for k in range(n)) for j in range(n)]
                for i in range(n)]

    r = measure_per_query(work, [None], repeats=3, idle_watts=idle_w)
    print(f"  {r['joules_per_query']:.4f} J/call   {r['ms_per_query']:.1f} ms/call")
    print(f"  breakdown: " + ", ".join(f"{k}={v:.4f}J" for k, v in r['breakdown'].items()))
