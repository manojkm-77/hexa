"""Verify multiple fixes in the hexaverse codebase."""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# ── 1. Config defaults ──────────────────────────────────────────────────
from config import Config

cfg = Config()
checks = [
    ("turbulence_rms_deg == 0.0", cfg.turbulence_rms_deg == 0.0, cfg.turbulence_rms_deg),
    ("exposure_rate_hz == 0.0",   cfg.exposure_rate_hz == 0.0,   cfg.exposure_rate_hz),
    ("occlusion_duration_s == 0.0", cfg.occlusion_duration_s == 0.0, cfg.occlusion_duration_s),
    ("false_beacon_count == 0",   cfg.false_beacon_count == 0,   cfg.false_beacon_count),
]
for label, ok, val in checks:
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] Config default {label}  (got {val!r})")
    assert ok, f"FAIL: {label}"

# ── 2. TrackerOutput API ────────────────────────────────────────────────
from detect import Tracker, TrackerOutput, TrackerState

tracker = Tracker()
tracker.start()
assert tracker.state == TrackerState.SEARCHING, f"Expected SEARCHING, got {tracker.state}"

frame = np.zeros((720, 1280), dtype=np.uint8)
output = tracker.track(frame)

assert isinstance(output, TrackerOutput), f"track() returned {type(output)}, not TrackerOutput"
for attr in ("state", "estimated_x", "estimated_y", "confidence", "detected"):
    assert hasattr(output, attr), f"TrackerOutput missing attribute: {attr}"
    print(f"  [PASS] TrackerOutput has '{attr}' = {getattr(output, attr)!r}")
print(f"  [PASS] TrackerOutput API is correct (state={output.state.name})")

# ── 3. RandomManeuvering.reset() determinism ────────────────────────────
from sim import RandomManeuvering, TargetState

rm = RandomManeuvering(seed=42)
state = TargetState()
dt, t = 0.1, 0.0

# First run: 10 steps
positions_a = []
for i in range(10):
    state = rm.update(state, dt, t)
    positions_a.append((state.azimuth_deg, state.elevation_deg))
    t += dt

# Reset and second run
rm.reset()
state = TargetState()
t = 0.0
positions_b = []
for i in range(10):
    state = rm.update(state, dt, t)
    positions_b.append((state.azimuth_deg, state.elevation_deg))
    t += dt

assert positions_a == positions_b, (
    f"RandomManeuvering.reset() is not deterministic!\n  Run A: {positions_a}\n  Run B: {positions_b}"
)
print(f"  [PASS] RandomManeuvering.reset() is deterministic (10 steps match)")

print("\nAll checks passed.")
