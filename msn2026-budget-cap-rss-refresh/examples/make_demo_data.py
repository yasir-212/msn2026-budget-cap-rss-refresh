"""Create a small synthetic RSS trace and Eve-survey file to try measured_replay.py.

    python examples/make_demo_data.py
    python code/measured_replay.py examples/demo_trace.csv --survey examples/demo_survey.csv --qual 0.17
Replace these files by your recorded RSSI traces (same columns) to run on measured data.
"""
import csv, math, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
from channel_emulator import gen

out = Path(__file__).resolve().parent
a, b = gen(8192 * 3, 7, rab_direct=0.9975, rssi_step=1.0)
with open(out / 'demo_trace.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['t_s', 'alice_rssi_dbm', 'bob_rssi_dbm'])
    for i, (x, y) in enumerate(zip(a, b)):
        w.writerow([round(i * 0.05, 3), round(10 * math.log10(x) - 60, 1), round(10 * math.log10(y) - 62, 1)])
rng = np.random.default_rng(1)
with open(out / 'demo_survey.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['alice_rssi_dbm', 'probe_rssi_dbm', 'position_id'])
    for p in range(3):
        ha = (rng.normal(size=20000) + 1j * rng.normal(size=20000)) / math.sqrt(2)
        he = 0.1 * ha + math.sqrt(0.99) * (rng.normal(size=20000) + 1j * rng.normal(size=20000)) / math.sqrt(2)
        for x, y in zip(ha, he):
            w.writerow([round(10 * math.log10(abs(x) ** 2), 2), round(10 * math.log10(abs(y) ** 2) + 5, 2), p])
print('wrote', out / 'demo_trace.csv', 'and', out / 'demo_survey.csv')
