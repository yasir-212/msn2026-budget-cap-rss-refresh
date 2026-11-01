from __future__ import annotations
import argparse, json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from rayleigh_entropy import simulate_correlated_rayleigh_blocks
from traces import session_traces

DEFAULT_RABS = (0.990, 0.9925, 0.995, 0.996, 0.9975, 0.9985, 0.999)


def one(args):
    rab, ri, t, N, NP, conf, cbits = args
    rows = []
    for H in (1, 2):
        # v1 used seed 77e6+H*2e6+ri*1e4+t for the H-th record; keep that exactly.
        s = 77_000_000 + H * 2_000_000 + ri * 10000 + t
        a, b, _ = simulate_correlated_rayleigh_blocks(N, rab, 0.1, s)
        tr = session_traces(a, b, s - 1000 * H, n_pilot=NP, Hs=(H,), confidence=conf, confirm_bits=cbits)
        r = tr[H]
        rows.append({'H': H, 'rho_ab_cal': rab, 'seed': s, 'pilot_bmr_upper': r['z'],
                     'pilot_bmr': float(tr['pilot'][H]['bmr']), 'raw_bmr': r['raw_bmr'],
                     'leak_bits': r['L'], 'required_cap': r['Lreq'], 'converged': int(r['C'])})
    return rows


def build(out: str, N=8192, NP=512, conf=0.99, cbits=64, rabs=DEFAULT_RABS, trials=60, workers=2):
    jobs = [(rab, ri, t, N, NP, conf, cbits) for ri, rab in enumerate(rabs) for t in range(trials)]
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, r in enumerate(ex.map(one, jobs, chunksize=4), 1):
            rows.extend(r)
    rows.sort(key=lambda r: (r['H'], r['pilot_bmr_upper'], r['rho_ab_cal'], r['seed']))
    meta = {'version': 'bolt-rss-protocol-cal-v4', 'n_blocks': N, 'n_pilot': NP, 'pilot_confidence': conf,
            'confirm_bits': cbits, 'rho_ab_grid': list(rabs), 'trials_per_rho': trials, 'records': rows,
            'definition': 'independent full-session calibration; required_cap = exact minimal hard cap'}
    Path(out).write_text(json.dumps(meta))
    return meta


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(Path(__file__).with_name('protocol_calibration_v4.json')))
    ap.add_argument('--N', type=int, default=8192); ap.add_argument('--NP', type=int, default=512)
    ap.add_argument('--conf', type=float, default=0.99); ap.add_argument('--cbits', type=int, default=64)
    ap.add_argument('--rabs', type=str, default=','.join(map(str, DEFAULT_RABS)))
    ap.add_argument('--trials', type=int, default=60); ap.add_argument('--workers', type=int, default=2)
    a = ap.parse_args()
    m = build(a.out, a.N, a.NP, a.conf, a.cbits, tuple(float(x) for x in a.rabs.split(',')), a.trials, a.workers)
    print('wrote', a.out, len(m['records']))
