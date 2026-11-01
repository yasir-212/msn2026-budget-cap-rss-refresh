"""Protocol-parameter ablations requiring new calibration libraries (v4).

For each configuration a fresh 840-record (or 1260 for H_max=3) calibration
library is built with the same construction and seed ranges, and the policies
are evaluated on new evaluation sessions (rho_AB in {0.99,0.995,0.9975},
20 trials, all seven Eve certificates).  One parameter changes at a time:
pilot length N_P, block length N (N_P = N/16), pilot CP confidence,
confirmation-tag length, and the resolution set H_max=3.
"""
from __future__ import annotations
import json, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from rayleigh_entropy import simulate_correlated_rayleigh_blocks, analog_eve_certificate, independent_eve_spec
from traces import session_traces
from bolt_policy import CalibrationLibrary, PolicyParams, choose_cap, oracle, outcome

ROOT = Path(__file__).resolve().parent
CAL_RABS = (0.990, 0.9925, 0.995, 0.996, 0.9975, 0.9985, 0.999)
EVAL_RABS = (0.99, 0.995, 0.9975)
RAES = (.1, .2, .3, .35, .4, .5, .7)


def cal_job(args):
    cfg, H, ri, rab, t = args
    s = 77_000_000 + H * 2_000_000 + ri * 10000 + t
    a, b, _ = simulate_correlated_rayleigh_blocks(cfg['N'], rab, 0.1, s)
    tr = session_traces(a, b, s - 1000 * H, n_pilot=cfg['NP'], Hs=(H,), confidence=cfg['conf'],
                        confirm_bits=cfg['cbits'], pilot_Hs=cfg['Hs'])
    r = tr[H]
    return {'H': H, 'rho_ab_cal': rab, 'pilot_bmr_upper': r['z'], 'converged': int(r['C']), 'required_cap': r['Lreq'], 'leak_bits': r['L']}


def eval_job(args):
    cfg, rab, t = args
    s = 55_000_000 + int(round(rab * 10000)) * 1000 + t
    a, b, _ = simulate_correlated_rayleigh_blocks(cfg['N'], rab, 0.1, s)
    tr = session_traces(a, b, s, n_pilot=cfg['NP'], Hs=cfg['Hs'], confidence=cfg['conf'],
                        confirm_bits=cfg['cbits'], pilot_Hs=cfg['Hs'])
    return {'rho_ab': rab, 't': t, 'tr': {H: tr[H] for H in cfg['Hs']}}


def run_cfg(name, cfg, ex):
    jobs = [(cfg, H, ri, rab, t) for H in cfg['Hs'] for ri, rab in enumerate(CAL_RABS) for t in range(60)]
    recs = list(ex.map(cal_job, jobs, chunksize=8))
    lib = CalibrationLibrary(recs)
    ev = list(ex.map(eval_job, [(cfg, rab, t) for rab in EVAL_RABS for t in range(20)], chunksize=2))
    N = cfg['N']; npay = N - cfg['NP']
    hs = {rae: {H: float(analog_eve_certificate(H, eve_spec=independent_eve_spec(rae, 1)).h_min_per_block) for H in cfg['Hs']} for rae in RAES}
    res = {}
    for pol in ('bolt', 'bolt_lcb', 'q99', 'oracle'):
        cells = []
        for rab in EVAL_RABS:
            for rae in RAES:
                d = []
                for x in ev:
                    if x['rho_ab'] != rab: continue
                    tb = {int(k): v for k, v in x['tr'].items()}
                    if pol == 'oracle':
                        d.append(oracle(tb, hs[rae], npay)); continue
                    chs = [choose_cap(lib, H, tb[H]['z'], hs[rae][H], npay, pol, PolicyParams()) for H in cfg['Hs']]
                    ch = max(chs, key=lambda c: (c.objective, -c.H))
                    d.append(outcome(ch, tb))
                cells.append({'rho_ab': rab, 'rho_ae': rae, 'rate': float(np.mean(d) / N),
                              'Hsel': None})
        res[pol] = {'mean_rate_over_cells': float(np.mean([c['rate'] for c in cells])),
                    'best': float(max(c['rate'] for c in cells)), 'cells': cells}
    print(f"{name:14s} " + ' '.join(f"{p}:{v['mean_rate_over_cells']:.4f}/{v['best']:.3f}" for p, v in res.items()), flush=True)
    return res


BASE = {'N': 8192, 'NP': 512, 'conf': 0.99, 'cbits': 64, 'Hs': (1, 2)}
CFGS = [('baseline', {})] + [(f'NP={v}', {'NP': v}) for v in (128, 256, 1024)] \
     + [(f'N={v}', {'N': v, 'NP': v // 16}) for v in (2048, 4096, 16384)] \
     + [(f'conf={v}', {'conf': v}) for v in (0.9, 0.999)] + [('cbits=32', {'cbits': 32})] \
     + [('Hmax=3', {'Hs': (1, 2, 3)})]

if __name__ == '__main__':
    only = sys.argv[1:] or [c[0] for c in CFGS]
    path = ROOT / 'results_protocol_ablation_v4.json'
    out = json.loads(path.read_text()) if path.exists() else {}
    with ProcessPoolExecutor(max_workers=2) as ex:
        for name, over in CFGS:
            if name not in only or name in out: continue
            cfg = {**BASE, **over}
            out[name] = {'cfg': {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()}, 'res': run_cfg(name, cfg, ex)}
            path.write_text(json.dumps(out, indent=1))
