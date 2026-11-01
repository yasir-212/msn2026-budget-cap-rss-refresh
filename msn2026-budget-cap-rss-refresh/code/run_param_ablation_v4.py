"""Selection-parameter ablations on the main-grid sessions (exact replay, v4).

Varies one parameter at a time around the pre-specified configuration
(K=80, Gaussian kernel, alpha=0.05, uniform-over-caps Hoeffding LCB,
eps_PA=2^-64) and reports grid/feasible means for BOLT, BOLT-LCB and q99.
The pre-specified values are NOT re-tuned on these results.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from bolt_policy import CalibrationLibrary, PolicyParams
from run_main_grid_v4 import load_traces, h_table, evaluate

ROOT = Path(__file__).resolve().parent
POL = [('BOLT', 'bolt', None, {}), ('BOLT-LCB', 'bolt_lcb', None, {}), ('q99', 'q99', None, {}), ('q95', 'q95', None, {})]


def summ(rows):
    feas = [r for r in rows if r['policies']['oracle']['mean_rate'] > 0]
    o = {}
    for nm in ('BOLT', 'BOLT-LCB', 'q99', 'q95', 'oracle'):
        v = np.array([r['policies'][nm]['mean_rate'] for r in rows]); vf = np.array([r['policies'][nm]['mean_rate'] for r in feas])
        o[nm] = {'grid_mean': float(v.mean()), 'feasible_mean': float(vf.mean()), 'best': float(v.max())}
    return o


def main():
    tr = load_traces(); lib = CalibrationLibrary.load(); hs = h_table()
    base = PolicyParams()
    variants = [('baseline', {})]
    variants += [(f'K={k}', {'K': k}) for k in (20, 40, 160, 420)]
    variants += [('kernel=uniform', {'kernel': 'uniform'})]
    variants += [(f'alpha={a}', {'alpha': a}) for a in (0.01, 0.10, 0.25)]
    variants += [('LCB two-sided bracket', {'one_sided': False}), ('LCB two-sided pointwise (orig. Eq.12)', {'lcb': 'hoeffding', 'one_sided': False}),
                 ('LCB two-sided Bonferroni', {'lcb': 'hoeffding_uniform', 'one_sided': False})]
    variants += [(f'epsPA=2^-{e}', {'eps_pa': 2.0 ** -e}) for e in (32, 128)]
    out = {}
    for name, over in variants:
        p = PolicyParams(**{**base.__dict__, **over})
        rows = evaluate(tr, lib, hs, params=p, policies=POL)
        out[name] = summ(rows)
        print(f"{name:16s} " + ' '.join(f"{k}:{v['grid_mean']:.4f}/{v['feasible_mean']:.4f}/{v['best']:.3f}" for k, v in out[name].items()), flush=True)
    (ROOT / 'results_param_ablation_v4.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
