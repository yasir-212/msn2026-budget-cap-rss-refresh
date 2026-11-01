from __future__ import annotations
import json, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon
from rayleigh_entropy import simulate_correlated_rayleigh_blocks, independent_eve_spec, analog_eve_certificate, gaussian_cmi_upper_bound
from traces import session_traces
from bolt_policy import CalibrationLibrary, PolicyParams, select, outcome, oracle, global_fixed_caps, vl_length, vl_select_H

ROOT = Path(__file__).resolve().parent
N = 8192; NP = 512; NPAY = N - NP; TRIALS = 20
RABS = [.95, .97, .98, .99, .995, .9975]; RAES = [.1, .2, .3, .35, .4, .5, .7]
TRACE_FILE = ROOT / 'eval_traces_v4.json'


def seed_of(rab, t):
    return 55_000_000 + int(round(rab * 10000)) * 1000 + t


def _trace_job(args):
    rab, t = args
    s = seed_of(rab, t)
    a, b, _ = simulate_correlated_rayleigh_blocks(N, rab, 0.1, s)
    tr = session_traces(a, b, s, NP)
    return {'rho_ab': rab, 't': t, 'seed': s, 'H1': tr[1], 'H2': tr[2]}


def load_traces(workers=2):
    if TRACE_FILE.exists():
        return json.loads(TRACE_FILE.read_text())
    jobs = [(rab, t) for rab in RABS for t in range(TRIALS)]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        rows = list(ex.map(_trace_job, jobs, chunksize=2))
    TRACE_FILE.write_text(json.dumps(rows))
    return rows


def h_table(scale=(1.0, 1.0), raes=RAES):
    return {rae: {H: float(analog_eve_certificate(H, eve_spec=independent_eve_spec(rae, 1),
                                                  alice_scale_ratio_bounds=scale).h_min_per_block)
                  for H in (1, 2)} for rae in raes}


POLICIES = [  # name, policy, fixed_H, params-override
    ('BOLT', 'bolt', None, {}),
    ('BOLT-LCB', 'bolt_lcb', None, {}),
    ('BOLT-1s', 'bolt_os', None, {}),
    ('LCB-2s-bracket', 'bolt_lcb', None, {'one_sided': False}),
    ('LCB-2s-pointwise', 'bolt_lcb', None, {'lcb': 'hoeffding', 'one_sided': False}),
    ('q95', 'q95', None, {}),
    ('q99', 'q99', None, {}),
    ('BOLT(H=2)', 'bolt', 2, {}),
    ('BOLT-LCB(H=2)', 'bolt_lcb', 2, {}),
    ('q95(H=2)', 'q95', 2, {}),
    ('q99(H=2)', 'q99', 2, {}),
    ('fixedcap(H=2)', 'fixedcap', 2, {}),
]


def evaluate(traces, lib, hs, params=PolicyParams(), policies=POLICIES, raes=RAES, rabs=RABS, keep_sessions=False):
    fixed = global_fixed_caps(lib, (1, 2), sorted({hs[r][H] for r in raes for H in (1, 2)}), NPAY, params.eps_pa)
    rows = []
    for rab in rabs:
        tl = [x for x in traces if x['rho_ab'] == rab]
        for rae in raes:
            h = hs[rae]
            row = {'rho_ab': rab, 'rho_ae': rae, 'policies': {}}
            for name, pol, fH, over in policies:
                p = PolicyParams(**{**params.__dict__, **over})
                sess = []
                for x in tl:
                    tb = {1: x['H1'], 2: x['H2']}
                    ch = select(lib, {1: tb[1]['z'], 2: tb[2]['z']}, h, NPAY, pol, p, fixed_H=fH, fixed_caps=fixed)
                    d = outcome(ch, tb)
                    sess.append({'H': ch.H, 'cap': ch.cap_bits, 'score': ch.predicted_accept, 'ghat': ch.point_accept,
                                 'acc': int(d > 0), 'delivered': d})
                rr = np.array([s['delivered'] for s in sess], float) / N
                row['policies'][name] = {
                    'mean_rate': float(rr.mean()), 'positive': int((rr > 0).sum()),
                    'mean_H': float(np.mean([s['H'] for s in sess])),
                    'mean_cap': float(np.mean([s['cap'] for s in sess])),
                    'mean_score': float(np.mean([s['score'] for s in sess])),
                    'mean_ghat': float(np.mean([s['ghat'] for s in sess])),
                    'accept_rate': float(np.mean([s['acc'] for s in sess])),
                    'rates': rr.tolist(), **({'sessions': sess} if keep_sessions else {})}
            # variable-length charging comparisons (computed from the same traces)
            vl, bvl = [], []
            for x in tl:
                tb = {1: x['H1'], 2: x['H2']}; z = {1: tb[1]['z'], 2: tb[2]['z']}
                Hv = vl_select_H(lib, z, h, NPAY, params)
                vl.append(vl_length(NPAY, h[Hv], tb[Hv]['L'], Hv, NPAY * Hv, params.eps_pa) if (Hv and tb[Hv]['C']) else 0)
                ch = select(lib, z, h, NPAY, 'bolt', params)
                t = tb[ch.H]; ok = ch.cap_bits > 0 and t['C'] and t['Lreq'] <= ch.cap_bits
                bvl.append(vl_length(NPAY, h[ch.H], t['L'], ch.H, ch.cap_bits, params.eps_pa) if ok else 0)
            for name, v in (('VL', vl), ('BOLT+VL', bvl)):
                rr = np.array(v, float) / N
                row['policies'][name] = {'mean_rate': float(rr.mean()), 'positive': int((rr > 0).sum()), 'rates': rr.tolist()}
            for name, fH in (('oracle', None), ('oracle(H=2)', 2)):
                rr = np.array([oracle({1: x['H1'], 2: x['H2']}, h, NPAY, fH, params.eps_pa) for x in tl], float) / N
                row['policies'][name] = {'mean_rate': float(rr.mean()), 'positive': int((rr > 0).sum()), 'rates': rr.tolist()}
            row['cmi_upper'] = NPAY / N * gaussian_cmi_upper_bound(rab, independent_eve_spec(rae, 1))
            rows.append(row)
    return rows


def boot_ci(x, seed, n=20000):
    x = np.asarray(x, float); rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), size=(n, len(x)))].mean(1)
    return [float(np.quantile(m, .025)), float(np.quantile(m, .975))]


def summarize(rows, names):
    S = {}
    feas = [r for r in rows if r['policies']['oracle']['mean_rate'] > 0]
    S['n_cells'] = len(rows); S['n_feasible_cells'] = len(feas)
    for nm in names:
        v = np.array([r['policies'][nm]['mean_rate'] for r in rows])
        vf = np.array([r['policies'][nm]['mean_rate'] for r in feas])
        S[nm] = {'grid_mean': float(v.mean()), 'grid_sum': float(v.sum()), 'best': float(v.max()),
                 'feasible_mean': float(vf.mean()), 'positive_cells': int((v > 0).sum())}
    def gain(a, b, key='grid_mean'):
        return 100 * (S[a][key] - S[b][key]) / max(S[b][key], 1e-15)
    S['gains'] = {f'{a}_vs_{b}': gain(a, b) for a, b in [
        ('BOLT', 'q99'), ('BOLT-LCB', 'q99'), ('BOLT', 'q95'), ('BOLT-LCB', 'q95'), ('BOLT-LCB', 'BOLT'),
        ('BOLT(H=2)', 'q99(H=2)'), ('BOLT-LCB(H=2)', 'q99(H=2)'), ('BOLT(H=2)', 'fixedcap(H=2)'),
        ('BOLT-LCB(H=2)', 'fixedcap(H=2)'), ('q95', 'q99')]}
    # paired, per-cell head-to-head among cells where someone is positive
    def h2h(a, b):
        w = l = t = 0
        for r in rows:
            x = r['policies'][a]['mean_rate']; y = r['policies'][b]['mean_rate']
            if max(x, y) <= 0: continue
            if x > y + 1e-12: w += 1
            elif y > x + 1e-12: l += 1
            else: t += 1
        return {'wins': w, 'losses': l, 'ties': t}
    S['h2h'] = {k: h2h(*k.split('|')) for k in ['BOLT|q99', 'BOLT-LCB|q99', 'BOLT-LCB|BOLT', 'BOLT|q95', 'BOLT-LCB|q95',
                                                'BOLT(H=2)|fixedcap(H=2)', 'BOLT-LCB(H=2)|fixedcap(H=2)']}
    # session-level paired statistics pooled over feasible cells (cell-stratified bootstrap)
    def paired(a, b, seed):
        d = np.concatenate([np.array(r['policies'][a]['rates']) - np.array(r['policies'][b]['rates']) for r in feas])
        try:
            p = float(wilcoxon(d, alternative='greater').pvalue)
        except ValueError:
            p = 1.0
        return {'mean_diff': float(d.mean()), 'ci95': boot_ci(d, seed), 'wilcoxon_p': p, 'n': int(len(d))}
    def clustered(a, b, seed):
        from collections import defaultdict
        acc = defaultdict(list)
        for r in feas:
            for t, (x, y) in enumerate(zip(r['policies'][a]['rates'], r['policies'][b]['rates'])):
                acc[(r['rho_ab'], t)].append(x - y)
        d = np.array([np.mean(v) for v in acc.values()])
        try: p = float(wilcoxon(d, alternative='greater').pvalue)
        except ValueError: p = 1.0
        return {'mean_diff': float(d.mean()), 'ci95': boot_ci(d, seed), 'wilcoxon_p': p, 'n_clusters': int(len(d))}
    S['paired_clustered'] = {f'{a}-{b}': clustered(a, b, 300 + i) for i, (a, b) in enumerate(
        [('BOLT', 'q99'), ('BOLT-LCB', 'q99'), ('BOLT(H=2)', 'q99(H=2)')])}
    S['paired_feasible'] = {f'{a}-{b}': paired(a, b, 100 + i) for i, (a, b) in enumerate(
        [('BOLT', 'q99'), ('BOLT-LCB', 'q99'), ('BOLT-LCB', 'BOLT'), ('BOLT(H=2)', 'q99(H=2)'), ('BOLT', 'q95'),
         ('BOLT-LCB', 'q95'), ('BOLT-LCB(H=2)', 'q99(H=2)'), ('BOLT', 'BOLT-LCB')])}
    best = max(rows, key=lambda r: r['policies']['BOLT']['mean_rate'])
    S['best_cell'] = {'rho_ab': best['rho_ab'], 'rho_ae': best['rho_ae'],
                      **{nm: best['policies'][nm]['mean_rate'] for nm in names}}
    hp = next(r for r in rows if r['rho_ab'] == .9975 and r['rho_ae'] == .1)
    S['headline_point'] = {nm: hp['policies'][nm]['mean_rate'] for nm in names}
    S['headline_point_paired'] = {k: (lambda d: {'mean': float(d.mean()), 'ci95': boot_ci(d, 7),
                                               'wilcoxon_p': float(wilcoxon(d, alternative='greater').pvalue) if np.any(d != 0) else 1.0})(
        np.array(hp['policies'][a]['rates']) - np.array(hp['policies'][b]['rates']))
        for k, (a, b) in {'BOLT(H=2)-q99(H=2)': ('BOLT(H=2)', 'q99(H=2)'), 'BOLT-LCB(H=2)-q99(H=2)': ('BOLT-LCB(H=2)', 'q99(H=2)'),
                          'BOLT-q99': ('BOLT', 'q99'), 'BOLT-LCB-q99': ('BOLT-LCB', 'q99')}.items()}
    hi = [r for r in rows if r['rho_ab'] in (.995, .9975)]
    S['high_reciprocity_sums'] = {nm: float(sum(r['policies'][nm]['mean_rate'] for r in hi)) for nm in names}
    return S


NAMES = [p[0] for p in POLICIES] + ['VL', 'BOLT+VL', 'oracle', 'oracle(H=2)']

if __name__ == '__main__':
    scale = (1.0, 1.0)
    tag = 'exact'
    if len(sys.argv) > 1 and sys.argv[1] == 'finite':
        scale = (0.894, 1.125); tag = 'finite'
    traces = load_traces()
    lib = CalibrationLibrary.load()
    hs = h_table(scale)
    rows = evaluate(traces, lib, hs, keep_sessions=(tag == 'exact'))
    S = summarize(rows, NAMES)
    out = {'version': 'bolt-rss-v4-main-grid', 'scale': list(scale), 'h_table': {str(k): v for k, v in hs.items()},
           'params': PolicyParams().__dict__, 'summary': S, 'rows': rows}
    (ROOT / f'results_main_grid_v4_{tag}.json').write_text(json.dumps(out))
    print(json.dumps({k: S[k] for k in ['n_feasible_cells', 'gains', 'h2h', 'best_cell', 'headline_point', 'high_reciprocity_sums']}, indent=1))
    for nm in NAMES:
        print(f"{nm:16s} grid_mean={S[nm]['grid_mean']:.4f} feas_mean={S[nm]['feasible_mean']:.4f} best={S[nm]['best']:.3f} pos_cells={S[nm]['positive_cells']}")
    print(json.dumps(S['paired_feasible'], indent=1)); print(json.dumps(S['headline_point_paired'], indent=1))
