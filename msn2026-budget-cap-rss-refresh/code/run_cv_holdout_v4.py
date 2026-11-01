from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from bolt_policy import CalibrationLibrary, PolicyParams, choose_cap, global_fixed_caps, delivered_length_for_cap
from run_main_grid_v4 import h_table, RAES, N, NPAY

ROOT = Path(__file__).resolve().parent
POLS = [('BOLT', 'bolt', {}), ('BOLT-LCB', 'bolt_lcb', {}), ('LCB-2s-bracket', 'bolt_lcb', {'one_sided': False}),
        ('LCB-2s-pointwise', 'bolt_lcb', {'lcb': 'hoeffding', 'one_sided': False}),
        ('q95', 'q95', {}), ('q99', 'q99', {}), ('fixedcap', 'fixedcap', {})]


def eval_split(train_recs, test_recs, H, hs, raes=RAES, params=PolicyParams()):
    lib = CalibrationLibrary(train_recs)
    fixed = global_fixed_caps(lib, (H,), [hs[r][H] for r in raes], NPAY)
    res = {}
    for rae in raes:
        h = hs[rae][H]
        out = {nm: {'deliv': [], 'score': [], 'ghat': [], 'recacc': []} for nm, _, _ in POLS}
        out['oracle'] = {'deliv': []}
        for r in test_recs:
            z = float(r['pilot_bmr_upper']); C = bool(r['converged']); L = int(r['required_cap'])
            for nm, pol, over in POLS:
                p = PolicyParams(**{**params.__dict__, **over})
                ch = choose_cap(lib, H, z, h, NPAY, pol, p, fixed)
                acc = ch.cap_bits > 0 and C and L <= ch.cap_bits
                out[nm]['deliv'].append(ch.delivered_if_accept if acc else 0)
                if ch.cap_bits > 0:
                    out[nm]['score'].append(ch.predicted_accept); out[nm]['ghat'].append(ch.point_accept)
                    out[nm]['recacc'].append(int(acc))
            out['oracle']['deliv'].append(delivered_length_for_cap(NPAY, h, L, NPAY * H) if C else 0)
        res[rae] = out
    return res


def kfold(n, k, rng):
    idx = rng.permutation(n)
    return [idx[i::k] for i in range(k)]


def main():
    recs = json.loads((ROOT / 'protocol_calibration_v4.json').read_text())['records']
    hs = h_table()
    out = {'version': 'bolt-rss-v4-cv', 'reps': 20, 'folds': 5, 'cv': {}}
    names = [p[0] for p in POLS] + ['oracle']
    for H in (1, 2):
        R = [r for r in recs if int(r['H']) == H]
        other = [r for r in recs if int(r['H']) != H]
        per_rep = {rae: {nm: [] for nm in names} for rae in RAES}
        cover = {nm: {'score': [], 'ghat': [], 'recacc': []} for nm, _, _ in POLS}
        for rep in range(20):
            rng = np.random.default_rng(9000 + 31 * rep + H)
            acc = {rae: {nm: [] for nm in names} for rae in RAES}
            for f in kfold(len(R), 5, rng):
                fs = set(int(i) for i in f)
                train = [r for i, r in enumerate(R) if i not in fs] + other
                test = [R[i] for i in sorted(fs)]
                res = eval_split(train, test, H, hs)
                for rae in RAES:
                    for nm in names:
                        acc[rae][nm].extend(res[rae][nm]['deliv'])
                    for nm, _, _ in POLS:
                        for k in ('score', 'ghat', 'recacc'):
                            cover[nm][k].extend(res[rae][nm][k])
            for rae in RAES:
                for nm in names:
                    per_rep[rae][nm].append(float(np.mean(acc[rae][nm]) / N))
        summ = {}
        for rae in RAES:
            summ[str(rae)] = {nm: {'mean': float(np.mean(v)), 'ci95_reps': [float(np.quantile(v, .025)), float(np.quantile(v, .975))]}
                              for nm, v in per_rep[rae].items()}
        avg = {nm: float(np.mean([summ[str(rae)][nm]['mean'] for rae in RAES])) for nm in names}
        cov = {nm: {'mean_score': float(np.mean(c['score'])), 'mean_ghat': float(np.mean(c['ghat'])),
                    'realized_accept': float(np.mean(c['recacc'])), 'n': len(c['recacc'])} for nm, c in cover.items()}
        # reliability by score decile for BOLT vs BOLT-LCB
        rel = {}
        for nm in ('BOLT', 'BOLT-LCB'):
            s = np.array(cover[nm]['score']); a = np.array(cover[nm]['recacc'])
            bins = np.linspace(0, 1, 11); rows = []
            for lo, hi in zip(bins[:-1], bins[1:]):
                m = (s >= lo) & (s < hi if hi < 1 else s <= hi)
                if m.sum() >= 20:
                    rows.append({'lo': float(lo), 'hi': float(hi), 'n': int(m.sum()), 'score': float(s[m].mean()), 'realized': float(a[m].mean())})
            rel[nm] = rows
        out['cv'][str(H)] = {'per_rae': summ, 'avg_over_rae': avg, 'coverage': cov, 'reliability': rel}
        print('H', H, json.dumps(avg), flush=True)
        print(' coverage', json.dumps(cov), flush=True)
    # distribution shift (H=2 and H=1, all rae)
    shift = {}
    for H in (1, 2):
        R = [r for r in recs if int(r['H']) == H]
        low = [r for r in recs if float(r['rho_ab_cal']) <= .995]; high_t = [r for r in R if float(r['rho_ab_cal']) >= .9975]
        high = [r for r in recs if float(r['rho_ab_cal']) >= .9975]; low_t = [r for r in R if float(r['rho_ab_cal']) <= .995]
        for tag, tr, te in (('low_to_high', low, high_t), ('high_to_low', high, low_t)):
            res = eval_split(tr, te, H, hs)
            shift[f'H{H}_{tag}'] = {str(rae): {nm: float(np.mean(res[rae][nm]['deliv']) / N) for nm in names} for rae in RAES}
    out['shift'] = shift
    print('shift', json.dumps({k: v['0.2'] for k, v in shift.items()}), flush=True)
    # library-size curve, H=2, all rae averaged and rae=0.2
    H = 2; R = [r for r in recs if int(r['H']) == H]; other = [r for r in recs if int(r['H']) != H]
    sizes = [40, 80, 160, 240, 320]; curve = {}
    for n in sizes:
        vals = {nm: [] for nm in names}; vals02 = {nm: [] for nm in names}
        for rep in range(50):
            rng = np.random.default_rng(5000 + rep)
            idx = rng.permutation(len(R)); te = [R[i] for i in idx[:105]]; pool = idx[105:]
            tr = [R[i] for i in rng.choice(pool, size=min(n, len(pool)), replace=False)] + other
            res = eval_split(tr, te, H, hs)
            for nm in names:
                vals[nm].append(float(np.mean([np.mean(res[rae][nm]['deliv']) for rae in RAES]) / N))
                vals02[nm].append(float(np.mean(res[.2][nm]['deliv']) / N))
        curve[str(n)] = {nm: {'mean': float(np.mean(v)), 'q05': float(np.quantile(v, .05)), 'q95': float(np.quantile(v, .95)),
                              'mean_rae02': float(np.mean(vals02[nm]))} for nm, v in vals.items()}
        print('size', n, {nm: round(curve[str(n)][nm]['mean'], 4) for nm in names}, flush=True)
    out['size_curve'] = curve
    (ROOT / 'results_cv_holdout_v4.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
