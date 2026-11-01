from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np
from scipy.stats import chi2, kendalltau
from rayleigh_entropy import analog_eve_certificate, independent_eve_spec
from traces import session_traces
from bolt_policy import CalibrationLibrary, PolicyParams, select, outcome


def ljung_box_ok(p, lags=5, level=0.01):
    x = p - p.mean(); d = float(np.dot(x, x)); M = len(x)
    Q = M * (M + 2) * sum((float(np.dot(x[:-k], x[k:])) / d) ** 2 / (M - k) for k in range(1, lags + 1))
    return Q <= chi2.ppf(1 - level, lags), Q


def tau_to_r_table(seed=11):
    rng = np.random.default_rng(seed); rs = np.linspace(0, .98, 50); ts = []
    for r in rs:
        a = (rng.normal(size=20000) + 1j * rng.normal(size=20000)) / math.sqrt(2)
        e = math.sqrt(r) * a + math.sqrt(1 - r) * (rng.normal(size=20000) + 1j * rng.normal(size=20000)) / math.sqrt(2)
        ts.append(kendalltau(np.abs(a) ** 2, np.abs(e) ** 2)[0])
    return rs, np.maximum.accumulate(np.array(ts))


def eve_certificate_from_survey(path, eps_E):
    import csv
    rows = list(csv.DictReader(open(path)))
    pos = sorted({r['position_id'] for r in rows}); S = len(pos)
    rs, ts = tau_to_r_table(); worst = 0.0; per = {}
    for p in pos:
        sub = [r for r in rows if r['position_id'] == p]
        xa = np.array([float(r['alice_rssi_dbm']) for r in sub]); xe = np.array([float(r['probe_rssi_dbm']) for r in sub])
        t = kendalltau(xa, xe)[0]; m = len(sub) // 2
        tu = min(t + math.sqrt(2 * math.log(S / eps_E) / m), ts.max())
        rho = math.sqrt(float(np.interp(tu, ts, rs)))
        per[p] = {'n': len(sub), 'tau_hat': float(t), 'rho_upper': rho}; worst = max(worst, rho)
    return worst, per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('trace'); ap.add_argument('--survey'); ap.add_argument('--rho-ae', type=float)
    ap.add_argument('--qual', type=float, default=0.2); ap.add_argument('--rssi-step', type=float, default=1.0)
    ap.add_argument('--eps-src', type=float, default=1e-9); ap.add_argument('--eps-E', type=float, default=1e-6)
    ap.add_argument('--N', type=int, default=8192); ap.add_argument('--NP', type=int, default=512)
    ap.add_argument('--policy', default='bolt'); ap.add_argument('--library'); ap.add_argument('--out', default='measured_report.json')
    a = ap.parse_args()
    import csv
    rows = list(csv.DictReader(open(a.trace)))
    pa = 10 ** (np.array([float(r['alice_rssi_dbm']) for r in rows]) / 10)
    pb = 10 ** (np.array([float(r['bob_rssi_dbm']) for r in rows]) / 10)
    nq = int(a.qual * len(rows)); rep = {'n_rows': len(rows), 'qualification_rows': nq}
    # 1a independence / thinning
    thin = 1
    while True:
        ok, Q = ljung_box_ok(pa[:nq:thin])
        if ok or len(pa[:nq:thin]) < 200: break
        thin += 1
    rep['independence'] = {'thinning': thin, 'ljung_box_Q': float(Q), 'passed': bool(ok)}
    if not ok:
        rep['result'] = 'fail-closed: block independence not qualified'; Path(a.out).write_text(json.dumps(rep, indent=1)); print(rep); return
    qa, qb = pa[:nq:thin], pb[:nq:thin]
    # 1b scale certificate (Alice) and nominal scales (Bob's affects only BMR)
    M = len(qa); S = float(qa.sum()); mhat = S / M
    lo = 2 * S / chi2.ppf(1 - a.eps_src / 2, 2 * M) / mhat; hi = 2 * S / chi2.ppf(a.eps_src / 2, 2 * M) / mhat
    g = 10 ** (a.rssi_step / 20); mu = (lo / g, hi * g)
    rep['scale'] = {'samples': M, 'mu_L': mu[0], 'mu_U': mu[1], 'eps_src': a.eps_src}
    # 1c Eve certificate
    if a.survey:
        rho, per = eve_certificate_from_survey(a.survey, a.eps_E); rep['eve'] = {'rho_upper': rho, 'eps_E': a.eps_E, 'positions': per}
    elif a.rho_ae is not None:
        rho = a.rho_ae; rep['eve'] = {'rho_upper': rho, 'source': 'user-supplied'}
    else:
        rep['result'] = 'fail-closed: no Eve certificate'; Path(a.out).write_text(json.dumps(rep, indent=1)); print(rep); return
    hs = {H: float(analog_eve_certificate(H, eve_spec=independent_eve_spec(rho, 1), alice_scale_ratio_bounds=mu).h_min_per_block) for H in (1, 2)}
    rep['h_min'] = hs
    # 2 sessions
    lib = CalibrationLibrary.load(a.library) if a.library else CalibrationLibrary.load()
    A = pa[nq::thin] / mhat; B = pb[nq::thin] / float(qb.mean())
    sess = []
    for k in range(len(A) // a.N):
        x = session_traces(A[k * a.N:(k + 1) * a.N], B[k * a.N:(k + 1) * a.N], 4242 + k, a.NP)
        tb = {1: x[1], 2: x[2]}
        ch = select(lib, {1: x[1]['z'], 2: x[2]['z']}, hs, a.N - a.NP, a.policy, PolicyParams())
        d = outcome(ch, tb)
        sess.append({'H': ch.H, 'cap': ch.cap_bits, 'predicted_accept': ch.predicted_accept, 'delivered': d,
                     'pilot_z': {H: x[H]['z'] for H in (1, 2)}, 'raw_bmr': {H: x[H]['raw_bmr'] for H in (1, 2)}})
    rep['sessions'] = sess
    rep['rate_bit_per_block'] = float(np.mean([s['delivered'] for s in sess]) / a.N) if sess else 0.0
    rep['rate_bit_per_channel_use'] = rep['rate_bit_per_block'] / 2
    rep['note'] = 'library is the simulated Rayleigh library unless --library points to an in-situ library'
    Path(a.out).write_text(json.dumps(rep, indent=1)); print(json.dumps({k: v for k, v in rep.items() if k != 'sessions'}, indent=1))


if __name__ == '__main__':
    main()
