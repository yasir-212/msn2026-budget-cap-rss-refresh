from __future__ import annotations
import json, math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from scipy.special import j0
from scipy.optimize import brentq
from scipy.stats import chi2
from rayleigh_entropy import analog_eve_certificate, independent_eve_spec
from channel_emulator import gen, rician_guessing_probability
from traces import session_traces
from bolt_policy import CalibrationLibrary, PolicyParams, select, outcome, oracle

ROOT = Path(__file__).resolve().parent
N = 8192; NP = 512; NPAY = N - NP
CAL_RABS = (0.990, 0.9925, 0.995, 0.996, 0.9975, 0.9985, 0.999)
EVAL_RABS = (0.995, 0.9975)
RAE = 0.2

SCEN = {
    'S0 ideal i.i.d. Rayleigh': {},
    'S1 TDD delay + 30 dB SNR': {'snr_db': 30.0, 'tdd': True},
    'S2 1-dB RSSI reporting': {'rssi_step': 1.0},
    'S2b 0.5-dB RSSI reporting': {'rssi_step': 0.5},
    'S3 +1 dB Bob gain offset': {'bob_gain_db': 1.0},
    'S4 Rician K=3 dB': {'K_db': 3.0},
}


def fdtau_for(r):
    return brentq(lambda x: j0(2 * math.pi * x) - r, 0.0, 0.38)


def make(sc, rab, seed):
    kw = {k: v for k, v in sc.items() if k != 'tdd'}
    if sc.get('tdd'):
        # choose the TDD delay so that the *measured* correlation equals rab after noise
        s2 = 10 ** (-sc['snr_db'] / 10)
        rt = min(rab * (1 + s2), 0.999999)
        return gen(N, seed, fd_tau=fdtau_for(rt), **kw)
    return gen(N, seed, rab_direct=rab, **kw)


def cal_job(args):
    name, rab, ri, t, H = args
    s = 88_000_000 + H * 2_000_000 + ri * 10000 + t
    a, b = make(SCEN[name], rab, s)
    tr = session_traces(a, b, s - 1000 * H, NP, Hs=(H,))
    r = tr[H]
    return {'H': H, 'rho_ab_cal': rab, 'pilot_bmr_upper': r['z'], 'converged': int(r['C']), 'required_cap': r['Lreq'], 'leak_bits': r['L']}


def eval_job(args):
    name, rab, t = args
    s = 66_000_000 + int(round(rab * 10000)) * 1000 + t
    a, b = make(SCEN[name], rab, s)
    tr = session_traces(a, b, s, NP)
    return {'rho_ab': rab, 't': t, 'H1': tr[1], 'H2': tr[2]}


def certificate(name):
    sc = SCEN[name]
    spec = independent_eve_spec(RAE, 1)
    if 'rssi_step' in sc:
        g = 10 ** (sc['rssi_step'] / 20)  # +-step/2 dB per-sample multiplicative bound
        return {H: float(analog_eve_certificate(H, eve_spec=spec, alice_scale_ratio_bounds=(1 / g, g)).h_min_per_block) for H in (1, 2)}, True, 'Prop. 1 with per-sample bound'
    if 'K_db' in sc:
        claimed = {H: float(analog_eve_certificate(H, eve_spec=spec).h_min_per_block) for H in (1, 2)}
        true = {}
        for H in (1, 2):
            p, ub, _ = rician_guessing_probability(H, sc['K_db'], RAE)
            true[H] = -math.log2(ub)
        return {'claimed_rayleigh': claimed, 'rician_certificate': true}, False, 'Rayleigh law violated'
    return {H: float(analog_eve_certificate(H, eve_spec=spec).h_min_per_block) for H in (1, 2)}, True, 'model holds'


def rates(ev, lib, hs):
    out = {}
    for pol in ('bolt', 'bolt_lcb', 'q99'):
        out[pol] = {}
        for rab in EVAL_RABS:
            d = []
            for x in ev:
                if x['rho_ab'] != rab: continue
                tb = {1: x['H1'], 2: x['H2']}
                ch = select(lib, {1: tb[1]['z'], 2: tb[2]['z']}, hs, NPAY, pol, PolicyParams())
                d.append(outcome(ch, tb))
            out[pol][str(rab)] = float(np.mean(d) / N)
    out['oracle'] = {str(rab): float(np.mean([oracle({1: x['H1'], 2: x['H2']}, hs, NPAY) for x in ev if x['rho_ab'] == rab]) / N) for rab in EVAL_RABS}
    return out


def main():
    lib_ray = CalibrationLibrary.load()
    res = {}
    with ProcessPoolExecutor(max_workers=2) as ex:
        for name in SCEN:
            recs = list(ex.map(cal_job, [(name, rab, ri, t, H) for H in (1, 2) for ri, rab in enumerate(CAL_RABS) for t in range(60)], chunksize=8))
            lib_situ = CalibrationLibrary(recs)
            ev = list(ex.map(eval_job, [(name, rab, t) for rab in EVAL_RABS for t in range(20)], chunksize=2))
            hs, valid, note = certificate(name)
            row = {'certificate_valid_as_rayleigh': valid, 'note': note,
                   'raw_bmr_H2': {str(rab): float(np.mean([x['H2']['raw_bmr'] for x in ev if x['rho_ab'] == rab])) for rab in EVAL_RABS},
                   'pilot_z_H2': {str(rab): float(np.mean([x['H2']['z'] for x in ev if x['rho_ab'] == rab])) for rab in EVAL_RABS}}
            if valid:
                row['h'] = hs
                row['rayleigh_library'] = rates(ev, lib_ray, hs)
                row['in_situ_library'] = rates(ev, lib_situ, hs)
            else:
                row['h_claimed'] = hs['claimed_rayleigh']; row['h_rician_cert'] = hs['rician_certificate']
                row['rayleigh_library_with_invalid_claim'] = rates(ev, lib_ray, hs['claimed_rayleigh'])
                row['in_situ_library_rician_certificate'] = rates(ev, lib_situ, hs['rician_certificate'])
            res[name] = row
            print(name, json.dumps({k: v for k, v in row.items()}), flush=True)
    # independence qualification: Ljung-Box style detection of block dependence
    dep = []
    rng = np.random.default_rng(3)
    from channel_emulator import sos
    for fdTs in (0.02, 0.05, 0.2, 0.5, 1.0, None):
        rej = 0; reps = 200; M = 2000; lags = 5
        for _ in range(reps):
            if fdTs is None:
                p = rng.exponential(1, M)
            else:
                p = np.abs(sos(np.random.default_rng(rng.integers(1 << 62)), np.arange(M) * fdTs)) ** 2
            x = p - p.mean(); ac = [np.dot(x[:-k], x[k:]) / np.dot(x, x) for k in range(1, lags + 1)]
            Q = M * (M + 2) * sum(r * r / (M - k) for k, r in zip(range(1, lags + 1), ac))
            rej += Q > chi2.ppf(0.99, lags)
        dep.append({'fdTs': fdTs, 'lag1_power_corr_theory': (None if fdTs is None else float(j0(2 * math.pi * fdTs) ** 2)),
                    'ljung_box_reject_rate_1pct': rej / reps})
    res['independence_test'] = dep
    print('independence', dep, flush=True)
    (ROOT / 'results_stress_tests_v4.json').write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
