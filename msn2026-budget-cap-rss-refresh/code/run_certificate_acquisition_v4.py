"""How the external certificates are obtained and validated (revision v4).

A. RSS-scale certificate from M qualification samples (exact chi-square
   interval); empirical coverage with i.i.d. samples, with temporally
   correlated (Jakes) samples, and after decorrelation thinning; certified
   entropy and end-to-end rate versus qualification effort (M, eps_src).
B. Survey-based Eve certificate: a trusted probe at S candidate Eve positions
   records M paired RSS samples with Alice.  A distribution-free Kendall-tau
   upper confidence bound (Hoeffding for U-statistics) is mapped to an upper
   bound on |rho_AE| through the certified Rayleigh pair law; union bound over
   positions.  Empirical coverage; certified rate versus M.
C. Guard-zone Eve certificate: isotropic (Clarke) envelope
   sup_{d>=d_min}|J0(2 pi d/lambda)|, and a von Mises non-isotropic
   counterexample showing when the envelope is NOT a valid certificate.
D. Misspecification: secrecy bound when the true Eve correlation exceeds the
   certified one.
"""
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
from scipy.stats import chi2, kendalltau
from scipy.special import j0, i0
from scipy.interpolate import interp1d
from rayleigh_entropy import analog_eve_certificate, independent_eve_spec
from bolt_policy import CalibrationLibrary
from replay_tools import cell_rate
from run_main_grid_v4 import load_traces, NPAY
from security import DEFAULT_EPS_PA

ROOT = Path(__file__).resolve().parent
RNG = np.random.default_rng(20261027)


def cn(rng, size):
    return (rng.normal(size=size) + 1j * rng.normal(size=size)) / math.sqrt(2)


def jakes(rng, n, fdTs, M=32):
    """Unit-power sum-of-sinusoids Rayleigh process sampled at k*Ts (Zheng-Xiao)."""
    k = np.arange(n)[:, None]
    th = rng.uniform(-math.pi, math.pi); phi = rng.uniform(-math.pi, math.pi, size=M); psi = rng.uniform(-math.pi, math.pi, size=M)
    m = np.arange(1, M + 1)[None, :]
    alpha = (2 * math.pi * m - math.pi + th) / (4 * M)
    re = np.cos(2 * math.pi * fdTs * k * np.cos(alpha) + phi).sum(1)
    im = np.cos(2 * math.pi * fdTs * k * np.sin(alpha) + psi).sum(1)
    return (re + 1j * im) / math.sqrt(M)


def scale_interval(x, eps):
    M = len(x); S = float(np.sum(x))
    lo = 2 * S / chi2.ppf(1 - eps / 2, 2 * M); hi = 2 * S / chi2.ppf(eps / 2, 2 * M)
    mhat = S / M
    return lo / mhat, hi / mhat, lo, hi


# ---------------------------------------------------------------- A
def part_A(traces, lib):
    out = {'coverage': {}, 'rate_vs_effort': []}
    reps = 20000
    for M, eps in ((500, 0.01), (500, 0.05), (2000, 0.01)):
        miss = 0
        for _ in range(reps):
            x = RNG.exponential(1.0, M)
            _, _, lo, hi = scale_interval(x, eps)
            miss += not (lo <= 1.0 <= hi)
        out['coverage'][f'iid_M{M}_eps{eps}'] = {'miss_rate': miss / reps, 'target': eps}
    # temporally correlated samples (Jakes, f_d*Ts = 0.02) with and without thinning
    for fdTs, thin in ((0.02, 1), (0.02, 25), (0.05, 1), (0.05, 10)):
        M, eps = 500, 0.05; miss = 0; reps2 = 4000
        for _ in range(reps2):
            h = jakes(RNG, M * thin, fdTs)[::thin]
            _, _, lo, hi = scale_interval(np.abs(h) ** 2, eps)
            miss += not (lo <= 1.0 <= hi)
        lag1 = float(j0(2 * math.pi * fdTs * thin) ** 2)
        out['coverage'][f'jakes_fdTs{fdTs}_thin{thin}'] = {'miss_rate': miss / reps2, 'target': eps, 'power_corr_at_spacing': lag1}
    # certified h and rate versus qualification effort
    for M in (500, 2000, 10_000, 50_000, 200_000, 10 ** 6):
        for eps in (1e-2, 1e-6, 1e-9):
            mL = 2 * M / chi2.ppf(1 - eps / 2, 2 * M); mU = 2 * M / chi2.ppf(eps / 2, 2 * M)  # interval at S=M
            row = {'M': M, 'eps_src': eps, 'mu_L': float(mL), 'mu_U': float(mU)}
            for rae in (0.1, 0.3):
                hs = {H: float(analog_eve_certificate(H, eve_spec=independent_eve_spec(rae, 1), alice_scale_ratio_bounds=(mL, mU)).h_min_per_block) for H in (1, 2)}
                row[f'h2_rae{rae}'] = hs[2]
                row[f'rate_9975_rae{rae}'] = cell_rate(traces, lib, .9975, hs, 'bolt')
                row[f'rate_995_rae{rae}'] = cell_rate(traces, lib, .995, hs, 'bolt')
                row[f'rate_lcb_9975_rae{rae}'] = cell_rate(traces, lib, .9975, hs, 'bolt_lcb')
                row[f'rate_q99_9975_rae{rae}'] = cell_rate(traces, lib, .9975, hs, 'q99')
            out['rate_vs_effort'].append(row)
            print('A', M, eps, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    return out


# ---------------------------------------------------------------- B
TAU_MC_N = 400_000
TAU_MC_MARGIN = 0.003   # about 3 Monte Carlo standard deviations of tau at TAU_MC_N samples


def tau_of_r_table():
    """Kendall tau of (|h_A|^2,|h_E|^2) for complex-Gaussian pairs with |rho|^2=r.

    Estimated by Monte Carlo with TAU_MC_N samples per grid point and shifted DOWN by
    TAU_MC_MARGIN, so that mapping an upper bound on tau through the inverse table gives a
    conservative (larger) correlation bound."""
    rs = np.linspace(0, 0.98, 50); taus = []
    rng = np.random.default_rng(11)
    for r in rs:
        rho = math.sqrt(r); n = TAU_MC_N
        a = cn(rng, n); e = rho * a + math.sqrt(1 - r) * cn(rng, n)
        taus.append(kendalltau(np.abs(a) ** 2, np.abs(e) ** 2)[0])
    taus = np.maximum.accumulate(np.clip(np.array(taus) - TAU_MC_MARGIN, 0.0, None))
    return rs, taus


def part_B(traces, lib):
    rs, taus = tau_of_r_table()
    r_of_tau = interp1d(taus, rs, bounds_error=False, fill_value=(0.0, 0.999))
    out = {'tau_table': {'r': rs.tolist(), 'tau': taus.tolist()}, 'coverage': {}, 'rate_vs_M': []}

    def ucb_rho(xa, xe, eps):
        t = kendalltau(xa, xe)[0]; m = len(xa) // 2
        tu = min(t + math.sqrt(2 * math.log(1 / eps) / m), taus.max())
        return math.sqrt(float(r_of_tau(tu)))
    for true_rho in (0.1, 0.3):
        for M in (500, 2000):
            reps = 2000; miss = 0; eps = 0.05
            for _ in range(reps):
                a = cn(RNG, M); e = true_rho * a + math.sqrt(1 - true_rho ** 2) * cn(RNG, M)
                miss += ucb_rho(np.abs(a) ** 2, 10 * np.log10(np.abs(e) ** 2) + 3.0, eps) < true_rho  # dB + gain offset: tau invariant
            out['coverage'][f'rho{true_rho}_M{M}'] = {'miss_rate': miss / reps, 'target': eps}
    for M in (2000, 10_000, 50_000, 200_000):
        for S in (1, 25):
            eps_E = 1e-6
            rb = survey_bound(0.1, M, S, eps_E, r_of_tau, taus)
            # Eve hears both probe directions: two links per position, combined through s_E (Lemma 1)
            hs = {H: float(analog_eve_certificate(H, eve_spec=independent_eve_spec(rb, 2)).h_min_per_block) for H in (1, 2)}
            row = {'M': M, 'S': S, 'eps_E': eps_E, 'certified_rho': rb, 'links': 2,
                   'effective_rho': effective_rho(rb, 2), 'h2': hs[2],
                   'rate_9975_bolt': cell_rate(traces, lib, .9975, hs, 'bolt'),
                   'rate_9975_lcb': cell_rate(traces, lib, .9975, hs, 'bolt_lcb'),
                   'rate_9975_q99': cell_rate(traces, lib, .9975, hs, 'q99')}
            out['rate_vs_M'].append(row); print('B', row, flush=True)
    return out


# ---------------------------------------------------------------- C
def effective_rho(rho_link, links):
    s = links * rho_link ** 2 / (1 - rho_link ** 2)
    return math.sqrt(s / (1 + s))


def survey_bound(true_rho, M, S, eps_E, r_of_tau, taus, draws=3):
    """Expected certified per-link bound: S positions x 2 links (Alice->probe, Bob->probe),
    each with M paired samples; Hoeffding radius for U-statistics with a union bound over
    the 2S tests; the max over all tests is taken exactly (no subsampling)."""
    vals = []
    for _ in range(draws):
        worst = 0.0
        for _t in range(2 * S):
            a = cn(RNG, M); e = true_rho * a + math.sqrt(1 - true_rho ** 2) * cn(RNG, M)
            t = kendalltau(np.abs(a) ** 2, np.abs(e) ** 2)[0]
            tu = min(t + math.sqrt(2 * math.log(2 * S / eps_E) / (M // 2)), taus.max())
            worst = max(worst, math.sqrt(float(r_of_tau(tu))))
        vals.append(worst)
    return float(np.mean(vals))


def vonmises_corr(d_over_lambda, kappa, mu, alpha=0.0, n=4096):
    th = np.linspace(-math.pi, math.pi, n, endpoint=False)
    p = np.exp(kappa * np.cos(th - mu)); p /= p.sum()
    return abs(np.sum(p * np.exp(1j * 2 * math.pi * d_over_lambda * np.cos(th - alpha))))


def part_C(traces, lib):
    out = {'guard_zone': [], 'nonisotropic': []}
    grid = np.linspace(0.01, 40, 40000)
    env = np.abs(j0(2 * math.pi * grid))
    for dmin in (0.5, 1, 2, 4, 8, 16, 32):
        rb = float(env[grid >= dmin].max())
        hs = {H: float(analog_eve_certificate(H, eve_spec=independent_eve_spec(rb, 2)).h_min_per_block) for H in (1, 2)}
        row = {'d_min_over_lambda': dmin, 'certified_rho': rb, 'links': 2, 'effective_rho': effective_rho(rb, 2), 'h2': hs[2],
               'rate_9975_bolt': cell_rate(traces, lib, .9975, hs, 'bolt'),
               'rate_9975_lcb': cell_rate(traces, lib, .9975, hs, 'bolt_lcb'),
               'rate_995_bolt': cell_rate(traces, lib, .995, hs, 'bolt'),
               'rate_9975_q99': cell_rate(traces, lib, .9975, hs, 'q99')}
        out['guard_zone'].append(row); print('C', row, flush=True)
    for kappa in (0, 2, 8, 32):
        for dmin in (1, 4, 16):
            ds = np.linspace(dmin, dmin + 4, 200)
            worst = max(vonmises_corr(d, kappa, 0.0, 0.0) for d in ds)  # displacement along the mean AoA
            out['nonisotropic'].append({'kappa': kappa, 'd_min_over_lambda': dmin, 'sup_rho': float(worst),
                                        'isotropic_envelope': float(env[grid >= dmin].max())})
    print('C nonisotropic', out['nonisotropic'], flush=True)
    return out


# ---------------------------------------------------------------- D
def part_D():
    out = []
    n = NPAY
    for H in (1, 2):
        for rc in (0.1, 0.3, 0.5):
            hc = analog_eve_certificate(H, eve_spec=independent_eve_spec(rc, 1)).h_min_per_block
            for dr in (0.005, 0.01, 0.02, 0.05, 0.1):
                rt = rc + dr
                ht = analog_eve_certificate(H, eve_spec=independent_eve_spec(rt, 1)).h_min_per_block
                deficit = n * (hc - ht)
                log2eps = math.log2(DEFAULT_EPS_PA) + deficit / 2
                out.append({'H': H, 'rho_cert': rc, 'rho_true': rt, 'h_cert': hc, 'h_true': ht,
                            'deficit_bits': deficit, 'log2_eps_bound': min(0.0, log2eps)})
    print('D', [(o['H'], o['rho_cert'], o['rho_true'], round(o['deficit_bits'], 1), round(o['log2_eps_bound'], 1)) for o in out], flush=True)
    return out


if __name__ == '__main__':
    import sys
    parts = sys.argv[1] if len(sys.argv) > 1 else 'ABCD'   # e.g. "BC" re-runs parts B and C only
    tr = load_traces(); lib = CalibrationLibrary.load()
    path = ROOT / 'results_certificate_acquisition_v4.json'
    res = json.loads(path.read_text()) if (parts != 'ABCD' and path.exists()) else {'version': 'bolt-rss-v4-certificates'}
    if 'D' in parts: res['D_misspecification'] = part_D()
    if 'C' in parts: res['C_guard_zone'] = part_C(tr, lib)
    if 'B' in parts: res['B_survey'] = part_B(tr, lib)
    if 'A' in parts: res['A_scale'] = part_A(tr, lib)
    path.write_text(json.dumps(res, indent=1))
