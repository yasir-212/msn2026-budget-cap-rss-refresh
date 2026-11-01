"""Rate with BOTH qualification certificates (scale + survey Eve), Table II row.

Scale: exact chi-square interval from M_q Alice samples at eps_src = 1e-9.
Eve:   survey Kendall-tau bound (S = 25 positions x 2 probe directions, M_s samples,
       eps_E = 1e-6), true per-link |rho_AE| equal to the cell's nominal value; the two
       links are combined through s_E (Lemma 1).
Rates are BOLT replays on the main-grid sessions (bit/block and bit/channel use).
"""
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
from scipy.stats import chi2, kendalltau
from scipy.interpolate import interp1d
from rayleigh_entropy import analog_eve_certificate, independent_eve_spec
from bolt_policy import CalibrationLibrary
from replay_tools import cell_rate
from run_main_grid_v4 import load_traces
from run_certificate_acquisition_v4 import tau_of_r_table, survey_bound, effective_rho

ROOT = Path(__file__).resolve().parent
RNG = np.random.default_rng(20261029)


def main():
    traces = load_traces(); lib = CalibrationLibrary.load()
    B = json.loads((ROOT / 'results_certificate_acquisition_v4.json').read_text())['B_survey']['rate_vs_M']
    rs, taus = tau_of_r_table()
    r_of_tau = interp1d(taus, rs, bounds_error=False, fill_value=(0.0, 0.999))
    out = []
    for rab, rae in ((0.9975, 0.1), (0.995, 0.3)):
        for Mq, Ms in ((10_000, 200_000), (1_000_000, 200_000)):
            eps = 1e-9
            mL = 2 * Mq / chi2.ppf(1 - eps / 2, 2 * Mq); mU = 2 * Mq / chi2.ppf(eps / 2, 2 * Mq)
            prior = [r for r in B if r['M'] == Ms and r['S'] == 25] if rae == 0.1 else []
            # reuse the Sec. VIII survey value for the same setting, so both places report one number
            rb = prior[0]['certified_rho'] if prior else survey_bound(rae, Ms, 25, 1e-6, r_of_tau, taus)
            hs = {H: float(analog_eve_certificate(H, eve_spec=independent_eve_spec(rb, 2), alice_scale_ratio_bounds=(mL, mU)).h_min_per_block) for H in (1, 2)}
            r = cell_rate(traces, lib, rab, hs, 'bolt')
            row = {'rho_ab': rab, 'true_rho_ae': rae, 'M_q': Mq, 'M_s': Ms, 'mu': [float(mL), float(mU)],
                   'certified_rho_ae': rb, 'effective_rho_ae': effective_rho(rb, 2), 'h2': hs[2], 'rate_bolt_blk': r, 'rate_bolt_cu': r / 2}
            print(row, flush=True); out.append(row)
    (ROOT / 'results_combined_certificates_v4.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
