"""Unit tests for the revision-v4 additions."""
import math, sys, unittest
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))

from bolt_policy import CalibrationLibrary, PolicyParams, choose_cap, select, outcome, bracket_radius, local_neighbourhood
from traces import session_traces
from rayleigh_entropy import simulate_correlated_rayleigh_blocks, independent_eve_spec, analog_eve_certificate
from bolt_rss import SourceModelCertificate, bolt_rss_session, select_H, pilot_bmr_table
from auth import fresh_auth_material


class V4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lib = CalibrationLibrary.load()

    def test_replay_matches_live_session(self):
        src = SourceModelCertificate(True, 1, 1, 0)
        eve = independent_eve_spec(0.2, 1)
        for t in range(3):
            s = 55_000_000 + 9975 * 1000 + t
            a, b, _ = simulate_correlated_rayleigh_blocks(8192, 0.9975, 0.1, s)
            tr = session_traces(a, b, s, 512)
            hs = {H: analog_eve_certificate(H, eve_spec=eve).h_min_per_block for H in (1, 2)}
            for pol in ('bolt', 'bolt_lcb', 'q99'):
                ch = select(self.lib, {1: tr[1]['z'], 2: tr[2]['z']}, hs, 7680, pol)
                pt, _ = pilot_bmr_table(a[:512], b[:512])
                H, _ = select_H(7680, pt, eve, objective=pol)
                r = bolt_rss_session(a, b, None, source_certificate=src, eve_certificate=eve,
                                     auth_key_material=fresh_auth_material(np.random.default_rng(1)),
                                     active_health_passed=True, n_pilot=512, public_seed=s + 1000 * H,
                                     fixed_H=H, selector_objective=pol, materialize_keys=True)
                self.assertEqual(r['delivered_length'], outcome(ch, {1: tr[1], 2: tr[2]}))
                self.assertLessEqual(r['realized_source_dependent_leak_bits'], max(r['precommitted_leak_cap_bits'], 0))

    def test_bracket_bound_is_uniform_in_lambda(self):
        # weighted Bernoulli step functions with known means: sup-deviation coverage >= 1-alpha
        rng = np.random.default_rng(0); n = 60; w = rng.uniform(.5, 1, n); w /= w.sum(); alpha = 0.1
        r = bracket_radius(float((w * w).sum()), alpha)
        grid = np.arange(0, 50); miss = 0; reps = 2000
        Lmeans = rng.uniform(5, 45, n)
        for _ in range(reps):
            L = rng.poisson(Lmeans); C = rng.random(n) < 0.8
            ghat = np.array([np.sum(w * (C & (L <= g))) for g in grid])
            from scipy.stats import poisson
            gbar = np.array([np.sum(w * 0.8 * poisson.cdf(g, Lmeans)) for g in grid])
            miss += np.any(ghat - gbar > r)
        self.assertLessEqual(miss / reps, alpha)

    def test_one_sided_neighbourhood(self):
        C, L, w, zz = local_neighbourhood(self.lib, 2, 0.06, PolicyParams(), one_sided=True)
        self.assertTrue(np.all(zz >= 0.06)); self.assertAlmostEqual(float(w.sum()), 1.0)

    def test_lcb_score_below_point_estimate(self):
        for z in (0.03, 0.05, 0.08):
            c = choose_cap(self.lib, 2, z, 1.9, 7680, 'bolt_lcb')
            self.assertLessEqual(c.predicted_accept, c.point_accept + 1e-12)

    def test_rssi_rounding_is_conservative(self):
        spec = independent_eve_spec(.2, 1)
        g = 10 ** (1 / 20)
        h_exact = analog_eve_certificate(2, eve_spec=spec).h_min_per_block
        h_round = analog_eve_certificate(2, eve_spec=spec, alice_scale_ratio_bounds=(1 / g, g)).h_min_per_block
        self.assertLess(h_round, h_exact)


if __name__ == '__main__':
    unittest.main()
