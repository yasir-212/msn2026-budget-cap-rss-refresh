import math, sys, unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))

from security import (pa_margin_bits, variable_transcript_support_bits,
                      extracted_length, lifetime_failure_bound)
from auth import fresh_auth_material, parse_auth_material, wc_tag, wc_verify
from rayleigh_entropy import (symbols_to_bits, analog_eve_certificate,
                              independent_eve_spec, effective_eve_posterior_snr,
                              EveGaussianSpec)
from bolt_cap import q95_choice, q99_choice, _weighted_hoeffding_lower
from cascade import cascade_reconcile
from bolt_rss import bolt_rss_session

class SecurityPolicyTests(unittest.TestCase):
    def test_variable_transcript_support_charge(self):
        self.assertEqual(variable_transcript_support_bits(0),1)
        self.assertEqual(variable_transcript_support_bits(17),18)

    def test_leftover_hash_margin(self):
        self.assertAlmostEqual(pa_margin_bits(2.0**-64),126.0,places=12)

    def test_extracted_length_decreases_with_cap(self):
        a=extracted_length(1000,1.5,101,2000)
        b=extracted_length(1000,1.5,201,2000)
        self.assertGreaterEqual(a,b)

    def test_lifetime_external_event_charged_once(self):
        x=lifetime_failure_bound(1e-9,10,1e-3)
        self.assertAlmostEqual(x,1e-3+1e-8,places=15)

    def test_wegman_carter_tamper_detection(self):
        material=fresh_auth_material(np.random.default_rng(7))
        hk,pad=parse_auth_material(material)
        m=b'BOLT-RSS transcript'
        tag=wc_tag(hk,pad,m)
        self.assertTrue(wc_verify(hk,pad,m,tag))
        self.assertFalse(wc_verify(hk,pad,m+b'!',tag))

    def test_gray_mapping_H2(self):
        bits=symbols_to_bits(np.array([0,1,2,3]),2).reshape(-1,2)
        expected=np.array([[0,0],[0,1],[1,1],[1,0]],dtype=np.uint8)
        np.testing.assert_array_equal(bits,expected)

    def test_eve_strength_reduces_entropy(self):
        hweak=analog_eve_certificate(2,eve_spec=independent_eve_spec(.1,1)).h_min_per_block
        hstrong=analog_eve_certificate(2,eve_spec=independent_eve_spec(.7,1)).h_min_per_block
        self.assertGreater(hweak,hstrong)

    def test_scale_uncertainty_is_conservative(self):
        spec=independent_eve_spec(.3,1)
        exact=analog_eve_certificate(2,eve_spec=spec,alice_scale_ratio_bounds=(1,1)).h_min_per_block
        finite=analog_eve_certificate(2,eve_spec=spec,alice_scale_ratio_bounds=(.894,1.125)).h_min_per_block
        self.assertLessEqual(finite,exact+1e-12)

    def test_correlated_eve_effective_snr(self):
        S=np.array([[1,.3],[.3,1]],dtype=np.complex128)
        a=np.array([.25,.2],dtype=np.complex128)
        spec=EveGaussianSpec.from_arrays(a,S)
        ref=float(np.real(np.vdot(a,np.linalg.solve(S,a))))
        self.assertAlmostEqual(effective_eve_posterior_snr(spec),ref,places=12)

    def test_q95_cap_not_above_q99(self):
        c95=q95_choice(7680,2,.03,1.5)
        c99=q99_choice(7680,2,.03,1.5)
        self.assertLessEqual(c95.cap_bits,c99.cap_bits)

    def test_weighted_hoeffding_not_above_point_mean(self):
        x=np.array([1,1,0,1],float); w=np.array([.1,.2,.3,.4])
        lb=_weighted_hoeffding_lower(x,w,.05)
        self.assertLessEqual(lb,float(np.sum((w/w.sum())*x))+1e-15)
        self.assertGreaterEqual(lb,0.0)

    def test_cascade_fails_closed_at_tiny_cap(self):
        a=np.zeros(64,dtype=np.uint8); b=a.copy(); b[0]=1
        seed=np.zeros(len(a)+64-1,dtype=np.uint8)
        r=cascade_reconcile(a,b,est_bmr=.05,confirm_seed_bits=seed,max_source_leak_bits=0)
        self.assertTrue(r['budget_abort'])
        self.assertEqual(r['leak_bits'],0)

    def test_missing_certificate_returns_zero(self):
        a=np.ones(256); b=np.ones(256)
        r=bolt_rss_session(a,b,None,source_certificate=None,eve_certificate=None,
                           auth_key_material=bytes(32),active_health_passed=True,
                           n_pilot=64,materialize_keys=False)
        self.assertEqual(r['delivered_length'],0)
        self.assertFalse(r['certificate_applicable'])

if __name__=='__main__': unittest.main()
