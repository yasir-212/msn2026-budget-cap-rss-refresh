from __future__ import annotations

import numpy as np
from scipy.stats import beta

from cascade import cascade_reconcile
from rayleigh_entropy import quantize_power, symbols_to_bits


def pilot_table(a_pilot, b_pilot, Hs=(1, 2), confidence=0.99):
    """Nested pilot statistics at Hmax=max(Hs); Bonferroni over |Hs| (as bolt_rss)."""
    Hmax = max(Hs)
    qa = quantize_power(a_pilot, Hmax); qb = quantize_power(b_pilot, Hmax)
    alpha = (1.0 - confidence) / len(Hs)
    out = {}
    for H in Hs:
        ba = symbols_to_bits(qa >> (Hmax - H), H); bb = symbols_to_bits(qb >> (Hmax - H), H)
        k = int(np.count_nonzero(ba != bb)); n = len(ba)
        pu = 1.0 if k >= n else float(beta.ppf(1.0 - alpha, k + 1, n - k))
        out[H] = {'mismatches': k, 'n_bits': n, 'bmr': k / n, 'bmr_upper': float(min(pu, 0.5))}
    return out


def session_traces(a: np.ndarray, b: np.ndarray, public_seed_base: int, n_pilot: int = 512,
                   Hs=(1, 2), confidence: float = 0.99, confirm_bits: int = 64,
                   max_passes: int = 6, pilot_Hs=None) -> dict:
    """Return {'pilot': table, H: {'z','C','Lreq','L','nbits','raw_bmr'}}.

    Reproduces bolt_rss_session(public_seed=public_seed_base+1000*H) exactly
    when pilot_Hs=(1,2) (the default live configuration).
    """
    pHs = tuple(pilot_Hs) if pilot_Hs is not None else ((1, 2) if max(Hs) <= 2 else tuple(Hs))
    pt = pilot_table(a[:n_pilot], b[:n_pilot], pHs, confidence)
    out = {'pilot': pt}
    for H in Hs:
        qa = quantize_power(a[n_pilot:], H); qb = quantize_power(b[n_pilot:], H)
        ba = symbols_to_bits(qa, H); bb = symbols_to_bits(qb, H)
        ps = int(public_seed_base) + 1000 * H
        rng = np.random.default_rng(ps)
        cs = rng.integers(0, 2, size=len(ba) + confirm_bits - 1, dtype=np.uint8)
        r = cascade_reconcile(ba, bb, est_bmr=pt[H]['bmr_upper'], seed=ps & 0x7fffffff,
                              confirm_bits=confirm_bits, confirm_seed_bits=cs, max_passes=max_passes)
        out[H] = {'z': float(pt[H]['bmr_upper']), 'C': bool(r['converged']),
                  'Lreq': int(r['required_cap']), 'L': int(r['leak_bits']),
                  'nbits': int(len(ba)), 'raw_bmr': float(np.mean(ba != bb)),
                  'confirm_attempts': int(r.get('confirmation_attempts', 0))}
    return out
