"""Unified cap/resolution selection policies for BOLT-RSS (revision v4).

All policies share one implementation of the calibration neighbourhood and
one candidate-cap set, so BOLT and BOLT-LCB differ *only* in the acceptance
score that multiplies the application-key length.

Calibration record (per resolution H):
    z   : pilot one-sided BMR upper bound
    C   : 1 if uncapped distributed Cascade reached a confirmation match
    Lreq: smallest hard cap under which that run passes every pre-transmission
          budget check (worst-case reservations included); a run under cap
          Lambda accepts iff C=1 and Lreq<=Lambda (exact, see cascade.py)

Policies (chosen before any payload-dependent message):
    bolt        argmax_Lambda  Ghat(Lambda|z) * ell_app(Lambda)
    bolt_lcb    argmax_Lambda  Glow(Lambda|z) * ell_app(Lambda)
    qXX         Lambda = weighted XX-th percentile of local Lreq
    fixedcap    one z-independent cap per (H, h) from the whole library
Candidate caps for bolt and bolt_lcb: every distinct Lreq value in the local
neighbourhood (both objectives are step functions that change only there).

Lower confidence bounds for bolt_lcb (weights w fixed given {z_i}):
    'hoeffding'         pointwise weighted Hoeffding (original Eq. 12)
    'hoeffding_uniform' weighted Hoeffding + Bonferroni over the m candidate
                        caps -> simultaneous (1-alpha) bound, valid for the
                        data-dependent argmax (default in the revision)
None of these scores enters the secrecy accounting.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np

from security import AUTH_REPLENISH_BITS, DEFAULT_EPS_PA, pa_margin_bits, variable_transcript_support_bits

DEFAULT_CAL = Path(__file__).with_name('protocol_calibration_v4.json')


@dataclass(frozen=True)
class PolicyParams:
    K: int = 80
    kernel: str = 'gauss'          # 'gauss' | 'uniform'
    alpha: float = 0.05
    lcb: str = 'bracket'            # 'hoeffding' | 'hoeffding_uniform' | 'bracket'
    one_sided: bool = True          # bolt_lcb only: neighbours with z_i >= z
    domain_slack: float = 1.05      # pre-abort if z > slack * max calibrated z
    eps_pa: float = DEFAULT_EPS_PA


@dataclass(frozen=True)
class CapChoice:
    H: int
    cap_bits: int
    predicted_accept: float        # score used (point or LCB)
    point_accept: float            # point estimate Ghat at the chosen cap
    delivered_if_accept: int
    objective: float
    policy: str


class CalibrationLibrary:
    def __init__(self, records: list[dict]):
        self.records = list(records)
        self.by_H: dict[int, dict[str, np.ndarray]] = {}
        for H in sorted({int(r['H']) for r in self.records}):
            rows = [r for r in self.records if int(r['H']) == H]
            self.by_H[H] = {
                'z': np.array([float(r['pilot_bmr_upper']) for r in rows]),
                'C': np.array([bool(r['converged']) for r in rows]),
                'L': np.array([int(r['required_cap']) for r in rows]),
            }

    @classmethod
    def load(cls, path: str | Path = DEFAULT_CAL) -> 'CalibrationLibrary':
        return cls(json.loads(Path(path).read_text())['records'])

    def subset(self, keep) -> 'CalibrationLibrary':
        return CalibrationLibrary([r for i, r in enumerate(self.records) if keep(i, r)])

    def domain(self, H: int) -> tuple[float, float]:
        z = self.by_H[H]['z']
        return float(z.min()), float(z.max())


def delivered_length_for_cap(n_payload: int, h: float, cap: int, max_raw: int,
                             eps_pa: float = DEFAULT_EPS_PA) -> int:
    ell = math.floor(n_payload * h - variable_transcript_support_bits(int(cap)) - pa_margin_bits(eps_pa))
    ell = min(max(int(ell), 0), int(max_raw))
    return max(ell - AUTH_REPLENISH_BITS, 0)


def _weighted_quantile(v: np.ndarray, w: np.ndarray, q: float) -> float:
    o = np.argsort(v, kind='stable'); v = v[o]; w = w[o]
    c = np.cumsum(w); c /= c[-1]
    return float(v[min(int(np.searchsorted(c, q, side='left')), len(v) - 1)])


def local_neighbourhood(lib: CalibrationLibrary, H: int, z: float, p: PolicyParams, one_sided: bool = False):
    d = lib.by_H[H]
    zz = d['z']
    lo, hi = float(zz.min()), float(zz.max())
    zc = min(max(z, lo), hi)
    dist = np.abs(zz - zc)
    if one_sided:
        # only records whose pilot statistic is at least as bad as the live one
        dist = np.where(zz >= z, zz - z, np.inf)
        if not np.isfinite(dist).any():
            return d['C'][:0], d['L'][:0], np.zeros(0), zz[:0]
    idx = np.argsort(dist, kind='stable')[:min(p.K, int(np.isfinite(dist).sum()))]
    dd = dist[idx]
    if p.kernel == 'gauss':
        b = max(float(dd.max()), 1e-6)
        w = np.exp(-0.5 * (dd / b) ** 2)
    elif p.kernel == 'uniform':
        w = np.ones(len(idx))
    else:
        raise ValueError('unknown kernel')
    w = w / w.sum()
    return d['C'][idx], d['L'][idx], w, zz[idx]


def bracket_radius(s2: float, alpha: float, kmax: int = 400) -> float:
    """Uniform-in-Lambda Hoeffding radius via bracketing of the monotone target.

    With k fixed brackets of the nondecreasing step function Gbar(.), a union
    bound over k+1 fixed events plus the bracket width 1/k gives, w.p. >=1-alpha,
    Gbar(Lambda) >= Ghat(Lambda) - r for ALL Lambda simultaneously (so the
    data-dependent argmax is covered).  k is chosen to minimize r; it depends
    only on the weights, which are fixed given the pilot coordinates.
    """
    k = np.arange(1, kmax + 1)
    r = np.sqrt(0.5 * np.log((k + 1) / alpha) * s2) + 1.0 / k
    return float(r.min())


def lcb_radius(w: np.ndarray, alpha: float, m: int, kind: str) -> float:
    s2 = float(np.sum(w * w))
    if kind == 'bracket':
        return bracket_radius(s2, alpha)
    if kind == 'hoeffding':
        return math.sqrt(0.5 * math.log(1.0 / alpha) * s2)
    if kind == 'hoeffding_uniform':
        return math.sqrt(0.5 * math.log(max(m, 1) / alpha) * s2)
    raise ValueError('unknown lcb kind')


def choose_cap(lib: CalibrationLibrary, H: int, z: float, h: float, n_payload: int,
               policy: str, p: PolicyParams = PolicyParams(),
               fixed_caps: dict | None = None) -> CapChoice:
    lo, hi = lib.domain(H)
    nH = n_payload * H
    if z > hi * p.domain_slack:
        return CapChoice(H, 0, 0.0, 0.0, 0, 0.0, policy + '-preabort')
    C, L, w, _ = local_neighbourhood(lib, H, z, p, one_sided=((policy == 'bolt_lcb' and p.one_sided) or policy == 'bolt_os'))
    if len(w) == 0:
        return CapChoice(H, 0, 0.0, 0.0, 0, 0.0, policy + '-preabort')
    if policy.startswith('q') and policy[1:].isdigit():
        q = int(policy[1:]) / 100.0
        cap = int(math.ceil(_weighted_quantile(L.astype(float), w, q)))
        pacc = float(np.sum(w * (C & (L <= cap))))
        ell = delivered_length_for_cap(n_payload, h, cap, nH, p.eps_pa)
        return CapChoice(H, cap, pacc, pacc, ell, pacc * ell, policy)
    if policy == 'fixedcap':
        cap = int(fixed_caps[(H, round(h, 12))])
        pacc = float(np.sum(w * (C & (L <= cap))))
        ell = delivered_length_for_cap(n_payload, h, cap, nH, p.eps_pa)
        return CapChoice(H, cap, pacc, pacc, ell, pacc * ell, policy)
    if policy not in ('bolt', 'bolt_lcb', 'bolt_os'):
        raise ValueError('unknown policy ' + policy)
    caps = np.unique(L)
    m = len(caps)
    rad = lcb_radius(w, p.alpha, m, p.lcb) if policy == 'bolt_lcb' else 0.0
    # Vectorized step-function evaluation at every candidate cap.
    o = np.argsort(L, kind='stable')
    cw = np.cumsum((w * C)[o])
    pos = np.searchsorted(L[o], caps, side='right') - 1
    g = np.where(pos >= 0, cw[np.maximum(pos, 0)], 0.0)
    score = np.maximum(0.0, g - rad)
    raw = np.floor(n_payload * h - (caps + 1) - pa_margin_bits(p.eps_pa))
    ell = np.maximum(np.minimum(np.maximum(raw, 0), nH) - AUTH_REPLENISH_BITS, 0).astype(np.int64)
    J = score * ell
    k = int(np.argmax(J))  # first maximizer = smallest cap among ties (as the loop version)
    best = CapChoice(H, int(caps[k]), float(score[k]), float(g[k]), int(ell[k]), float(J[k]), policy)
    if best.objective <= 0:
        # Nothing positive is predicted: pre-abort with no payload discussion.
        return CapChoice(H, 0, 0.0, 0.0, 0, 0.0, policy + '-preabort')
    return best


def select(lib: CalibrationLibrary, z_by_H: dict[int, float], h_by_H: dict[int, float],
           n_payload: int, policy: str, p: PolicyParams = PolicyParams(),
           fixed_H: int | None = None, fixed_caps: dict | None = None) -> CapChoice:
    Hs = [fixed_H] if fixed_H is not None else sorted(z_by_H)
    ch = [choose_cap(lib, H, z_by_H[H], h_by_H[H], n_payload, policy, p, fixed_caps) for H in Hs]
    return max(ch, key=lambda c: (c.objective, -c.H))


def global_fixed_caps(lib: CalibrationLibrary, Hs: Iterable[int], hs: Iterable[float],
                      n_payload: int, eps_pa: float = DEFAULT_EPS_PA) -> dict:
    """One z-independent cap per (H,h) maximizing library-average yield."""
    out = {}
    for H in Hs:
        d = lib.by_H[H]
        for h in hs:
            best = (-1.0, 0)
            for cap in np.unique(d['L']):
                pacc = float(np.mean(d['C'] & (d['L'] <= cap)))
                J = pacc * delivered_length_for_cap(n_payload, h, int(cap), n_payload * H, eps_pa)
                if J > best[0]:
                    best = (J, int(cap))
            out[(H, round(h, 12))] = best[1]
    return out


def outcome(choice: CapChoice, trace_by_H: dict[int, dict]) -> int:
    """Delivered application bits for a replayed session (exact)."""
    if choice.cap_bits <= 0:
        return 0
    t = trace_by_H[choice.H]
    return int(choice.delivered_if_accept) if (t['C'] and t['Lreq'] <= choice.cap_bits) else 0


def oracle(trace_by_H: dict[int, dict], h_by_H: dict[int, float], n_payload: int,
           fixed_H: int | None = None, eps_pa: float = DEFAULT_EPS_PA) -> int:
    best = 0
    for H, t in trace_by_H.items():
        if fixed_H is not None and H != fixed_H:
            continue
        if t['C']:
            best = max(best, delivered_length_for_cap(n_payload, h_by_H[H], t['Lreq'], n_payload * H, eps_pa))
    return best


# ---------------------------------------------------------------------------
# Variable-length charging (per-length union bound), reported as a comparison.
# For an accepted transcript of realized length L <= lam_max, charging
# L + 2*log2(lam_max+1) (+ the usual PA margin) is secure: transcripts of length
# l number at most 2^l, and a union bound over the lam_max+1 lengths with
# eps/(lam_max+1) each gives total distance <= eps (cf. variable-length QKD proofs).
def vl_length(n_payload: int, h: float, L: int, H: int, lam_max: int, eps_pa: float = DEFAULT_EPS_PA) -> int:
    ell = math.floor(n_payload * h - L - pa_margin_bits(eps_pa) - 2.0 * math.log2(lam_max + 1))
    return max(min(max(int(ell), 0), n_payload * H) - AUTH_REPLENISH_BITS, 0)


def vl_select_H(lib: CalibrationLibrary, z_by_H: dict, h_by_H: dict, n_payload: int,
                p: PolicyParams = PolicyParams()):
    """Choose H maximizing library-expected variable-length yield with no cap (lam_max = nH)."""
    best = (None, 0.0)
    for H in sorted(z_by_H):
        if z_by_H[H] > lib.domain(H)[1] * p.domain_slack:
            continue
        C, L, w, _ = local_neighbourhood(lib, H, z_by_H[H], p)
        J = float(np.sum(w * np.array([vl_length(n_payload, h_by_H[H], int(l), H, n_payload * H, p.eps_pa) if c else 0
                                        for c, l in zip(C, L)])))
        if J > best[1]:
            best = (H, J)
    return best[0]
