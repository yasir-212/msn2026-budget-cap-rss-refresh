"""Realistic RSS measurement emulator for stress-testing the certificate (v4).

Generates Alice/Bob normalized RSS powers for N blocks with impairments that a
deployment would see, each of which touches a different element of the
certificate:
  fd_tau   : Doppler x TDD probing delay -> reciprocity loss J0(2 pi fd tau)
  snr_db   : receiver noise at Alice and Bob (complex AWGN on the channel estimate)
  fdTs     : Doppler x block spacing -> temporal correlation between blocks
             (fdTs=None draws independent blocks)
  rssi_step: RSSI reporting resolution in dB (0 = continuous)
  bob_gain_db : uncalibrated Bob receiver gain offset (dB)
  K_db     : Rician K-factor of a LOS component known to everybody (None = Rayleigh)
Mean measured power is normalized to 1 (the nominal scale certificate).
"""
from __future__ import annotations
import math
import numpy as np
from scipy.special import j0


def cn(rng, size):
    return (rng.normal(size=size) + 1j * rng.normal(size=size)) / math.sqrt(2)


def sos(rng, t, M=48):
    """Unit-power isotropic Rayleigh process at normalized times t (units of 1/f_d)."""
    th = rng.uniform(-math.pi, math.pi); phi = rng.uniform(-math.pi, math.pi, M); psi = rng.uniform(-math.pi, math.pi, M)
    m = np.arange(1, M + 1)
    alpha = (2 * math.pi * m - math.pi + th) / (4 * M)
    tt = np.asarray(t, float)[:, None]
    re = np.cos(2 * math.pi * tt * np.cos(alpha)[None, :] + phi[None, :]).sum(1)
    im = np.cos(2 * math.pi * tt * np.sin(alpha)[None, :] + psi[None, :]).sum(1)
    return (re + 1j * im) / math.sqrt(M)


def gen(N: int, seed: int, fd_tau: float = 0.0, snr_db: float | None = None, fdTs: float | None = None,
        rssi_step: float = 0.0, bob_gain_db: float = 0.0, K_db: float | None = None, rab_direct: float | None = None):
    rng = np.random.default_rng(seed)
    if fdTs is None:
        # independent blocks; Alice/Bob channel coefficients with Clarke time correlation
        r = J = float(j0(2 * math.pi * fd_tau)) if rab_direct is None else float(rab_direct)
        ha = cn(rng, N); hb = r * ha + math.sqrt(max(1 - r * r, 0)) * cn(rng, N)
    else:
        t = np.arange(N) * fdTs
        # a long-run SoS realization: concatenate independent segments to avoid SoS periodicity
        seg = 512; ha = np.empty(N, complex); hb = np.empty(N, complex)
        for s0 in range(0, N, seg):
            tt = t[s0:s0 + seg] - t[s0]
            sub = np.random.default_rng(rng.integers(1 << 62))
            st = sub.bit_generator.state
            ha[s0:s0 + seg] = sos(sub, tt)
            sub.bit_generator.state = st
            hb[s0:s0 + seg] = sos(sub, tt + fd_tau)
    if K_db is not None:
        K = 10 ** (K_db / 10)
        phase = rng.uniform(0, 2 * math.pi)
        los = math.sqrt(K / (K + 1)) * np.exp(1j * phase)
        ha = los + ha / math.sqrt(K + 1); hb = los + hb / math.sqrt(K + 1)
    if snr_db is not None:
        s2 = 10 ** (-snr_db / 10)
        ha = (ha + math.sqrt(s2) * cn(rng, N)) / math.sqrt(1 + s2)
        hb = (hb + math.sqrt(s2) * cn(rng, N)) / math.sqrt(1 + s2)
    pa = np.abs(ha) ** 2; pb = (np.abs(hb) ** 2) * 10 ** (bob_gain_db / 10)
    if rssi_step and rssi_step > 0:
        # RSSI in dB relative to the nominal mean, rounded to the reporting step
        pa = 10 ** (np.round(10 * np.log10(np.maximum(pa, 1e-12)) / rssi_step) * rssi_step / 10)
        pb = 10 ** (np.round(10 * np.log10(np.maximum(pb, 1e-12)) / rssi_step) * rssi_step / 10)
    return pa, pb


def rician_guessing_probability(H: int, K_db: float, rho_e: float, n: int = 400_000, seed: int = 5,
                                thresholds: str = 'rayleigh'):
    """MC estimate (+ 1e-9-level Hoeffding upper bound) of Eve's guessing probability
    for the radial quantizer when Alice's channel is Rician with a LOS component known
    to Eve and Eve observes the scattered part with correlation rho_e."""
    from scipy.stats import ncx2
    from rayleigh_entropy import exponential_quantizer_edges
    rng = np.random.default_rng(seed)
    K = 10 ** (K_db / 10) if K_db is not None else 0.0
    sl = math.sqrt(K / (K + 1)); ss = math.sqrt(1 / (K + 1)); c = rho_e ** 2
    g = cn(rng, n); e = rho_e * g + math.sqrt(1 - c) * cn(rng, n)
    m = sl + ss * rho_e * e          # posterior mean of H_A (LOS phase w.l.o.g. 0)
    v = ss * ss * (1 - c)             # posterior variance
    if thresholds == 'rayleigh':
        edges = exponential_quantizer_edges(H)
    else:  # equiprobable under the Rician marginal (certified Rician source law)
        xs = np.abs(sl + ss * cn(np.random.default_rng(99), 2_000_000)) ** 2
        qs = np.quantile(xs, np.arange(1, 2 ** H) / 2 ** H)
        edges = np.concatenate([[0.0], qs, [np.inf]])
    nc = 2 * np.abs(m) ** 2 / v
    probs = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        flo = np.zeros(n) if lo <= 0 else ncx2.cdf(2 * lo / v, 2, nc)
        fhi = np.ones(n) if not np.isfinite(hi) else ncx2.cdf(2 * hi / v, 2, nc)
        probs.append(np.maximum(fhi - flo, 0))
    q = np.max(np.vstack(probs), 0)
    p = float(q.mean()); ub = p + math.sqrt(math.log(1 / 1e-9) / (2 * n))
    return p, ub, edges
