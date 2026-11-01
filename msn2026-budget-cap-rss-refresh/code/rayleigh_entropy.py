"""Exact model-conditional entropy tools for BOLT-RSS v3.

Certified source model
----------------------
For each independent diversity block, H_A~CN(0,1),

    H_B = rho_AB H_A + sqrt(1-rho_AB^2) N_B,
    E   = a H_A + N_E,

where N_B~CN(0,1), N_E~CN(0,Sigma_N), and the noises are independent of
H_A and of each other.  Sigma_N may be fully correlated and must be Hermitian
positive definite.  Eve is granted the *entire continuous vector* E.

A Gaussian sufficient statistic reduces E to a scalar observation with
posterior SNR s_E=a^H Sigma_N^{-1} a and equivalent squared correlation
c_eff=s_E/(1+s_E).  For X=|H_A|^2 and Y=|H_E,eff|^2, the conditional law is a
noncentral chi-square, which yields a one-dimensional exact average guessing
probability for any fixed radial power quantizer.

The theorem assumes externally calibrated RSS scale so Alice's normalized
power follows Exp(1).  The live protocol never estimates this scale from the
payload; absence of a scale/source certificate causes fail-closed output.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
import math
from typing import Any, Dict, Sequence

import numpy as np
from scipy.integrate import quad, IntegrationWarning
from scipy.special import j0
from scipy.stats import ncx2
import warnings


@dataclass(frozen=True)
class EveGaussianSpec:
    """Externally supplied Gaussian side-information certificate parameters."""
    loading_real: tuple[float, ...]
    loading_imag: tuple[float, ...]
    noise_cov_real: tuple[tuple[float, ...], ...]
    noise_cov_imag: tuple[tuple[float, ...], ...]
    failure_probability: float = 0.0
    label: str = "externally-certified Gaussian Eve model"

    @staticmethod
    def from_arrays(loading: Sequence[complex], noise_cov: np.ndarray,
                    failure_probability: float = 0.0,
                    label: str = "externally-certified Gaussian Eve model") -> "EveGaussianSpec":
        a = np.asarray(loading, dtype=np.complex128).reshape(-1)
        s = np.asarray(noise_cov, dtype=np.complex128)
        if s.shape != (len(a), len(a)):
            raise ValueError("noise covariance dimension mismatch")
        return EveGaussianSpec(
            tuple(float(x.real) for x in a), tuple(float(x.imag) for x in a),
            tuple(tuple(float(v.real) for v in row) for row in s),
            tuple(tuple(float(v.imag) for v in row) for row in s),
            float(failure_probability), str(label))

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        a = np.asarray(self.loading_real, float) + 1j*np.asarray(self.loading_imag, float)
        sr = np.asarray(self.noise_cov_real, float)
        si = np.asarray(self.noise_cov_imag, float)
        return a.astype(np.complex128), (sr + 1j*si).astype(np.complex128)


@dataclass(frozen=True)
class AnalogEveCertificate:
    H: int
    effective_rho_upper: float
    p_guess_upper: float
    h_min_per_block: float
    quadrature_error: float
    tail_bound: float
    y_max: float
    valid: bool
    eve_dimension: int
    posterior_snr: float
    eps_certificate: float
    alice_scale_ratio_lower: float = 1.0
    alice_scale_ratio_upper: float = 1.0
    method: str = "exact-correlated-Rayleigh-continuous-vector-Eve"
    assumptions: str = (
        "independent unit-variance proper-complex-Gaussian diversity blocks; "
        "externally calibrated RSS scale; fixed equiprobable Exp(1) power quantizer; "
        "Eve has the full continuous Gaussian vector observation described by the "
        "certified loading vector and conditional-noise covariance; public transcript "
        "and certificate failure are accounted separately"
    )
    details: Dict[str, Any] | None = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def validate_eve_spec(spec: EveGaussianSpec) -> tuple[np.ndarray, np.ndarray]:
    a, sigma = spec.arrays()
    if len(a) < 1:
        raise ValueError("Eve dimension must be positive")
    if sigma.shape != (len(a), len(a)):
        raise ValueError("noise covariance dimension mismatch")
    if not np.allclose(sigma, sigma.conj().T, atol=1e-10, rtol=1e-10):
        raise ValueError("noise covariance must be Hermitian")
    eig = np.linalg.eigvalsh(sigma)
    if not np.all(np.isfinite(eig)) or float(np.min(eig)) <= 1e-12:
        raise ValueError("noise covariance must be positive definite")
    if not np.all(np.isfinite(a)):
        raise ValueError("invalid Eve loading")
    if not (0.0 <= float(spec.failure_probability) < 1.0):
        raise ValueError("certificate failure probability must lie in [0,1)")
    return a, sigma


def independent_eve_spec(rho_ae_upper: float | Sequence[float], eve_antennas: int = 1,
                         failure_probability: float = 0.0) -> EveGaussianSpec:
    """Build E_m=rho_m H_A+sqrt(1-rho_m^2)N_m with independent N_m."""
    if isinstance(rho_ae_upper, (list, tuple, np.ndarray)):
        r = np.abs(np.asarray(rho_ae_upper, dtype=float).reshape(-1))
    else:
        if int(eve_antennas) < 1:
            raise ValueError("eve_antennas must be positive")
        r = np.full(int(eve_antennas), abs(float(rho_ae_upper)), dtype=float)
    if len(r) < 1 or np.any(r < 0) or np.any(r >= 1.0):
        raise ValueError("certified per-antenna correlations must lie in [0,1)")
    sigma = np.diag(1.0-r*r).astype(np.complex128)
    return EveGaussianSpec.from_arrays(r.astype(np.complex128), sigma,
                                       failure_probability=failure_probability,
                                       label="independent per-antenna Eve upper-bound model")


def effective_eve_posterior_snr(spec: EveGaussianSpec) -> float:
    a, sigma = validate_eve_spec(spec)
    sol = np.linalg.solve(sigma, a)
    s = float(np.real(np.vdot(a, sol)))
    if s < -1e-10 or not np.isfinite(s):
        raise ValueError("invalid effective Eve SNR")
    return max(s, 0.0)


def effective_eve_correlation(spec: EveGaussianSpec) -> float:
    s = effective_eve_posterior_snr(spec)
    c = s/(1.0+s)
    return float(math.sqrt(max(min(c, 1.0), 0.0)))


def effective_colluding_eve_correlation(rho_ae_upper: float | Sequence[float],
                                        eve_antennas: int = 1) -> float:
    return effective_eve_correlation(independent_eve_spec(rho_ae_upper, eve_antennas))


def exponential_quantizer_edges(H: int) -> np.ndarray:
    if H < 1 or H > 8:
        raise ValueError("H must be in [1,8]")
    M = 2**H
    p = np.arange(M+1, dtype=float)/M
    edges = np.empty(M+1, dtype=float)
    edges[0] = 0.0
    edges[-1] = np.inf
    edges[1:-1] = -np.log1p(-p[1:-1])
    return edges


def quantize_power(power: np.ndarray, H: int) -> np.ndarray:
    x = np.asarray(power, dtype=float)
    if np.any(x < 0) or np.any(~np.isfinite(x)):
        raise ValueError("power samples must be finite and non-negative")
    edges = exponential_quantizer_edges(H)
    return np.clip(np.digitize(x, edges, right=False)-1, 0, 2**H-1).astype(int)


def gray(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=int)
    return x ^ (x >> 1)


def symbols_to_bits(symbols: np.ndarray, H: int) -> np.ndarray:
    g = gray(symbols)
    return ((g[:, None] >> np.arange(H-1, -1, -1)) & 1).astype(np.uint8).reshape(-1)


def _conditional_interval_probability(y: float, lo_edge: float, hi_edge: float, c: float) -> float:
    """P(lo_edge <= |G|^2 < hi_edge | effective Eve power y)."""
    if c <= 1e-15:
        flo=0.0 if lo_edge<=0 else 1.0-math.exp(-lo_edge)
        fhi=1.0 if not math.isfinite(hi_edge) else 1.0-math.exp(-hi_edge)
        return max(min(fhi-flo,1.0),0.0)
    if c >= 1.0-1e-15:
        return float(lo_edge <= y < hi_edge)
    scale=1.0-c
    nc=2.0*c*y/scale
    flo=0.0 if lo_edge<=0 else float(ncx2.cdf(2.0*lo_edge/scale,df=2,nc=nc))
    fhi=1.0 if not math.isfinite(hi_edge) else float(ncx2.cdf(2.0*hi_edge/scale,df=2,nc=nc))
    return max(min(fhi-flo,1.0),0.0)

def _conditional_bin_probabilities(y: float, H: int, c: float, scale_ratio: float=1.0) -> np.ndarray:
    """Exact bin probabilities for a fixed actual/nominal Alice power scale."""
    if scale_ratio<=0: raise ValueError("scale_ratio must be positive")
    edges=exponential_quantizer_edges(H)/float(scale_ratio)
    return np.array([_conditional_interval_probability(y,float(lo),float(hi),c)
                     for lo,hi in zip(edges[:-1],edges[1:])],dtype=float)

@lru_cache(maxsize=4096)
def continuous_eve_guessing_probability(H: int, effective_rho: float,
                                        y_max: float = 40.0,
                                        eps_abs: float = 2e-11,
                                        eps_rel: float = 2e-10,
                                        scale_ratio_lower: float = 1.0,
                                        scale_ratio_upper: float = 1.0) -> tuple[float,float,float]:
    """Conservative average guessing probability with certified scale uncertainty.

    Let the true Alice mean power divided by the nominal calibration scale lie
    in [mu_lo,mu_hi].  For nominal quantizer bin [t_j,t_{j+1}), the union of
    normalized |G|^2 intervals over every admissible mu is
    [t_j/mu_hi, t_{j+1}/mu_lo).  Giving Eve this expanded event can only
    increase her best bin probability, so integrating the largest expanded-bin
    probability is a rigorous upper bound uniform over the whole scale interval.
    For a singleton interval this reduces to the exact fixed-scale integral.
    """
    rho=abs(float(effective_rho)); lo=float(scale_ratio_lower); hi=float(scale_ratio_upper)
    if not (0.0<=rho<=1.0): raise ValueError("effective_rho must lie in [0,1]")
    if not (0.0<lo<=hi<math.inf): raise ValueError("invalid Alice scale-ratio interval")
    if rho>=1.0-1e-14: return 1.0,0.0,0.0
    c=rho*rho; edges=exponential_quantizer_edges(H)
    def integrand(y: float) -> float:
        best=0.0
        for e0,e1 in zip(edges[:-1],edges[1:]):
            z0=float(e0/hi)
            z1=math.inf if not np.isfinite(e1) else float(e1/lo)
            best=max(best,_conditional_interval_probability(y,z0,z1,c))
        return math.exp(-y)*best
    with warnings.catch_warnings():
        warnings.simplefilter("ignore",IntegrationWarning)
        val,err=quad(integrand,0.0,float(y_max),epsabs=eps_abs,epsrel=eps_rel,limit=800)
    tail=math.exp(-float(y_max))
    upper=min(1.0,max(0.0,val+abs(err)+tail))
    return float(upper),float(abs(err)),float(tail)


def analog_eve_certificate(H: int, *, eve_spec: EveGaussianSpec | None = None,
                           rho_ae_upper: float | None = None, eve_antennas: int = 1,
                           eps_certificate: float = 0.0, y_max: float = 40.0,
                           alice_scale_ratio_bounds: tuple[float,float] = (1.0,1.0)) -> AnalogEveCertificate:
    if eve_spec is None:
        if rho_ae_upper is None:
            raise ValueError("an external Eve certificate is required")
        eve_spec = independent_eve_spec(rho_ae_upper, eve_antennas,
                                        failure_probability=eps_certificate)
    a, _ = validate_eve_spec(eve_spec)
    s = effective_eve_posterior_snr(eve_spec)
    rho_eff = math.sqrt(s/(1+s)) if np.isfinite(s) else 1.0
    slo,shi=map(float,alice_scale_ratio_bounds)
    p, err, tail = continuous_eve_guessing_probability(H, rho_eff, y_max,
                                                       scale_ratio_lower=slo,scale_ratio_upper=shi)
    h = -math.log2(max(min(p, 1.0), 2.0**(-H)))
    valid = bool(np.isfinite(h) and h > 0.0 and rho_eff < 1.0)
    return AnalogEveCertificate(
        H=int(H), effective_rho_upper=float(rho_eff), p_guess_upper=float(p),
        h_min_per_block=float(max(h,0.0)), quadrature_error=err,
        tail_bound=tail, y_max=float(y_max), valid=valid,
        eve_dimension=int(len(a)), posterior_snr=float(s),
        eps_certificate=float(eve_spec.failure_probability),
        alice_scale_ratio_lower=slo, alice_scale_ratio_upper=shi,
        details={
            "eve_observation":"full continuous Gaussian vector",
            "conditional_noise":"full certified Hermitian covariance",
            "alice_quantizer":"fixed nominal unit-exponential RSS with certified scale interval",
            "scale_uncertainty_rule":"expanded-bin upper bound uniform over certified scale interval",
            "sequence_rule":"independent-block additivity of average conditional min-entropy",
            "certificate_label":eve_spec.label,
        })


def gaussian_cmi_upper_bound(rho_ab: float, eve_spec: EveGaussianSpec) -> float:
    """Loose source-model SK upper benchmark I(H_A;H_B|E), bit/block.

    This is an information-theoretic upper benchmark for the underlying complex
    Gaussian source and is intentionally not presented as an achievable rate of
    the RSS quantizer/protocol.
    """
    r = abs(float(rho_ab))
    if not (0.0 <= r < 1.0):
        if r >= 1.0: return math.inf
        raise ValueError("rho_ab must lie in [0,1)")
    s = effective_eve_posterior_snr(eve_spec)
    var_a_given_e = 1.0/(1.0+s)
    noise_b = 1.0-r*r
    if noise_b <= 0: return math.inf
    return float(math.log2(1.0 + (r*r*var_a_given_e)/noise_b))


def spatial_correlation_upper(d_over_lambda: float, margin: float=0.0) -> float:
    """Illustrative isotropic-scattering sensitivity only; never a live certificate."""
    base = abs(float(j0(2.0*math.pi*float(d_over_lambda))))
    return float(min(max(base+float(margin),0.0),0.999999))


def simulate_correlated_rayleigh_blocks(n: int, rho_ab: float, rho_ae: float,
                                        seed: int) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    spec = independent_eve_spec(rho_ae, 1)
    a,b,e = simulate_correlated_rayleigh_vector(n, rho_ab, spec, seed)
    return a,b,e[:,0]


def simulate_correlated_rayleigh_vector(n: int, rho_ab: float, eve_spec: EveGaussianSpec,
                                         seed: int) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    if n <= 0: raise ValueError("n must be positive")
    r = abs(float(rho_ab))
    if not (0.0 <= r <= 1.0): raise ValueError("rho_ab must lie in [0,1]")
    loading, sigma = validate_eve_spec(eve_spec)
    rng = np.random.default_rng(int(seed))
    def cn(shape):
        return (rng.normal(size=shape)+1j*rng.normal(size=shape))/math.sqrt(2.0)
    ha = cn(n)
    nb = cn(n)
    hb = r*ha + math.sqrt(max(1-r*r,0.0))*nb
    L = np.linalg.cholesky(sigma)
    z = cn((n,len(loading)))
    ne = z @ L.conj().T
    he = ha[:,None]*loading[None,:] + ne
    return np.abs(ha)**2, np.abs(hb)**2, np.abs(he)**2


def monte_carlo_p_guess(H: int, effective_rho: float, n: int=200_000,
                        seed: int=1234) -> tuple[float,float]:
    """Vectorized independent Monte-Carlo cross-check of the guessing integral."""
    rng=np.random.default_rng(seed)
    c=float(effective_rho)**2
    y=rng.exponential(1.0,size=int(n))
    edges=exponential_quantizer_edges(H)
    if c <= 1e-15:
        probs=np.diff(1.0-np.exp(-edges))
        q=np.full(len(y),float(np.max(probs)))
    elif c >= 1.0-1e-12:
        # Eve essentially knows Alice's complex coefficient, hence its power/bin.
        q=np.ones(len(y),dtype=float)
    else:
        scale=(1.0-c)/2.0
        nc=2.0*c*y/(1.0-c)
        vals=[]
        for lo,hi in zip(edges[:-1],edges[1:]):
            flo=np.zeros_like(y) if lo<=0 else ncx2.cdf(lo/scale,df=2,nc=nc)
            fhi=np.ones_like(y) if not np.isfinite(hi) else ncx2.cdf(hi/scale,df=2,nc=nc)
            vals.append(np.maximum(fhi-flo,0.0))
        q=np.max(np.vstack(vals),axis=0)
    return float(q.mean()), float(1.96*q.std(ddof=1)/math.sqrt(len(q)))
