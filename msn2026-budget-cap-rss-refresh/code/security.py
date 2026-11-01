from __future__ import annotations

import math

DEFAULT_EPS_PA = 2.0 ** -64
AUTH_REPLENISH_BITS = 256
CONFIRM_BITS = 64
DEFAULT_MAX_REFRESH_SESSIONS = 2**16


def pa_margin_bits(eps_pa: float = DEFAULT_EPS_PA) -> float:
    """Leftover-hash-lemma margin for two-universal hashing.

    Using Delta <= 1/2 sqrt(2^(ell-Hmin)), solve for ell.
    """
    if not (0.0 < eps_pa < 1.0):
        raise ValueError("eps_pa must lie in (0,1)")
    return 2.0 * math.log2(1.0 / (2.0 * eps_pa))


def variable_transcript_support_bits(max_source_bits: int) -> int:
    """Conservative log2 support of a variable-length binary transcript.

    All source-dependent strings of length 0..Lambda number 2^(Lambda+1)-1,
    hence ceil(log2 |support|) <= Lambda+1.  This explicitly charges the public
    stopping time/length and avoids treating a realized data-dependent length
    as if it were a fixed leakage value.
    """
    if max_source_bits < 0:
        raise ValueError("max_source_bits must be non-negative")
    return int(max_source_bits) + 1


def extracted_length(
    n_blocks: int,
    h_min_per_block: float,
    transcript_support_bits: int,
    max_raw_bits: int,
    eps_pa: float = DEFAULT_EPS_PA,
) -> int:
    if n_blocks < 0 or max_raw_bits < 0 or transcript_support_bits < 0:
        raise ValueError("counts must be non-negative")
    h_total = float(n_blocks) * float(h_min_per_block)
    ell = math.floor(h_total - float(transcript_support_bits) - pa_margin_bits(eps_pa))
    return int(min(max(ell, 0), int(max_raw_bits)))


def delivered_application_length(pre_auth_length: int) -> int:
    return max(int(pre_auth_length) - AUTH_REPLENISH_BITS, 0)


def abort_aware_conditional_distance(eps_pa: float, p_accept_lower: float | None) -> float | None:
    """Optional accepted-branch distance from the abort-aware statement.

    The primary composable statement is p_acc*Delta_acc <= eps_pa and does not
    require a lower bound on p_acc.  If an *independent* lower bound on
    acceptance is available, this helper reports Delta_acc <= eps_pa/p_lb.
    """
    if p_accept_lower is None:
        return None
    p = float(p_accept_lower)
    if not (0.0 < p <= 1.0):
        raise ValueError("p_accept_lower must lie in (0,1]")
    return min(1.0, float(eps_pa) / p)


def total_failure_bound(
    *,
    eps_pa: float,
    eps_confirm: float,
    eps_auth: float,
    eps_certificate: float = 0.0,
    numerical_error: float = 0.0,
) -> float:
    vals = [eps_pa, eps_confirm, eps_auth, eps_certificate, numerical_error]
    if any(float(v) < 0 for v in vals):
        raise ValueError("failure terms must be non-negative")
    return min(1.0, float(sum(float(v) for v in vals)))


def lifetime_failure_bound(per_session_cryptographic_bound: float, max_sessions: int,
                           eps_epoch_certificate: float = 0.0) -> float:
    """Composable bound over a bounded qualification epoch.

    The externally justified source/Eve qualification failure is an epoch-level
    event and is therefore charged once.  Fresh per-session PA, confirmation
    and authentication failures compose by a union/hybrid bound across T
    refreshes.  This is tighter and semantically correct compared with
    multiplying the same epoch certificate failure T times.
    """
    if per_session_cryptographic_bound < 0 or max_sessions < 0 or eps_epoch_certificate < 0:
        raise ValueError("arguments must be non-negative")
    return min(1.0, float(eps_epoch_certificate)
               + float(per_session_cryptographic_bound) * int(max_sessions))
