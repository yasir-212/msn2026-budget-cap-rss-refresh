"""
Cascade information reconciliation (Brassard & Salvail, EUROCRYPT '93),
with the standard binary-search error location and the cascading backtrack
to earlier passes.

WHY THIS REPLACES THE ITERATIVE HAMMING(t=1) RECONCILER (this revision):
the previous reconciler in this project applied a fixed Hamming(t=1) code
with a fresh public interleaver each round. It works, and it is honest
(all leakage is counted), but it was MEASURED in this project's own
calibration to cost roughly 3-9x the Slepian-Wolf bound n*h2(p) -- because
a Hamming(t=1) block of length 2^m-1 spends m syndrome bits whether or not
it contains an error, and can only ever fix ONE error per block per round,
so it must use small blocks (high overhead) whenever p is not tiny.

That inefficiency was the direct, root cause of this project's most-cited
open problem: the total leak budget exceeded the payload at high raw
bit-mismatch rates, so reconciliation could not converge net-positive and
every scheme (including ours) collapsed to zero net secret bits on the
harder synthetic trace at sigma >= 2.0 dB.

This implementation conservatively charges both Alice's parity and Bob's
mismatch/branch response: 2 source-dependent bits per pass block, plus two
bits per binary-search step.  Its
backtracking step corrects errors in EARLIER passes for free (no new
parity exchange: after a correction at a public position, both parties
locally recompute which of their previously-exchanged block parities have
flipped). Published and reproduced efficiencies are ~1.1-1.3x the Shannon
bound over p in [0.01, 0.15], i.e. roughly 3-6x less leakage than the
Hamming scheme it replaces at the same p.

LEAKAGE ACCOUNTING (conservative, fully counted -- nothing is free):
  * 2 bits per block for every initial per-pass exchange (Alice parity plus
    Bob mismatch decision), including error-free blocks.
  * 2 bits per binary-search halving step (Alice parity plus Bob branch),
    including searches triggered by cascade backtracking.
  * 64 tag bits plus one decision bit for each attempted pass confirmation (see below).
No leakage is omitted or amortized away. This matters because the whole
security argument downstream (see SECURITY_ANALYSIS.md) subtracts the
counted leakage from the smooth min-entropy before privacy amplification;
undercounting here would silently overstate the final key's secrecy.

CONVERGENCE / FAIL-CLOSED: Cascade does not guarantee zero residual errors
(a block containing an even number of errors passes its parity check). We
therefore run additional passes while a budget remains, and perform an explicit universal-hash confirmation after each completed pass: both parties exchange a 64-bit universal hash
of their reconciled strings. If it does not match, the session is declared
NOT converged and the caller derives a zero-length key (fail-closed), the
same convention this project has used throughout. The 64-bit confirmation
is counted as leakage.
"""
import math

import numpy as np


def _block_parity_mismatch(a, b, idx):
    return (int(a[idx].sum()) & 1) != (int(b[idx].sum()) & 1)


def _binary_search_correct(a, b, block_idx):
    """Locate one error and return the compact public interaction.

    At every halving step Alice reveals the left-half parity and Bob reveals
    one branch bit (left mismatches or not).  Both source-dependent bits are
    charged.  The final position is implied by the branch sequence.
    """
    idx = block_idx
    steps: list[tuple[int, int]] = []
    while len(idx) > 1:
        half = len(idx) // 2
        left = idx[:half]
        pa = int(a[left].sum()) & 1
        pb = int(b[left].sum()) & 1
        branch_left = int(pa != pb)
        steps.append((pa, branch_left))
        idx = left if branch_left else idx[half:]
    pos = int(idx[0])
    b[pos] ^= 1
    return pos, steps


def _toeplitz_tag(bits, out_len, seed_bits):
    bits = np.asarray(bits, dtype=np.int8)
    if out_len <= 0:
        return np.zeros(0, dtype=np.uint8)
    idx = np.arange(out_len)[:, None] + len(bits) - 1 - np.arange(len(bits))[None, :]
    return ((seed_bits[idx] @ bits) & 1).astype(np.uint8)


def cascade_reconcile(a_flat, b_flat, est_bmr=None, max_passes=6,
                       confirm_bits=64, leak_cap_factor=1.0, seed=0,
                       confirm_seed_bits=None, max_source_leak_bits=None):
    """Reconcile b_flat to a_flat via Cascade.

    a_flat, b_flat : 0/1 integer arrays of equal length.
    est_bmr        : estimate of the raw mismatch rate, used only to size
                     the first pass's blocks (k1 = 0.73/p, the standard
                     Brassard-Salvail heuristic). If None, it is measured
                     directly from the inputs -- legitimate in simulation,
                     and in a real deployment this estimate comes from the
                     pilot slice the protocol already exchanges.
    max_passes     : maximum Cascade passes (block size doubles each pass).
    confirm_bits   : bits spent on the final confirmation hash.
    leak_cap_factor: abort once total leak exceeds this many bits per input
                     bit (safety valve for hopeless channels; 1.0 means we
                     stop as soon as leakage would consume the entire
                     payload, since no positive net key is possible then).

    Returns dict(out, leak_bits, rounds, final_bmr, converged).
    """
    n = len(a_flat)
    if n == 0:
        return dict(out=np.zeros(0, dtype=int), leak_bits=0, rounds=0,
                    final_bmr=0.0, converged=True)

    a = np.asarray(a_flat, dtype=np.int8).copy()
    b = np.asarray(b_flat, dtype=np.int8).copy()

    if est_bmr is None:
        est_bmr = float(np.mean(a != b))
    p = min(max(est_bmr, 1e-4), 0.45)

    k1 = max(4, int(round(0.73 / p)))
    k1 = min(k1, n)

    total_leak = 0
    legacy_leak_cap = int(leak_cap_factor * n)
    hard_cap = legacy_leak_cap if max_source_leak_bits is None else min(legacy_leak_cap, int(max_source_leak_bits))
    if hard_cap < 0:
        raise ValueError("max_source_leak_bits must be non-negative")
    budget_abort = False
    # Smallest hard cap under which this exact run would pass every
    # pre-transmission budget check (worst-case reservations included).  For a
    # fixed realization and public seed, a run under cap Lambda is identical to
    # this run up to the first failed check, so it accepts iff converged and
    # Lambda >= required_cap.  Used for exact offline cap replay.
    required_cap = 0

    pass_blocks = []   # per pass: list of index arrays
    pass_bitmap = []   # per pass: bit -> block id within that pass
    rng = np.random.default_rng(seed)  # PUBLIC, seeded: both parties agree
    rounds_used = 0

    # FULL INTERACTIVE TRANSCRIPT (review item). Previously only the final
    # corrected bit array was authenticated, which does NOT protect the
    # public discussion an active attacker would actually manipulate: block
    # parities, binary-search queries and responses, and correction
    # positions all travelled unauthenticated. We now record every public
    # message in order so the caller can authenticate the whole exchange.
    if confirm_seed_bits is None:
        raise ValueError("full public confirmation seed is required")
    confirm_seed = np.asarray(confirm_seed_bits, dtype=np.uint8).astype(int)
    if len(confirm_seed) != n + confirm_bits - 1:
        raise ValueError("confirmation seed must contain n+confirm_bits-1 bits")
    # Publish/bind the source-independent confirmation hash seed once.  It is
    # reused only within this reconciliation session; per-pass source tags are
    # still separately charged as leakage.
    transcript = [b"CASCADE-BOLT-RSS-v3|n=%d|k1=%d|seed=%d|confirm-seed=" % (n, k1, seed)
                  + np.packbits(confirm_seed.astype(np.uint8)).tobytes()]

    for pi in range(max_passes):
        rounds_used = pi + 1
        kk = min(k1 * (2 ** pi), n)
        perm = np.arange(n) if pi == 0 else rng.permutation(n)
        n_blocks = int(math.ceil(n / kk))
        blocks = [perm[j * kk:(j + 1) * kk] for j in range(n_blocks)]
        blocks = [blk for blk in blocks if len(blk) > 0]
        bitmap = np.empty(n, dtype=np.int64)
        for bid, blk in enumerate(blocks):
            bitmap[blk] = bid
        pass_blocks.append(blocks)
        pass_bitmap.append(bitmap)

        # Alice reveals one parity per block and Bob reveals one mismatch
        # decision per block.  Charging both bits upper-bounds the complete
        # source-dependent interaction, rather than charging Alice's parity
        # alone and silently treating Bob's branch messages as free.
        par_a = np.array([(int(a[blk].sum()) & 1) for blk in blocks], dtype=np.uint8)
        mismatch = np.array([int(_block_parity_mismatch(a, b, blk)) for blk in blocks], dtype=np.uint8)
        pass_cost = 2 * len(blocks)
        required_cap = max(required_cap, total_leak + pass_cost)
        # The cap is enforced *before* the source-dependent exchange.  Whether
        # this fixed-size pass can fit is determined entirely by public state
        # (pilot, pass number, block partition and the precommitted cap).
        if total_leak + pass_cost > hard_cap:
            budget_abort = True
            transcript.append(b"BUDGET-ABORT|before-pass=%d" % pi)
            break
        total_leak += pass_cost
        transcript.append(b"PASS|%d|k=%d|nb=%d|pa=" % (pi, kk, len(blocks))
                          + np.packbits(par_a).tobytes()
                          + b"|m=" + np.packbits(mismatch).tobytes())

        queue = [(pi, bid) for bid, flag in enumerate(mismatch) if flag]

        while queue:
            qpi, qbid = queue.pop()
            blk = pass_blocks[qpi][qbid]
            # The block may already have been repaired by an earlier
            # cascade step; both parties can check this locally from the
            # parities they already exchanged plus the public correction
            # positions, so no new leakage is incurred here.
            if not _block_parity_mismatch(a, b, blk):
                continue
            # A binary search on a public block of size m takes at most
            # ceil(log2 m) halving steps.  Reserve that worst-case cost before
            # revealing any path bit, so a budget abort itself does not depend
            # on a hidden path prefix.
            max_steps = int(math.ceil(math.log2(max(len(blk), 1))))
            required_cap = max(required_cap, total_leak + 2 * max_steps)
            if total_leak + 2 * max_steps > hard_cap:
                budget_abort = True
                transcript.append(b"BUDGET-ABORT|before-search|pass=%d|blk=%d" % (qpi, qbid))
                break
            pos, steps = _binary_search_correct(a, b, blk)
            total_leak += 2 * len(steps)
            packed = np.packbits(np.array(steps, dtype=np.uint8).reshape(-1)).tobytes()
            transcript.append(b"BSEARCH|pass=%d|blk=%d|steps=%d|pb="
                              % (qpi, qbid, len(steps)) + packed)

            # CASCADE BACKTRACK: flipping bit `pos` flips the parity of
            # every block (in every pass) that contains it. Any such block
            # that now mismatches contains a further odd error count and is
            # queued. This detection is free: both sides recompute it from
            # already-exchanged parities.
            for opi in range(len(pass_blocks)):
                if opi == qpi:
                    continue
                obid = int(pass_bitmap[opi][pos])
                oblk = pass_blocks[opi][obid]
                if _block_parity_mismatch(a, b, oblk):
                    queue.append((opi, obid))

            if budget_abort:
                break

        if budget_abort:
            break

        # Distributed early stopping: after every completed pass, both parties
        # publish a universal-hash confirmation under the SAME full public
        # Toeplitz seed.  A match is an observable protocol event and therefore
        # legitimately stops Cascade; there is no private equality oracle.
        # Reusing the public hash function within this session is safe for the
        # fixed pass outputs; a union bound over attempted confirmations is
        # charged to correctness.  Each source-dependent tag and the one-bit
        # decision are included in the hard leakage cap.
        confirm_cost = confirm_bits + 1
        required_cap = max(required_cap, total_leak + confirm_cost)
        if total_leak + confirm_cost > hard_cap:
            budget_abort = True
            transcript.append(b"BUDGET-ABORT|before-confirm|pass=%d" % pi)
            break
        total_leak += confirm_cost
        ta_bits = _toeplitz_tag(a, confirm_bits, confirm_seed)
        tb_bits = _toeplitz_tag(b, confirm_bits, confirm_seed)
        agreed = bool(np.array_equal(ta_bits, tb_bits))
        confirm_attempts = pi + 1
        transcript.append(b"CONFIRM|pass=%d|tagA=" % pi + np.packbits(ta_bits).tobytes())
        transcript.append(b"DECISION|pass=%d|" % pi + (b"ACCEPT" if agreed else b"CONTINUE"))
        if agreed:
            final_bmr = float(np.mean(a != b))
            return dict(out=b.astype(int), leak_bits=int(total_leak), rounds=rounds_used,
                        final_bmr=final_bmr, converged=True,
                        confirm_collision_prob=min(1.0, confirm_attempts * 2.0 ** (-confirm_bits)),
                        confirmation_attempts=confirm_attempts, budget_abort=False,
                        required_cap=int(required_cap),
                        transcript=b"\n".join(transcript))

    # If every precommitted pass completed without a confirmation match, or if
    # the hard cap was hit, the protocol aborts and releases no key.
    final_bmr = float(np.mean(a != b))
    confirm_attempts = rounds_used if not budget_abort else max(rounds_used-1, 0)
    if budget_abort:
        transcript.append(b"FINAL|BUDGET-ABORT")
    else:
        transcript.append(b"FINAL|ABORT|no-confirmation-match")
    return dict(out=b.astype(int), leak_bits=int(total_leak), rounds=rounds_used,
                final_bmr=final_bmr, converged=False,
                confirm_collision_prob=min(1.0, confirm_attempts * 2.0 ** (-confirm_bits)),
                confirmation_attempts=confirm_attempts, budget_abort=bool(budget_abort),
                required_cap=int(required_cap),
                transcript=b"\n".join(transcript))
