"""Standard PLKG key-rate metrics and source benchmarks (revision v4).

Reports, on the same live sessions as the main grid:
  * certified key rate in secret bits per channel use (cu), where one diversity
    block costs two channel uses (Alice->Bob and Bob->Alice probe, TDD RSS);
  * the same rate when every public-discussion bit is also charged as a
    channel use at spectral efficiency eta (eta=1 conservative);
  * conventional (non-certified) rates on the same sessions, so that the
    number can be placed next to KGRs reported in the PLKG literature:
      R_quant  : quantizer output bits (no reconciliation)
      R_recon  : reconciled bits minus *realized* public bits (Eve-agnostic)
      R_realEve: Eve min-entropy minus *realized* transcript (not abort-aware)
  * asymptotic one-way secret-key rate of the same quantized RSS source,
    I(Q_A;Q_B) - I(Q_A;Y_E)  (Csiszar-Korner), and the CSI-level conditional
    mutual-information upper benchmark I(H_A;H_B|E).
"""
from __future__ import annotations
import json, math
from functools import lru_cache
from pathlib import Path
import numpy as np
from scipy.integrate import quad
from rayleigh_entropy import (exponential_quantizer_edges, _conditional_interval_probability,
                              analog_eve_certificate, independent_eve_spec, gaussian_cmi_upper_bound)
from security import pa_margin_bits, AUTH_REPLENISH_BITS
from run_main_grid_v4 import RAES, N, NP, NPAY

ROOT = Path(__file__).resolve().parent


def _ent(p):
    p = np.asarray(p, float); p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


@lru_cache(maxsize=None)
def mi_QA_QB(H: int, rab: float) -> float:
    e = exponential_quantizer_edges(H); M = 2 ** H; c = rab * rab
    P = np.zeros((M, M))
    for i in range(M):
        lo, hi = e[i], e[i + 1]
        for j in range(M):
            f = lambda x: math.exp(-x) * _conditional_interval_probability(x, e[j], e[j + 1], c)
            P[i, j] = quad(f, lo, hi if np.isfinite(hi) else 60.0, limit=400)[0]
    P /= P.sum()
    return _ent(P.sum(1)) + _ent(P.sum(0)) - _ent(P.ravel())


@lru_cache(maxsize=None)
def mi_QA_Y(H: int, rae: float) -> float:
    e = exponential_quantizer_edges(H); c = rae * rae
    def g(y):
        p = [_conditional_interval_probability(y, e[j], e[j + 1], c) for j in range(len(e) - 1)]
        return math.exp(-y) * _ent(np.array(p) / max(sum(p), 1e-300))
    return H - quad(g, 0, 50, limit=400)[0]


def one_way_rate(H, rab, rae):
    return max(mi_QA_QB(H, rab) - mi_QA_Y(H, rae), 0.0)


def main():
    live = json.loads((ROOT / 'results_live_validation_v4.json').read_text())['rows']
    tr = json.loads((ROOT / 'eval_traces_v4.json').read_text())
    trk = {(x['rho_ab'], x['t']): x for x in tr}
    hs = {rae: {H: float(analog_eve_certificate(H, eve_spec=independent_eve_spec(rae, 1)).h_min_per_block) for H in (1, 2)} for rae in RAES}
    delta = pa_margin_bits()
    cells = []
    for rab in sorted({x['rho_ab'] for x in live}):
        for rae in RAES:
            cell = {'rho_ab': rab, 'rho_ae': rae, 'policies': {}}
            for pol in ('BOLT', 'BOLT-LCB', 'q99'):
                rows = [x for x in live if x['rho_ab'] == rab and x['rho_ae'] == rae and x['policy'] == pol]
                dl = np.array([x['delivered'] for x in rows], float)
                # discussion bits: pilot labels + 2 counts + (H,cap) + realized source-dependent
                # transcript + confirmation-hash seed + PA seed + auth tag (public randomness included)
                disc = []
                for x in rows:
                    nH = NPAY * x['H']
                    d = NP * 2 + 64 + 40 + x['leak']
                    if x['cap'] > 0:
                        d += nH + 63 + 64  # confirmation Toeplitz seed + Cascade permutation seed
                    if x['pre_auth'] > 0:
                        d += nH + x['pre_auth'] - 1 + 128  # PA Toeplitz seed + WC tag
                    disc.append(d)
                disc = np.array(disc, float)
                cu_probe = 2 * N
                cell['policies'][pol] = {
                    'R_block': float(dl.mean() / N),
                    'R_cu_probe': float(dl.mean() / cu_probe),
                    'R_cu_total_eta1': float(np.mean(dl / (cu_probe + disc))),
                    'R_cu_total_eta1_noseed': float(np.mean(dl / (cu_probe + disc - np.array([(NPAY * x['H'] + 127 if x['cap'] > 0 else 0) + (NPAY * x['H'] + x['pre_auth'] - 1 if x['pre_auth'] > 0 else 0) for x in rows])))),
                    'mean_discussion_bits': float(disc.mean())}
            # conventional rates on the same sessions with H=2 (and H=1) uncapped Cascade
            conv = {}
            for H in (1, 2):
                rq, rr, re_ = [], [], []
                for t in range(20):
                    x = trk[(rab, t)][f'H{H}']
                    rq.append(NPAY * H / N)
                    if x['C']:
                        rr.append(max(NPAY * H - x['L'], 0) / N)
                        re_.append(max(math.floor(NPAY * hs[rae][H] - x['L'] - delta) - AUTH_REPLENISH_BITS, 0) / N)
                    else:
                        rr.append(0.0); re_.append(0.0)
                conv[f'H{H}'] = {'R_quant': float(np.mean(rq)), 'R_recon_realized': float(np.mean(rr)),
                                 'R_realized_eve': float(np.mean(re_))}
            cell['conventional'] = conv
            cell['one_way_rate_per_block'] = {f'H{H}': float(NPAY / N * one_way_rate(H, rab, rae)) for H in (1, 2)}
            cell['I_QAQB'] = {f'H{H}': mi_QA_QB(H, rab) for H in (1, 2)}
            cell['I_QAY'] = {f'H{H}': mi_QA_Y(H, rae) for H in (1, 2)}
            cell['h_min'] = {f'H{H}': hs[rae][H] for H in (1, 2)}
            cell['csi_cmi_upper_per_block'] = float(NPAY / N * gaussian_cmi_upper_bound(rab, independent_eve_spec(rae, 1)))
            cells.append(cell)
    out = {'version': 'bolt-rss-v4-kgr', 'channel_uses_per_block': 2, 'cells': cells}
    (ROOT / 'results_kgr_metrics_v4.json').write_text(json.dumps(out, indent=1))
    for c in cells:
        if c['rho_ab'] >= .995 and c['rho_ae'] in (.1, .3, .5, .7):
            b = c['policies']['BOLT']; q = c['policies']['q99']
            print(c['rho_ab'], c['rho_ae'], 'BOLT blk %.3f cu %.4f cu_tot %.4f | q99 cu %.4f | conv H2 %s | 1way %s | cmi %.2f' % (
                b['R_block'], b['R_cu_probe'], b['R_cu_total_eta1'], q['R_cu_probe'],
                {k: round(v, 3) for k, v in c['conventional']['H2'].items()},
                {k: round(v, 3) for k, v in c['one_way_rate_per_block'].items()}, c['csi_cmi_upper_per_block']))


if __name__ == '__main__':
    main()
