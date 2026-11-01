"""Figures for the revised manuscript (v4), generated from frozen JSON results."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

C = Path(__file__).resolve().parent
OUT = C.parent / 'outputs' / 'figures'; OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.size': 7.5, 'axes.labelsize': 7.5, 'axes.titlesize': 7.5, 'legend.fontsize': 6.3,
                     'xtick.labelsize': 6.8, 'ytick.labelsize': 6.8, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
                     'axes.grid': True, 'grid.alpha': 0.25, 'grid.linewidth': 0.4, 'lines.linewidth': 1.2,
                     'lines.markersize': 3.6, 'axes.spines.top': False, 'axes.spines.right': False,
                     'pdf.fonttype': 42, 'ps.fonttype': 42,
                     'font.family': 'sans-serif', 'font.sans-serif': ['Liberation Sans', 'Arial', 'Helvetica', 'DejaVu Sans']})  # TrueType fonts (IEEE PDF eXpress)
# categorical slots (fixed order): BOLT, BOLT-LCB, q95, q99; oracle/fixed in neutral ink
COL = {'BOLT': '#2a78d6', 'BOLT-LCB': '#eb6834', 'q95': '#1baf7a', 'q99': '#b07a00', 'oracle': '#6b6b66', 'fixedcap': '#8a8a84'}
MK = {'BOLT': 'o', 'BOLT-LCB': 's', 'q95': '^', 'q99': 'v', 'oracle': None, 'fixedcap': 'D'}
LS = {'BOLT': '-', 'BOLT-LCB': '-', 'q95': '--', 'q99': '--', 'oracle': ':', 'fixedcap': '-.'}
W = 3.45


def boot(x, seed=1, n=4000):
    x = np.asarray(x, float); rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return np.quantile(m, .025), np.quantile(m, .975)


def fig_rate():
    D = json.loads((C / 'results_main_grid_v4_exact.json').read_text())
    fig, axs = plt.subplots(1, 2, figsize=(W, 1.45), sharey=True)
    for ax, rab in zip(axs, (0.9975, 0.995)):
        rows = [r for r in D['rows'] if r['rho_ab'] == rab]
        x = np.array([r['rho_ae'] for r in rows])
        for nm in ('oracle', 'BOLT', 'BOLT-LCB', 'q95', 'q99', 'fixedcap(H=2)'):
            y = np.array([r['policies'][nm]['mean_rate'] for r in rows])
            if nm == 'oracle':
                ax.plot(x, y, ls=':', color=COL[nm], lw=1.0, label='oracle')
                continue
            if nm == 'fixedcap(H=2)':
                ax.plot(x, y, color=COL['fixedcap'], marker=MK['fixedcap'], ls=LS['fixedcap'], lw=1.0, ms=3.0, label='fixed cap', zorder=1)
                continue
            ci = np.array([boot(r['policies'][nm]['rates'], 3) for r in rows])
            ax.errorbar(x, y, yerr=[y - ci[:, 0], ci[:, 1] - y], color=COL[nm], marker=MK[nm], ls=LS[nm],
                        capsize=1.2, elinewidth=.5, label=nm, markerfacecolor='white' if nm == 'BOLT-LCB' else COL[nm])
        ax.set_title(rf'$|\rho_{{AB}}|={rab}$', pad=2)
        ax.set_xlabel(r'certified Eve bound $\bar\rho_{AE}$')
        ax.set_ylim(bottom=0)
    axs[0].set_ylabel('certified rate (bit/block)')
    sec = axs[1].secondary_yaxis('right', functions=(lambda v: v / 2, lambda v: v * 2))
    sec.set_ylabel('bit/channel use', fontsize=6.8)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=6, frameon=False, bbox_to_anchor=(0.5, 1.10), handlelength=1.6, columnspacing=.7, fontsize=6.2)
    fig.tight_layout(w_pad=0.6)
    fig.savefig(OUT / 'rate_vs_eve_v4.pdf'); fig.savefig(OUT / 'rate_vs_eve_v4.png', dpi=220); plt.close(fig)


def fig_cv():
    D = json.loads((C / 'results_cv_holdout_v4.json').read_text())
    fig, axs = plt.subplots(1, 2, figsize=(W, 1.45))
    ax = axs[0]
    sizes = sorted(int(s) for s in D['size_curve'])
    for nm in ('oracle', 'BOLT', 'BOLT-LCB', 'fixedcap', 'q95', 'q99'):
        m = [D['size_curve'][str(s)][nm]['mean'] for s in sizes]
        zo = 6 if nm == 'fixedcap' else 2
        if nm == 'oracle':
            ax.plot(sizes, m, ls=':', color=COL[nm], label='oracle'); continue
        lo = [D['size_curve'][str(s)][nm]['q05'] for s in sizes]; hi = [D['size_curve'][str(s)][nm]['q95'] for s in sizes]
        ax.plot(sizes, m, color=COL[nm], marker=MK[nm], ls=LS[nm], label=nm if nm != 'fixedcap' else 'fixed cap',
                markerfacecolor='white' if nm == 'BOLT-LCB' else COL[nm], zorder=zo,
                ms=2.8 if nm == 'fixedcap' else None, mec='white' if nm == 'fixedcap' else None, mew=0.4 if nm == 'fixedcap' else None)
        if nm in ('BOLT', 'BOLT-LCB'):
            ax.fill_between(sizes, lo, hi, color=COL[nm], alpha=.12, lw=0)
    ax.set_xlabel('library size ($H=2$ records)'); ax.set_ylabel('held-out rate (bit/block)'); ax.set_ylim(bottom=0)
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=6, frameon=False, bbox_to_anchor=(0.5, 1.10), handlelength=1.6, columnspacing=.8, fontsize=6.3)
    ax.set_title('(a) library size', pad=2)
    ax = axs[1]
    ax.plot([0, 1], [0, 1], color='#9a9a94', lw=.8, ls='-')
    for H in ('2', '1'):
        rel = D['cv'][H]['reliability']
        for nm in ('BOLT', 'BOLT-LCB'):
            s = [r['score'] for r in rel[nm]]; a = [r['realized'] for r in rel[nm]]
            ax.plot(s, a, color=COL[nm], marker=MK[nm], ls=LS[nm] if H == '2' else ':',
                    markerfacecolor=('white' if nm == 'BOLT-LCB' else COL[nm]), label=f'{nm}, $H={H}$', lw=1.0)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_xlabel('predicted acceptance score'); ax.set_ylabel('realized acceptance')
    ax.text(0.98, 0.04, 'solid: $H=2$\ndotted: $H=1$\ngray: ideal', transform=ax.transAxes, ha='right', va='bottom', fontsize=6.4, color='#444441')
    ax.set_title('(b) score calibration', pad=2)
    fig.tight_layout(w_pad=0.8)
    fig.savefig(OUT / 'cv_coverage_v4.pdf'); fig.savefig(OUT / 'cv_coverage_v4.png', dpi=220); plt.close(fig)


def fig_cert():
    D = json.loads((C / 'results_certificate_acquisition_v4.json').read_text())
    fig, axs = plt.subplots(1, 2, figsize=(W, 1.46))
    ax = axs[0]
    A = [r for r in D['A_scale']['rate_vs_effort'] if r['eps_src'] == 1e-9]
    ax.plot([r['M'] for r in A], [r['rate_9975_rae0.1'] for r in A], color=COL['BOLT'], marker='o', ms=2.8, label='scale, BOLT')
    ax.plot([r['M'] for r in A], [r['rate_q99_9975_rae0.1'] for r in A], color=COL['q99'], marker='v', ms=3.0, ls='--', label='scale, q99')
    B = [r for r in D['B_survey']['rate_vs_M'] if r['S'] == 25]
    ax.plot([r['M'] for r in B], [r['rate_9975_bolt'] for r in B], color=COL['BOLT'], marker='s', ms=4.4, ls='-.', markerfacecolor='none', label='survey, BOLT')
    ax.plot([r['M'] for r in B], [r['rate_9975_q99'] for r in B], color=COL['q99'], marker='s', ms=4.4, ls=':', markerfacecolor='none', label='survey, q99')
    ax.axhline(json.loads((C / 'results_main_grid_v4_exact.json').read_text())['summary']['headline_point']['BOLT'], color='#9a9a94', lw=.8)
    ax.set_xscale('log'); ax.set_xlabel('qualification samples'); ax.set_ylabel('rate (bit/block)')
    ax.set_xticks([1e3, 1e4, 1e5, 1e6]); ax.minorticks_off()
    ax.set_ylim(0, 1.05); ax.set_yticks([0, .2, .4, .6])
    ax.legend(loc='upper left', ncol=2, frameon=False, handlelength=1.0, fontsize=5.5, columnspacing=.3, borderaxespad=0.1, handletextpad=.2)
    ax.set_title('(a) cost of certification', pad=2)
    ax = axs[1]
    from scipy.special import j0
    d = np.linspace(0.5, 20, 600)
    env = np.array([np.abs(j0(2 * np.pi * np.linspace(x, 40, 4000))).max() for x in d])
    ax.plot(d, env, color=COL['BOLT'], label='isotropic envelope')
    ni = D['C_guard_zone']['nonisotropic']
    for kappa, mk, col in ((2, '^', COL['q95']), (8, 's', COL['BOLT-LCB'])):
        pts = [r for r in ni if r['kappa'] == kappa]
        ax.plot([r['d_min_over_lambda'] for r in pts], [r['sup_rho'] for r in pts], marker=mk, ls='--', color=col, label=rf'von Mises $\kappa={kappa}$')
        ax.text(1.15, pts[0]['sup_rho'] + (0.05 if kappa == 8 else 0.035), rf'$\kappa={kappa}$', fontsize=6.4, color=col)
    ax.set_xscale('log'); ax.set_xlabel(r'Eve distance $d_{\min}/\lambda$'); ax.set_ylabel(r'bound on $|\rho_{AE}|$'); ax.set_ylim(0, 1.0)
    ax.text(0.53, 0.14, 'Clarke', fontsize=6.4, color=COL['BOLT'])
    ax.set_title('(b) guard-zone certificate', pad=2)
    fig.tight_layout(w_pad=0.8)
    fig.savefig(OUT / 'certificates_v4.pdf'); fig.savefig(OUT / 'certificates_v4.png', dpi=220); plt.close(fig)


def fig_entropy():
    """Certified per-block min-entropy vs effective Eve correlation (Theorem 1, Prop. 1)."""
    import sys; sys.path.insert(0, str(C))
    from rayleigh_entropy import analog_eve_certificate
    r = np.linspace(0, 0.85, 18)
    fig, ax = plt.subplots(figsize=(W, 1.35))
    cols = {1: COL['q95'], 2: COL['BOLT'], 3: COL['BOLT-LCB']}
    for H in (1, 2, 3):
        ex = [analog_eve_certificate(H, rho_ae_upper=x).h_min_per_block for x in r]
        sc = [analog_eve_certificate(H, rho_ae_upper=x, alice_scale_ratio_bounds=(0.894, 1.125)).h_min_per_block for x in r]
        ax.plot(r, ex, color=cols[H], ls='-', label=f'$H={H}$, exact scale')
        ax.plot(r, sc, color=cols[H], ls='--', marker='o', markersize=2.6, markevery=3, label=f'$H={H}$, $[0.894,1.125]$')
    ax.set_xlabel(r'effective Eve correlation $\sqrt{c_E}$'); ax.set_ylabel(r'$h_H$ (bit/block)')
    ax.set_xlim(0, 0.85); ax.set_ylim(0, 3.1)
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(0.54, 1.22), fontsize=5.8, handlelength=2.0, columnspacing=.9)
    fig.tight_layout()
    fig.savefig(OUT / 'entropy_eve_v4.pdf'); fig.savefig(OUT / 'entropy_eve_v4.png', dpi=220); plt.close(fig)


def fig_objective():
    """Cap objective at a representative live pilot statistic: BOLT vs BOLT-LCB."""
    import sys; sys.path.insert(0, str(C))
    from bolt_policy import CalibrationLibrary, PolicyParams, local_neighbourhood, lcb_radius, delivered_length_for_cap, choose_cap
    from rayleigh_entropy import analog_eve_certificate
    lib = CalibrationLibrary.load(); p = PolicyParams(); H = 2; n = 7680
    tr = json.loads((C / 'eval_traces_v4.json').read_text())
    z = float(np.median([x['H2']['z'] for x in tr if x['rho_ab'] == 0.9975]))
    h = analog_eve_certificate(H, rho_ae_upper=0.2).h_min_per_block
    caps = np.arange(7000, 14001, 20)
    C2, L2, w2, _ = local_neighbourhood(lib, H, z, p)
    Cp, Lp, wp, _ = local_neighbourhood(lib, H, z, p, one_sided=True)
    g = np.array([np.sum(w2 * (C2 & (L2 <= c))) for c in caps])
    gp = np.array([np.sum(wp * (Cp & (Lp <= c))) for c in caps])
    glow = np.maximum(gp - lcb_radius(wp, p.alpha, 1, 'bracket'), 0)
    ell = np.array([delivered_length_for_cap(n, h, int(c), n * H) for c in caps])
    b = choose_cap(lib, H, z, h, n, 'bolt'); l = choose_cap(lib, H, z, h, n, 'bolt_lcb')
    fig, axs = plt.subplots(1, 2, figsize=(W, 1.42))
    ax = axs[0]
    ax.step(caps / 1e3, g, where='post', color=COL['BOLT'], label=r'$\widehat G$ (BOLT)')
    ax.step(caps / 1e3, gp, where='post', color=COL['BOLT-LCB'], ls='--', label=r'$\widehat G^{+}$ (one-sided)')
    ax.step(caps / 1e3, glow, where='post', color=COL['BOLT-LCB'], label=r'$\underline{G}$ (BOLT-LCB)')
    ax.set_xlabel(r'cap $\Lambda$ (kbit)'); ax.set_ylabel('acceptance'); ax.set_ylim(0, 1.05); ax.set_yticks([0, .25, .5, .75, 1])
    ax.set_title('(a) acceptance vs cap', pad=2)
    ax = axs[1]
    ax.plot(caps / 1e3, ell / 1e3, color='#6b6b66', ls=':', label=r'$\ell_{\rm app}(\Lambda)$')
    ax.plot(caps / 1e3, g * ell / 1e3, color=COL['BOLT'], ls='-.', label=r'$\widehat G\,\ell_{\rm app}$ (BOLT)')
    ax.plot(caps / 1e3, glow * ell / 1e3, color=COL['BOLT-LCB'], ls='-.', label=r'$\underline{G}\,\ell_{\rm app}$ (BOLT-LCB)')
    ax.axvline(b.cap_bits / 1e3, color=COL['BOLT'], lw=.7, ls='-.'); ax.axvline(l.cap_bits / 1e3, color=COL['BOLT-LCB'], lw=.7, ls='-.')
    ax.set_xlabel(r'cap $\Lambda$ (kbit)'); ax.set_ylabel('kbit per session'); ax.set_ylim(0, 1.15 * ell.max() / 1e3)
    i = np.argmax(g * ell); j = np.argmax(glow * ell)
    top = 1.15 * ell.max() / 1e3
    ax.text(b.cap_bits / 1e3 - .12, top * .93, r'$\Lambda^\star_{\rm BOLT}$', fontsize=6.4, color=COL['BOLT'], ha='right', va='top')
    ax.text(l.cap_bits / 1e3 + .12, top * .93, r'$\Lambda^\star_{\rm LCB}$', fontsize=6.4, color=COL['BOLT-LCB'], ha='left', va='top')
    ax.set_title('(b) expected key yield', pad=2)
    h1, l1 = axs[0].get_legend_handles_labels(); h2, l2 = ax.get_legend_handles_labels()
    fig.legend(h1 + h2, l1 + l2, loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.17), fontsize=6.2, handlelength=2.0, columnspacing=.8)
    fig.tight_layout(w_pad=0.8)
    fig.savefig(OUT / 'cap_objective_v4.pdf'); fig.savefig(OUT / 'cap_objective_v4.png', dpi=220); plt.close(fig)
    return {'z': z, 'h': h, 'bolt_cap': b.cap_bits, 'lcb_cap': l.cap_bits, 'bolt_G': b.point_accept, 'lcb_low': l.predicted_accept}


if __name__ == '__main__':
    fig_entropy(); info = fig_objective(); fig_rate(); fig_cv(); fig_cert()
    (C / 'figure_objective_info.json').write_text(json.dumps(info, indent=1))
    print('figures written to', OUT, info)
