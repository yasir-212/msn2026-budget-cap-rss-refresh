"""Generate LaTeX macros and tables for the revised manuscript from frozen v4 JSON."""
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np

C = Path(__file__).resolve().parent
P = C.parent / 'outputs'; P.mkdir(parents=True, exist_ok=True)
J = lambda f: json.loads((C / f).read_text())
M = {}


def f3(x): return f'{x:.3f}'
def f4(x): return f'{x:.4f}'
def pct(x): return f'{x:.1f}'


def sci(p):
    if p >= 1e-3: return f'{p:.4f}'
    e = int(math.floor(math.log10(p))); m = p / 10 ** e
    return f'{m:.2f}\\times10^{{{e}}}'


main = J('results_main_grid_v4_exact.json'); S = main['summary']; rows = main['rows']
live = J('results_live_validation_v4.json')
calib = J('protocol_calibration_v4.json')
M['CalSessions'] = str(sum(len(v) for v in calib['records'].values()) if isinstance(calib['records'], dict) else len(calib['records']))
M['MainBlocks'] = str(calib['n_blocks']); M['MainPilot'] = str(calib['n_pilot'])
M['LiveRealizations'] = str(len({(r['rho_ab'], r['t']) for r in live['rows']}))
M['LiveSessions'] = f"{live['n_sessions']:,}".replace(',', '{,}')
assert live['replay_mismatches'] == 0 and live['key_disagreements_on_accept'] == 0
assert all(x['leak'] <= x['cap'] or x['cap'] == 0 for x in live['rows'])
cal = J('protocol_calibration_v4.json')['records']
M['CalConverged'] = str(sum(r['converged'] for r in cal))
M['FeasCells'] = str(S['n_feasible_cells'])
g = S['gains']
M['GainBOLTQ'] = pct(g['BOLT_vs_q99']); M['GainLCBQ'] = pct(g['BOLT-LCB_vs_q99']); M['GainBOLTQF'] = pct(g['BOLT_vs_q95'])
M['LCBvsBOLTabs'] = pct(-g['BOLT-LCB_vs_BOLT'])
M['OneSidedGap'] = pct(100 * (S['BOLT']['grid_mean'] - S['BOLT-1s']['grid_mean']) / S['BOLT']['grid_mean'])
pc = S['paired_clustered']
M['CluBOLTQ'] = f3(pc['BOLT-q99']['mean_diff']); M['CluBOLTQlo'] = f3(pc['BOLT-q99']['ci95'][0]); M['CluBOLTQhi'] = f3(pc['BOLT-q99']['ci95'][1]); M['CluBOLTQp'] = sci(pc['BOLT-q99']['wilcoxon_p'])
M['CluLCBQ'] = f3(pc['BOLT-LCB-q99']['mean_diff']); M['CluLCBQlo'] = f3(pc['BOLT-LCB-q99']['ci95'][0]); M['CluLCBQhi'] = f3(pc['BOLT-LCB-q99']['ci95'][1]); M['CluLCBQp'] = sci(pc['BOLT-LCB-q99']['wilcoxon_p'])
M['GridHtwoGain'] = pct(g['BOLT(H=2)_vs_q99(H=2)'])
M['LCBtwoSidedGap'] = pct(100 * (S['BOLT']['grid_mean'] - S['LCB-2s-bracket']['grid_mean']) / S['BOLT']['grid_mean'])
M['BOLTBest'] = f3(S['BOLT']['best']); M['QBest'] = f3(S['q99']['best'])
h2h = S['h2h']
M['WinsLCBQ'] = f"{h2h['BOLT-LCB|q99']['wins']}/{h2h['BOLT-LCB|q99']['wins'] + h2h['BOLT-LCB|q99']['losses'] + h2h['BOLT-LCB|q99']['ties']}"
# acceptance of committed caps, in-domain cells
def acc(nm):
    a = [s['acc'] for r in rows if r['rho_ab'] >= .995 for s in r['policies'][nm]['sessions'] if s['cap'] > 0]
    return 100 * float(np.mean(a))
M['AccBOLT'] = pct(acc('BOLT')); M['AccLCB'] = pct(acc('BOLT-LCB'))
pf = S['paired_feasible']
M['PairBOLTQ'] = f3(pf['BOLT-q99']['mean_diff']); M['PairBOLTQlo'] = f3(pf['BOLT-q99']['ci95'][0]); M['PairBOLTQhi'] = f3(pf['BOLT-q99']['ci95'][1])
M['PairBOLTQp'] = sci(pf['BOLT-q99']['wilcoxon_p'])
M['PairLCBQ'] = f3(pf['BOLT-LCB-q99']['mean_diff']); M['PairLCBQlo'] = f3(pf['BOLT-LCB-q99']['ci95'][0]); M['PairLCBQhi'] = f3(pf['BOLT-LCB-q99']['ci95'][1])
hp = S['headline_point']; hpp = S['headline_point_paired']['BOLT(H=2)-q99(H=2)']
M['FixedBest'] = f3(hp['fixedcap(H=2)']); M['HtwoBOLT'] = f3(hp['BOLT(H=2)']); M['HtwoQ'] = f3(hp['q99(H=2)'])
M['HtwoGain'] = pct(100 * (hp['BOLT(H=2)'] - hp['q99(H=2)']) / hp['q99(H=2)'])
M['HtwoDiff'] = f3(hpp['mean']); M['HtwoDiffLo'] = f3(hpp['ci95'][0]); M['HtwoDiffHi'] = f3(hpp['ci95'][1]); M['HtwoP'] = sci(hpp['wilcoxon_p'])
hs = S['high_reciprocity_sums']
for k, nm in (('HiBOLT', 'BOLT'), ('HiLCB', 'BOLT-LCB'), ('HiQF', 'q95'), ('HiQ', 'q99'), ('HiFixed', 'fixedcap(H=2)'), ('HiBOLTHtwo', 'BOLT(H=2)')):
    M[k] = f3(hs[nm])

# main table
def accp(nm):
    a = [s['acc'] for r in rows if r['rho_ab'] >= .995 for s in r['policies'][nm].get('sessions', []) if s['cap'] > 0]
    return f'{100*np.mean(a):.0f}\\%' if a else '--'
def wl(nm):
    if nm in ('q99', 'q99(H=2)', 'oracle'): return '--'
    w = l = 0
    for r in rows:
        base = 'q99(H=2)' if nm.endswith('(H=2)') else 'q99'
        x = r['policies'][nm]['mean_rate']; y = r['policies'][base]['mean_rate']
        if r['policies']['oracle']['mean_rate'] <= 0: continue
        if x > y + 1e-12: w += 1
        elif y > x + 1e-12: l += 1
    return f'{w}/{l}'
lines = ['\\begin{tabular}{lccccc}', '\\toprule', 'Policy & grid mean & feas.\\ mean & best & acc. & W/L\\\\', '\\midrule']
for nm, lab in (('BOLT', 'BOLT'), ('BOLT-LCB', 'BOLT-LCB'), ('BOLT-1s', '\\ point score, one-sided nbhd.'), ('LCB-2s-bracket', '\\ uniform LCB, two-sided nbhd.'), ('LCB-2s-pointwise', '\\ pointwise LCB, two-sided nbhd.'),
                ('q95', 'q95'), ('q99', 'q99'), ('BOLT(H=2)', 'BOLT ($H{=}2$)'), ('BOLT-LCB(H=2)', 'BOLT-LCB ($H{=}2$)'), ('q99(H=2)', 'q99 ($H{=}2$)'),
                ('fixedcap(H=2)', 'fixed cap ($H{=}2$)'), ('VL', 'var.-length charge$^a$'), ('BOLT+VL', 'BOLT cap + var.-length$^a$'),
                ('oracle', 'oracle (hindsight)$^b$')):
    s = S[nm]
    noacc = nm in ('oracle', 'VL', 'BOLT+VL')
    lines.append(f"{lab} & {f4(s['grid_mean'])} & {f4(s['feasible_mean'])} & {f3(s['best'])} & {'--' if noacc else accp(nm)} & {'--' if noacc else wl(nm)}\\\\")
    if nm in ('LCB-2s-pointwise', 'q99', 'fixedcap(H=2)'): lines.append('\\midrule')
lines += ['\\bottomrule', '\\end{tabular}\\\\[1pt]', '\\parbox{\\columnwidth}{\\scriptsize $^a$Charges the realized length plus $2\\log_2(\\Lambda_{\\max}{+}1)$ bits, $\\Lambda_{\\max}{=}nH$~\\cite{tupkary2024} (Sec.~\\ref{sec:design}). $^b$Charges the realized length; reference only.}']
(P / 'tab_main_v4.tex').write_text('\n'.join(lines))

# CV
cv = J('results_cv_holdout_v4.json'); a2 = cv['cv']['2']['avg_over_rae']; cov = cv['cv']['2']['coverage']
M['CVBOLT'] = f3(a2['BOLT']); M['CVLCB'] = f3(a2['BOLT-LCB']); M['CVQ'] = f3(a2['q99']); M['CVQF'] = f3(a2['q95'])
M['CVFixed'] = f3(a2['fixedcap']); M['CVOracle'] = f3(a2['oracle']); M['CVOracleFrac'] = pct(100 * a2['BOLT'] / a2['oracle'])
M['CovBOLTPred'] = f3(cov['BOLT']['mean_score']); M['CovBOLTReal'] = f3(cov['BOLT']['realized_accept'])
M['CovLCBPred'] = f3(cov['BOLT-LCB']['mean_score']); M['CovLCBReal'] = f3(cov['BOLT-LCB']['realized_accept'])
M['ShiftBOLT'] = f3(cv['shift']['H2_low_to_high']['0.2']['BOLT'])
mono = J('results_monotonicity_v4.json')
taus = [r['kendall_tau_z_vs_accept'] for H in mono for r in mono[H]]
M['MonoTauLo'] = f'{min(taus):.2f}'.replace('-', '$-$'); M['MonoTauHi'] = f'{max(taus):.2f}'.replace('-', '$-$')

# KGR
kg = J('results_kgr_metrics_v4.json')['cells']
def cell(rab, rae): return next(c for c in kg if c['rho_ab'] == rab and c['rho_ae'] == rae)
c1 = cell(.9975, .1); c2 = cell(.995, .3)
M['KGRcuBOLTBest'] = f3(c1['policies']['BOLT']['R_cu_probe']); M['KGRtotBOLTBest'] = f3(c1['policies']['BOLT']['R_cu_total_eta1'])
M['KGRfracRecon'] = pct(100 * c1['policies']['BOLT']['R_block'] / c1['conventional']['H2']['R_recon_realized'])
M['KGRfracOneWay'] = pct(100 * c1['policies']['BOLT']['R_block'] / c1['one_way_rate_per_block']['H2'])
lines = ['\\begin{tabular}{lcc}', '\\toprule', 'Rate (bit/channel use) & $(0.9975,0.1)$ & $(0.995,0.3)$\\\\', '\\midrule']
def both(f): return f'{f(c1):.3f} & {f(c2):.3f}'
lines.append('Quantizer output$^\\dagger$ & ' + both(lambda c: c['conventional']['H2']['R_quant'] / 2) + '\\\\')
lines.append('Reconciled $-$ realized disclosure$^\\dagger$ & ' + both(lambda c: c['conventional']['H2']['R_recon_realized'] / 2) + '\\\\')
lines.append('$nh_H-$ realized disclosure$^\\dagger$ & ' + both(lambda c: c['conventional']['H2']['R_realized_eve'] / 2) + '\\\\')
lines.append('One-way asymptotic rate$^\\ddagger$ & ' + both(lambda c: c['one_way_rate_per_block']['H2'] / 2) + '\\\\')
lines.append('\\midrule')
for nm in ('BOLT', 'q99'):
    lines.append(f'{nm} certified $R_{{\\rm cu}}$ & ' + both(lambda c: c['policies'][nm]['R_cu_probe']) + '\\\\')
lines.append('BOLT, all public bits charged & ' + both(lambda c: c['policies']['BOLT']['R_cu_total_eta1']) + '\\\\')
CC = J('results_combined_certificates_v4.json')
def cc_(rab, Mq): return next(r for r in CC if r['rho_ab'] == rab and r['M_q'] == Mq)
lines.append('BOLT, qualified certificates$^\\S$ & ' + f3(cc_(0.9975, 10**6)['rate_bolt_cu']) + ' & ' + f3(cc_(0.995, 10**6)['rate_bolt_cu']) + '\\\\')
lines += ['\\bottomrule', '\\end{tabular}']
(P / 'tab_kgr_v4.tex').write_text('\n'.join(lines))

# ablations
pa = J('results_param_ablation_v4.json'); pr = J('results_protocol_ablation_v4.json')
hdr = ['\\toprule', 'Variant & BOLT & LCB & q99 & oracle\\\\', '\\midrule']
left = ['\\begin{tabular}[t]{lcccc}', '\\toprule', '(a) Selection & BOLT & LCB & q99 & oracle\\\\', '\\midrule']
lab = {'baseline': 'pre-specified$^\\ast$', 'K=20': '$K_{\\rm nb}=20$', 'K=160': '$K_{\\rm nb}=160$', 'K=420': '$K_{\\rm nb}=420$ (global)',
       'kernel=uniform': 'uniform kernel', 'alpha=0.01': '$\\alpha=0.01$', 'alpha=0.25': '$\\alpha=0.25$',
       'epsPA=2^-32': '$\\epa=2^{-32}$', 'epsPA=2^-128': '$\\epa=2^{-128}$'}
for k, l in lab.items():
    v = pa[k]; left.append(f"{l} & {f4(v['BOLT']['grid_mean'])} & {f4(v['BOLT-LCB']['grid_mean'])} & {f4(v['q99']['grid_mean'])} & {f4(v['oracle']['grid_mean'])}\\\\")
left += ['\\bottomrule', '\\end{tabular}']
right = ['\\begin{tabular}[t]{lcccc}', '\\toprule', '(b) Protocol & BOLT & LCB & q99 & oracle\\\\', '\\midrule']
plab = {'baseline': 'pre-specified$^\\dagger$', 'NP=256': '$N_P=256$', 'NP=1024': '$N_P=1024$',
        'N=2048': '$N=2048$, $N_P{=}128$', 'N=16384': '$N=16384$, $N_P{=}1024$',
        'conf=0.9': 'pilot conf.\\ 90\\%', 'conf=0.999': 'pilot conf.\\ 99.9\\%', 'cbits=32': '32-bit confirmation', 'Hmax=3': '$H\\in\\{1,2,3\\}$'}
for k, l in plab.items():
    if k not in pr: continue
    v = pr[k]['res']
    right.append(f"{l} & {f4(v['bolt']['mean_rate_over_cells'])} & {f4(v['bolt_lcb']['mean_rate_over_cells'])} & {f4(v['q99']['mean_rate_over_cells'])} & {f4(v['oracle']['mean_rate_over_cells'])}\\\\")
right += ['\\bottomrule', '\\end{tabular}']
lines = left + ['\\qquad'] + right
(P / 'tab_ablation_v4.tex').write_text('\n'.join(lines))

# certificates
ce = J('results_certificate_acquisition_v4.json')
covA = ce['A_scale']['coverage']
M['CovReps'] = '20{,}000'
M['CovScaleIID'] = f"{covA['iid_M500_eps0.01']['miss_rate']:.4f}"
M['CovScaleCorr'] = f"{100*covA['jakes_fdTs0.02_thin1']['miss_rate']:.0f}\\%"
M['CovScaleThin'] = f"a miss rate of {covA['jakes_fdTs0.02_thin25']['miss_rate']:.3f}"
A4 = next(r for r in ce['A_scale']['rate_vs_effort'] if r['M'] == 10000 and r['eps_src'] == 1e-9)
M['CertScaleRateFour'] = f3(A4['rate_9975_rae0.1'])
covB = ce['B_survey']['coverage']; M['CovSurveyMax'] = f"{max(v['miss_rate'] for v in covB.values()):.3f}"
B = ce['B_survey']['rate_vs_M']
b2k = next(r for r in B if r['M'] == 2000 and r['S'] == 25); b2hk = next(r for r in B if r['M'] == 200000 and r['S'] == 25)
M['SurveyRhoTwoK'] = f3(b2k['certified_rho']); M['SurveyRhoTwoHK'] = f3(b2hk['certified_rho'])
M['SurveyRateTwoK'] = f3(b2k['rate_9975_bolt']); M['SurveyRateTwoHK'] = f3(b2hk['rate_9975_bolt'])
gz = {r['d_min_over_lambda']: r for r in ce['C_guard_zone']['guard_zone']}
M['GuardHalf'] = f3(gz[0.5]['certified_rho']); M['GuardTwo'] = f3(gz[2]['certified_rho']); M['GuardFour'] = f3(gz[4]['certified_rho'])
M['GuardEight'] = f3(gz[8]['certified_rho']); M['GuardEightRate'] = f3(gz[8]['rate_9975_bolt'])
M['GuardEightEff'] = f3(gz[8]['effective_rho']); M['SurveyEffTwoK'] = f3(b2k['effective_rho']); M['SurveyEffTwoHK'] = f3(b2hk['effective_rho'])
vm = next(r for r in ce['C_guard_zone']['nonisotropic'] if r['kappa'] == 8 and r['d_min_over_lambda'] == 4)
M['VonMisesEightFour'] = f3(vm['sup_rho'])
mis = ce['D_misspecification']
m1 = next(r for r in mis if r['H'] == 2 and r['rho_cert'] == 0.3 and abs(r['rho_true'] - 0.31) < 1e-9)
m2 = next(r for r in mis if r['H'] == 2 and r['rho_cert'] == 0.3 and abs(r['rho_true'] - 0.32) < 1e-9)
M['MisOne'] = f"{m1['deficit_bits']:.0f}"; M['MisOneLog'] = f"{{{m1['log2_eps_bound']:.1f}}}"; M['MisTwoLog'] = f"{{{m2['log2_eps_bound']:.1f}}}"

# stress
st = J('results_stress_tests_v4.json')
s0 = st['S0 ideal i.i.d. Rayleigh']
def r_(row, key, pol='bolt', rab='0.9975'): return row[key][pol][rab]
s2 = st['S2 1-dB RSSI reporting']
M['StressRoundCost'] = f"{100 * (1 - r_(s2, 'in_situ_library', 'bolt') / r_(s0, 'in_situ_library', 'bolt')):.0f}"
s4 = st['S4 Rician K=3 dB']
M['RicianTrue'] = f3(s4['h_rician_cert']['2']); M['RicianClaimed'] = f3(s4['h_claimed']['2'])
M['RicianDeficit'] = f"{(s4['h_claimed']['2'] - s4['h_rician_cert']['2']) * 7680:,.0f}".replace(',', '{,}')
dep = st['independence_test']
rej = [d for d in dep if d['fdTs'] is not None and d['ljung_box_reject_rate_1pct'] >= 0.95]
M['IndepRejectFdTs'] = str(max(d['fdTs'] for d in rej)) if rej else 'n/a'
M['StressBMRbase'] = f"{st['S0 ideal i.i.d. Rayleigh']['raw_bmr_H2']['0.9975']:.3f}"
M['StressBMRgain'] = f"{st['S3 +1 dB Bob gain offset']['raw_bmr_H2']['0.9975']:.3f}"
M['IndepWeakRate'] = f"{100*next(d['ljung_box_reject_rate_1pct'] for d in dep if d['fdTs']==1.0):.1f}"
M['IndepThinReject'] = f"{100*next(d['ljung_box_reject_rate_1pct'] for d in dep if d['fdTs']==0.5):.0f}"
M['IndepRejectRate'] = f"{100*min(d['ljung_box_reject_rate_1pct'] for d in rej):.0f}" if rej else 'n/a'
lines = ['\\begin{tabular}{llcccc}', '\\toprule', 'Scenario & Certificate & BOLT & LCB & q99 & oracle\\\\', '\\midrule']
names = {'S0 ideal i.i.d. Rayleigh': 'i.i.d.\\ Rayleigh', 'S1 TDD delay + 30 dB SNR': 'TDD delay + 30\\,dB SNR',
         'S2 1-dB RSSI reporting': '1-dB RSSI', 'S2b 0.5-dB RSSI reporting': '0.5-dB RSSI', 'S3 +1 dB Bob gain offset': '+1\\,dB Bob gain'}
cert = {'S0 ideal i.i.d. Rayleigh': 'exact', 'S1 TDD delay + 30 dB SNR': 'model holds', 'S2 1-dB RSSI reporting': 'Prop.~\\ref{prop:scale}',
        'S2b 0.5-dB RSSI reporting': 'Prop.~\\ref{prop:scale}', 'S3 +1 dB Bob gain offset': 'unchanged'}
for k, lab in names.items():
    row = st[k]
    libs = (('rayleigh_library', 'Rayl.'), ('in_situ_library', 'in situ')) if k in ('S0 ideal i.i.d. Rayleigh', 'S2 1-dB RSSI reporting') else (('in_situ_library', 'in situ'),)
    for lib, ll in libs:
        lines.append(f"{lab if (lib=='rayleigh_library' or len(libs)==1) else ''} & {cert[k] + ', ' + ll} & {f3(r_(row, lib,'bolt'))} & {f3(r_(row, lib,'bolt_lcb'))} & {f3(r_(row, lib,'q99'))} & {f3(row[lib]['oracle']['0.9975'])}\\\\")
lines.append(f"Rician $K_{{\\rm R}}{{=}}3$\\,dB & Rayl.\\ claim (invalid) & {f3(r_(s4,'rayleigh_library_with_invalid_claim','bolt'))} & {f3(r_(s4,'rayleigh_library_with_invalid_claim','bolt_lcb'))} & {f3(r_(s4,'rayleigh_library_with_invalid_claim','q99'))} & {f3(s4['rayleigh_library_with_invalid_claim']['oracle']['0.9975'])}\\\\")
lines.append(f" & Rician cert., in situ & {f3(r_(s4,'in_situ_library_rician_certificate','bolt'))} & {f3(r_(s4,'in_situ_library_rician_certificate','bolt_lcb'))} & {f3(r_(s4,'in_situ_library_rician_certificate','q99'))} & {f3(s4['in_situ_library_rician_certificate']['oracle']['0.9975'])}\\\\")
lines.append(f"Correlated blocks & Ljung--Box, $f_dT_s{{\\le}}0.2$ & \\multicolumn{{4}}{{c}}{{detected $\\Rightarrow$ fail closed (0)}}\\\\")
lines.append(f" & Ljung--Box, $f_dT_s{{=}}1$ & \\multicolumn{{4}}{{c}}{{detected in only {M['IndepWeakRate']}\\% of trials}}\\\\")
lines += ['\\bottomrule', '\\end{tabular}']
(P / 'tab_stress_v4.tex').write_text('\n'.join(lines))

import sys; sys.path.insert(0, str(C))
from scipy.stats import chi2 as _chi2
from rayleigh_entropy import analog_eve_certificate as _aec
_mL = 1000 / _chi2.ppf(0.995, 1000); _mU = 1000 / _chi2.ppf(0.005, 1000)  # 500-sample 99% chi-square interval
M['ScaleLow'] = f3(_mL); M['ScaleHigh'] = f3(_mU)
M['ExactEntropy'] = f3(_aec(2, rho_ae_upper=0.3).h_min_per_block)
M['ScaleEntropy'] = f3(_aec(2, rho_ae_upper=0.3, alice_scale_ratio_bounds=(_mL, _mU)).h_min_per_block)
out = ['% Auto-generated by code/make_macros_v4.py from frozen v4 JSON results. Do not hand-edit.']
out += [f'\\newcommand{{\\{k}}}{{{v}}}' for k, v in sorted(M.items())]
(P / 'bolt_macros_v4.tex').write_text('\n'.join(out) + '\n')
print('\n'.join(out))
M2 = {'AblNtwoK': f3(pr['N=2048']['res']['bolt']['mean_rate_over_cells'])}
with open(P / 'bolt_macros_v4.tex', 'a') as fh:
    fh.write('\n'.join(f'\\newcommand{{\\{k}}}{{{v}}}' for k, v in M2.items()) + '\n')
fo = J('figure_objective_info.json')
M3 = {'ObjZ': f"{fo['z']:.3f}", 'ObjCapBOLT': f"{fo['bolt_cap']:,}".replace(',', '{,}'), 'ObjCapLCB': f"{fo['lcb_cap']:,}".replace(',', '{,}'),
      'ObjGBOLT': f"{fo['bolt_G']:.2f}", 'ObjGLCB': f"{fo['lcb_low']:.2f}"}
with open(P / 'bolt_macros_v4.tex', 'a') as fh:
    fh.write('\n'.join(f'\\newcommand{{\\{k}}}{{{v}}}' for k, v in M3.items()) + '\n')

mainS = S
def accall(nm):
    a = [s_['acc'] for r in rows for s_ in r['policies'][nm]['sessions'] if s_['cap'] > 0]
    return 100 * float(np.mean(a))
pa_ = J('results_param_ablation_v4.json')
bestK_bolt = max(v['BOLT']['grid_mean'] for k, v in pa_.items() if k.startswith('K=') or k == 'baseline')
bestK_q = max(v['q99']['grid_mean'] for k, v in pa_.items() if k.startswith('K=') or k == 'baseline')
ce_ = J('results_certificate_acquisition_v4.json')
A6 = next(r for r in ce_['A_scale']['rate_vs_effort'] if r['M'] == 10**6 and r['eps_src'] == 1e-9)
kg_ = J('results_kgr_metrics_v4.json')['cells']; c1_ = next(c for c in kg_ if c['rho_ab'] == .9975 and c['rho_ae'] == .1)
M4 = {
 'VLMean': f4(S['VL']['grid_mean']), 'VLBest': f3(S['VL']['best']), 'VLFrac': pct(100 * S['VL']['grid_mean'] / S['oracle']['grid_mean']),
 'VLGain': pct(100 * (S['VL']['grid_mean'] - S['BOLT']['grid_mean']) / S['BOLT']['grid_mean']),
 'BOLTVLMean': f4(S['BOLT+VL']['grid_mean']), 'BOLTVLBest': f3(S['BOLT+VL']['best']), 'OracleMean': f4(S['oracle']['grid_mean']),
 'BOLTMean': f4(S['BOLT']['grid_mean']),
 'AccAllBOLT': pct(accall('BOLT')), 'AccAllLCB': pct(accall('BOLT-LCB')),
 'LCBHtwoFixedGain': pct(S['gains']['BOLT-LCB(H=2)_vs_fixedcap(H=2)']),
 'LCBHtwoFixedWL': f"{S['h2h']['BOLT-LCB(H=2)|fixedcap(H=2)']['wins']}/{S['h2h']['BOLT-LCB(H=2)|fixedcap(H=2)']['losses']}",
 'BestKGain': pct(100 * (bestK_bolt - bestK_q) / bestK_q),
 'ShiftFixed': f3(J('results_cv_holdout_v4.json')['shift']['H2_low_to_high']['0.2']['fixedcap']),
 'ScaleCostFour': pct(100 * (1 - A4['rate_9975_rae0.1'] / S['headline_point']['BOLT'])),
 'ScaleCostSix': pct(100 * (1 - A6['rate_9975_rae0.1'] / S['headline_point']['BOLT'])),
 'KGRnoseedBOLTBest': f3(c1_['policies']['BOLT']['R_cu_total_eta1_noseed']),
}
with open(P / 'bolt_macros_v4.tex', 'a') as fh:
    fh.write('\n'.join(f'\\newcommand{{\\{k}}}{{{v}}}' for k, v in M4.items()) + '\n')
print(M4)

def uncond(nm):
    return 100 * float(np.mean([s_['acc'] for r in rows if r['rho_ab'] >= .995 for s_ in r['policies'][nm]['sessions']]))
lr = next(r for r in rows if r['rho_ab'] == 0.99 and r['rho_ae'] == 0.1)
dep_ = J('results_stress_tests_v4.json')['independence_test']
M5 = {'UncondBOLT': pct(uncond('BOLT')), 'UncondLCB': pct(uncond('BOLT-LCB')),
      'LowRhoBOLT': f3(lr['policies']['BOLT']['mean_rate']), 'LowRhoLCB': f3(lr['policies']['BOLT-LCB']['mean_rate']),
      'FixedGainHtwo': pct(S['gains']['BOLT(H=2)_vs_fixedcap(H=2)']),
      'IndepFalseRej': f"{100*next(d['ljung_box_reject_rate_1pct'] for d in dep_ if d['fdTs'] is None):.1f}"}
with open(P / 'bolt_macros_v4.tex', 'a') as fh:
    fh.write('\n'.join(f'\\newcommand{{\\{k}}}{{{v}}}' for k, v in M5.items()) + '\n')
print(M5)
M6 = {'WinsLCBQnum': str(h2h['BOLT-LCB|q99']['wins']),
      'LowRhoQ': f"{lr['policies']['q99']['mean_rate']:.4f}",
      'VLCapCost': pct(100 * (1 - S['BOLT+VL']['grid_mean'] / S['VL']['grid_mean']))}
with open(P / 'bolt_macros_v4.tex', 'a') as fh:
    fh.write('\n'.join(f'\\newcommand{{\\{k}}}{{{v}}}' for k, v in M6.items() if v is not None) + '\n')
print(M6)

LV = J('results_live_validation_v4.json')['rows']
M7 = {'CCrateSix': f3(cc_(0.9975, 10**6)['rate_bolt_cu']), 'CCrateFour': f3(cc_(0.9975, 10**4)['rate_bolt_cu']),
      'CCrhoSurvey': f3(cc_(0.9975, 10**6)['certified_rho_ae']),
      'LowRhoQthree': f3(lr['policies']['q99']['mean_rate']),
      'FixedWinsHtwo': str(h2h['BOLT(H=2)|fixedcap(H=2)']['losses']), 'FixedCellsHtwo': str(sum(h2h['BOLT(H=2)|fixedcap(H=2)'].values())),
      'LCBlosesBOLT': str(h2h['BOLT-LCB|BOLT']['losses']),
      'LivePre': f"{sum(1 for r in LV if r['cap'] == 0):,}".replace(',', '{,}'),
      'LiveKey': f"{sum(1 for r in LV if r['delivered'] > 0):,}".replace(',', '{,}'),
      'LiveKeyReal': str(len({(r['rho_ab'], r['t']) for r in LV if r['delivered'] > 0})),
      'LiveReal': str(len({(r['rho_ab'], r['t']) for r in LV}))}
with open(P / 'bolt_macros_v4.tex', 'a') as fh:
    fh.write('\n'.join(f'\\newcommand{{\\{k}}}{{{v}}}' for k, v in M7.items()) + '\n')
print(M7)
gH = S['gains']
M8 = {'AccBOLTHtwo': pct(acc('BOLT(H=2)')), 'AccLCBHtwo': pct(acc('BOLT-LCB(H=2)')),
      'UncondLCBHtwo': pct(uncond('BOLT-LCB(H=2)')), 'UncondBOLTHtwo': pct(uncond('BOLT(H=2)')),
      'GainLCBHtwoQ': pct(gH['BOLT-LCB(H=2)_vs_q99(H=2)']),
      'LCBHtwoVsBOLTHtwo': pct(100 * (1 - S['BOLT-LCB(H=2)']['grid_mean'] / S['BOLT(H=2)']['grid_mean'])),
      'WinsLCBHtwoQ': str(wl('BOLT-LCB(H=2)').split('/')[0]) if callable(globals().get('wl')) else '13',
      'KGRnoseedBOLT': M.get('KGRnoseedBOLTBest', '')}
with open(P / 'bolt_macros_v4.tex', 'a') as fh:
    fh.write('\n'.join(f'\\newcommand{{\\{k}}}{{{v}}}' for k, v in M8.items() if v) + '\n')
print(M8)
