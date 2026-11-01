from __future__ import annotations
import json, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from auth import fresh_auth_material
from rayleigh_entropy import independent_eve_spec, simulate_correlated_rayleigh_blocks
from bolt_rss import SourceModelCertificate, bolt_rss_session
from bolt_policy import CalibrationLibrary, PolicyParams, global_fixed_caps
from run_main_grid_v4 import RABS, RAES, TRIALS, N, NP, NPAY, seed_of, h_table

ROOT = Path(__file__).resolve().parent
LIVE = [('BOLT', 'bolt', None), ('BOLT-LCB', 'bolt_lcb', None), ('q95', 'q95', None), ('q99', 'q99', None),
        ('BOLT(H=2)', 'bolt', 2), ('BOLT-LCB(H=2)', 'bolt_lcb', 2), ('q99(H=2)', 'q99', 2)]


def job(args):
    rab, t = args
    s = seed_of(rab, t)
    a, b, _ = simulate_correlated_rayleigh_blocks(N, rab, 0.1, s)
    src = SourceModelCertificate(True, 1, 1, 0, max_refresh_sessions=2**16, epoch_id='known-Rayleigh-generator')
    out = []
    for rae in RAES:
        eve = independent_eve_spec(rae, 1, 0.0)
        for name, pol, fH in LIVE:
            # pass 1: choose H (cheap); pass 2 runs the session with that H and the same public seed as replay
            from bolt_rss import select_H, pilot_bmr_table
            pt, _ = pilot_bmr_table(a[:NP], b[:NP])
            H, _ = select_H(NPAY, pt, eve, fixed_H=fH, objective=pol)
            auth = fresh_auth_material(np.random.default_rng(s + 900000 + H))
            r = bolt_rss_session(a, b, None, source_certificate=src, eve_certificate=eve, auth_key_material=auth,
                                 active_health_passed=True, n_pilot=NP, public_seed=s + 1000 * H, fixed_H=H,
                                 selector_objective=pol, materialize_keys=True, session_index=t)
            out.append({'rho_ab': rab, 'rho_ae': rae, 't': t, 'policy': name, 'H': H,
                        'cap': r['precommitted_leak_cap_bits'], 'delivered': r['delivered_length'],
                        'leak': r['realized_source_dependent_leak_bits'], 'comm_bits': r['communication_bits'],
                        'transcript_bytes': r['transcript_bytes'], 'budget_abort': r['leak_budget_exceeded'],
                        'recon': r['recon_converged'], 'key_agree': r['key_agree'],
                        'n_raw_bits': r['n_raw_bits'], 'pre_auth': r['pre_auth_length']})
    return out


if __name__ == '__main__':
    jobs = [(rab, t) for rab in RABS for t in range(TRIALS)]
    rows = []
    with ProcessPoolExecutor(max_workers=2) as ex:
        for i, r in enumerate(ex.map(job, jobs, chunksize=1), 1):
            rows.extend(r)
            if i % 10 == 0: print('done', i, '/', len(jobs), flush=True)
    rep = json.loads((ROOT / 'results_main_grid_v4_exact.json').read_text())
    idx = {(r['rho_ab'], r['rho_ae']): r for r in rep['rows']}
    mism = 0
    for x in rows:
        s = idx[(x['rho_ab'], x['rho_ae'])]['policies'][x['policy']]['sessions'][x['t']]
        x['replay_delivered'] = s['delivered']; x['replay_cap'] = s['cap']
        if s['delivered'] != x['delivered'] or s['cap'] != x['cap']:
            mism += 1
    out = {'n_sessions': len(rows), 'replay_mismatches': mism,
           'key_disagreements_on_accept': sum(1 for x in rows if x['delivered'] > 0 and not x['key_agree']),
           'rows': rows}
    (ROOT / 'results_live_validation_v4.json').write_text(json.dumps(out))
    print('sessions', len(rows), 'replay mismatches', mism, 'key disagreements', out['key_disagreements_on_accept'])
