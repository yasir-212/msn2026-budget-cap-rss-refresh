# BOLT-RSS

**Budget-Optimal Leakage Thresholding for Certified Received-Signal-Strength Key Refresh**

Yasir Nawaz and Hui Zhang — School of Computer Science and Technology, University of Science and Technology of China.
Paper: *Proc. IEEE MSN 2026*.

This repository contains the reference implementation, the calibration library, every experiment script, and the frozen JSON results. Every number, table and figure in the paper is regenerated from the JSON results by `code/make_macros_v4.py`, `code/make_macros_v5.py` and `code/make_figures_v4.py` (output in `outputs/`).

## What BOLT-RSS does
In finite-block RSS key refresh, the public reconciliation transcript depends on the secret. BOLT-RSS commits a **hard cap Λ** on source-dependent discussion *before* any payload-dependent message, checks every message against it (aborting instead of over-disclosing), and charges the full transcript support (Λ+1 bits) before privacy amplification. The **BOLT** policy uses pilot statistics and a protocol-level calibration library to choose the quantizer resolution H and the cap Λ that maximize the expected certified key length. **BOLT-LCB** replaces the point estimate with a lower confidence bound that, for a fixed resolution and under the paper's sampling assumption (Assumption 2), is valid uniformly over all caps. For comparison, the code also implements variable-length charging (realized length plus a logarithmic penalty), which the paper reports as the strongest accounting on the same sessions.

## Repository layout
```
code/
  bolt_rss.py, cascade.py, security.py, auth.py   protocol: pilot, capped Cascade, Wegman-Carter, Toeplitz PA
  rayleigh_entropy.py                             entropy certificate (Theorem 1, Proposition 1)
  bolt_policy.py                                  BOLT, BOLT-LCB, q95/q99, fixed cap, variable-length, oracle
  traces.py, replay_tools.py                      exact session traces and replay
  calibrate_protocol_library_v4.py                840-session calibration library (protocol_calibration_v4.json)
  run_main_grid_v4.py, validate_live_v4.py        main results; 5,880 live end-to-end runs == replay
  run_cv_holdout_v4.py                            20x5-fold cross-validation, score calibration, shift, library size
  run_kgr_metrics_v4.py                           key rate in secret bits per channel use + benchmarks
  run_param_ablation_v4.py, run_protocol_ablation_v4.py   ablations
  run_certificate_acquisition_v4.py               scale / survey / guard-zone certificates, misspecification
  run_combined_certificates_v4.py                 rate with both certificates qualified (Table II)
  channel_emulator.py, run_stress_tests_v4.py     RSSI rounding, gain offset, TDD delay, Rician, correlated blocks
  measured_replay.py                              run the qualification tests and BOLT-RSS on recorded RSS traces
  make_figures_v4.py, make_macros_v4/v5.py        figures and LaTeX macros/tables from the JSON results
  results_*_v4*.json                              frozen results
tests/                                            unit and security tests (18 tests)
scripts/reproduce.sh                              regenerates everything (about 1 h on 2 cores)
outputs/                                          generated figures, tables and macros (created by the scripts)
examples/make_demo_data.py                        synthetic trace for trying measured_replay.py
```

## Quick start
```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v      # 18 tests
bash scripts/reproduce.sh                    # full reproduction
```

## Running on measured RSS data
`code/measured_replay.py` applies the paper's qualification procedure to recorded traces and fails closed if a certificate cannot be established.
```bash
python examples/make_demo_data.py            # or use your own CSV files with the same columns
python code/measured_replay.py examples/demo_trace.csv --survey examples/demo_survey.csv --policy bolt_lcb --rssi-step 1.0
```
* `trace.csv`: `t_s, alice_rssi_dbm, bob_rssi_dbm` (one row per reciprocal probe exchange).
* `survey.csv` (optional): `alice_rssi_dbm, probe_rssi_dbm, position_id` from a trusted probe at candidate eavesdropper positions; alternatively pass `--rho-ae <bound>`.

## Scope of the security claim
The secrecy statement is model-conditional: it holds for the certified block-independent Rayleigh source and the certified Gaussian eavesdropper, for a passive eavesdropper and an authenticated transcript. Calibration affects throughput only, never the accepted-transcript bound.

## License and citation
MIT License (see `LICENSE`). Please cite the paper if you use this code (see `CITATION.cff`).
