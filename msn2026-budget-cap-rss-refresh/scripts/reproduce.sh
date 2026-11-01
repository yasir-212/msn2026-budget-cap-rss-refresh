#!/usr/bin/env bash
# Regenerates every number, table and figure reported in the paper (written to outputs/).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/code"
python3 calibrate_protocol_library_v4.py        # 840-session library with exact required cap
python3 run_main_grid_v4.py                     # all policies on identical sessions (writes eval traces)
python3 run_main_grid_v4.py finite              # same grid, finite-scale certificate
python3 validate_live_v4.py                     # 5,880 live end-to-end sessions == replay
python3 run_cv_holdout_v4.py                    # 20x5-fold CV, coverage, shift, library size
python3 run_param_ablation_v4.py                # selection-parameter ablations
python3 run_protocol_ablation_v4.py             # N, N_P, pilot confidence, tag length, H_max
python3 run_kgr_metrics_v4.py                   # bits per channel use + benchmarks
python3 run_certificate_acquisition_v4.py       # certificate acquisition / validation
python3 run_combined_certificates_v4.py         # rate with both certificates qualified (Table II)
python3 run_stress_tests_v4.py                  # realistic-impairment stress tests
python3 check_monotonicity_v4.py                # Assumption 2(ii) check
python3 run_multi_eve_vector_check.py           # colluding-Eve vector check (unchanged)
python3 make_figures_v4.py && python3 make_macros_v4.py && python3 make_macros_v5.py
cd "$ROOT" && python3 -m unittest discover -s tests -v
echo "done"
