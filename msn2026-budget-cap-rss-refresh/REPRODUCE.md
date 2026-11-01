# Reproducing the paper

Requirements: Python 3.10+, `numpy`, `scipy`, `matplotlib`; `pdflatex` to rebuild the PDF.

```bash
bash scripts/reproduce.sh
```

The script (about 1 h on 2 CPU cores):
1. builds the 840-session calibration library with the exact required cap;
2. runs all policies on identical sessions of the 42-condition grid (exact replay);
3. reruns every grid session end to end (5,880 live protocol runs) and checks equality with the replay;
4. runs cross-validation, ablations, key-rate metrics, certificate-acquisition and stress-test experiments;
5. regenerates the figures, tables and macros into `outputs/` and runs the tests.

All random seeds are fixed; calibration seeds (77e6/88e6 ranges) are disjoint from evaluation seeds (55e6/66e6 ranges).
