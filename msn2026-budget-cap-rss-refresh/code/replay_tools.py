from __future__ import annotations
import numpy as np
from bolt_policy import CalibrationLibrary, PolicyParams, select, outcome


def cell_rate(traces, lib: CalibrationLibrary, rab: float, h_by_H: dict, policy: str = 'bolt',
              N: int = 8192, NP: int = 512, fixed_H=None, params: PolicyParams = PolicyParams()) -> float:
    npay = N - NP
    d = []
    for x in traces:
        if x['rho_ab'] != rab:
            continue
        tb = {1: x['H1'], 2: x['H2']}
        ch = select(lib, {1: tb[1]['z'], 2: tb[2]['z']}, h_by_H, npay, policy, params, fixed_H=fixed_H)
        d.append(outcome(ch, tb))
    return float(np.mean(d) / N)
