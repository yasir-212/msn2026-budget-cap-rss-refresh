from __future__ import annotations
import json, math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import numpy as np
from scipy.stats import beta as beta_dist
from security import AUTH_REPLENISH_BITS,DEFAULT_EPS_PA,pa_margin_bits,variable_transcript_support_bits

_CAL=Path(__file__).with_name('protocol_calibration.json')
K_NEIGHBORS=80

@dataclass(frozen=True)
class CapChoice:
    cap_bits:int
    predicted_accept_probability:float
    predicted_delivered_bits:int
    objective_value:float
    policy:str
    calibration_neighbors:int

@lru_cache(maxsize=1)
def _load():
    d=json.loads(_CAL.read_text()); by={}
    for r in d['records']:by.setdefault(int(r['H']),[]).append(r)
    for H in by:by[H].sort(key=lambda r:float(r['pilot_bmr_upper']))
    return d,by

def _neighbors(H:int,z:float,k:int=K_NEIGHBORS):
    _,by=_load()
    if H not in by:raise ValueError(f'no protocol calibration for H={H}')
    rows=by[H]; vals=np.array([float(r['pilot_bmr_upper']) for r in rows])
    idx=np.argsort(np.abs(vals-float(z)))[:min(k,len(rows))]
    return [rows[int(i)] for i in idx]

def calibration_domain(H:int):
    _,by=_load();v=[float(r['pilot_bmr_upper']) for r in by[H]];return min(v),max(v)

def delivered_length_for_cap(n_payload:int,h:float,cap:int,max_raw:int,eps_pa:float=DEFAULT_EPS_PA)->int:
    ell=math.floor(n_payload*h-variable_transcript_support_bits(int(cap))-pa_margin_bits(eps_pa))
    ell=min(max(int(ell),0),int(max_raw));return max(ell-AUTH_REPLENISH_BITS,0)

def _weighted_quantile(values,weights,q):
    v=np.asarray(values,float);w=np.asarray(weights,float);o=np.argsort(v);v=v[o];w=w[o];c=np.cumsum(w);c/=c[-1];return float(v[np.searchsorted(c,q,side='left')])

def _local(H:int,z:float):
    rows=_neighbors(H,z); zz=np.array([float(r['pilot_bmr_upper']) for r in rows]); d=np.abs(zz-z)
    # Smooth nearest-neighbour weights; bandwidth is the furthest selected neighbour.
    h=max(float(np.max(d)),1e-6); w=np.exp(-0.5*(d/h)**2);w/=w.sum()
    leak=np.array([int(r['leak_bits']) for r in rows]);acc=np.array([bool(r['converged']) for r in rows])
    return rows,w,leak,acc

def quantile_choice(n_payload:int,H:int,p_upper:float,h_min:float,q:float,eps_pa:float=DEFAULT_EPS_PA)->CapChoice:
    """Same-calibration weighted transcript-quantile baseline.

    ``q`` is chosen independently of the evaluation payload.  The resulting
    hard cap is enforced online exactly like BOLT, so the secrecy accounting is
    identical; only the performance policy differs.
    """
    if not (0.0 < float(q) <= 1.0):
        raise ValueError('q must lie in (0,1]')
    lo,hi=calibration_domain(H)
    label=f'q{int(round(100*q))}'
    if p_upper>hi*1.05:
        return CapChoice(0,0.0,0,0.0,label+'-preabort',0)
    rows,w,leak,acc=_local(H,min(max(p_upper,lo),hi))
    cap=int(math.ceil(_weighted_quantile(leak,w,float(q))))
    pacc=float(np.sum(w*(acc & (leak<=cap))))
    ell=delivered_length_for_cap(n_payload,h_min,cap,n_payload*H,eps_pa)
    return CapChoice(cap,pacc,ell,pacc*ell,label,len(rows))

def q95_choice(n_payload:int,H:int,p_upper:float,h_min:float,eps_pa:float=DEFAULT_EPS_PA)->CapChoice:
    return quantile_choice(n_payload,H,p_upper,h_min,.95,eps_pa)

def q99_choice(n_payload:int,H:int,p_upper:float,h_min:float,eps_pa:float=DEFAULT_EPS_PA)->CapChoice:
    return quantile_choice(n_payload,H,p_upper,h_min,.99,eps_pa)


def _weighted_hoeffding_lower(events: np.ndarray, weights: np.ndarray, alpha: float=0.05) -> float:
    """One-sided Hoeffding lower bound for a weighted local Bernoulli mean.

    The events need only be independent and bounded in [0,1]; they need not be
    identically distributed.  For normalized non-negative weights w_i,
    Pr{ sum_i w_i E[X_i] < sum_i w_i X_i - r } <= alpha with
    r = sqrt(0.5*log(1/alpha)*sum_i w_i^2).  This is therefore more appropriate
    than a binomial Clopper-Pearson interval for a heterogeneous k-NN
    calibration neighbourhood.  It bounds the neighbourhood-weighted success
    probability used by the performance policy; it is not part of the secrecy
    theorem.
    """
    x=np.asarray(events,float); w=np.asarray(weights,float)
    if len(x)==0 or len(x)!=len(w):
        return 0.0
    if np.any(w<0) or not np.isfinite(w).all() or w.sum()<=0:
        raise ValueError('weights must be finite and non-negative')
    w=w/w.sum(); phat=float(np.sum(w*x))
    radius=math.sqrt(0.5*math.log(1.0/float(alpha))*float(np.sum(w*w)))
    return max(0.0,phat-radius)

def lcb_cap_choice(n_payload:int,H:int,pilot_bmr_upper:float,h_min:float,eps_pa:float=DEFAULT_EPS_PA,alpha:float=0.05)->CapChoice:
    """Confidence-aware BOLT using a weighted Hoeffding lower bound.

    For each candidate cap, the local k-NN joint events C=1 and L<=cap are
    evaluated with the same smooth distance weights as point BOLT.  The
    one-sided Hoeffding bound is valid for independent, non-identically
    distributed bounded events and therefore does not require a binomial
    exchangeability assumption.  It bounds the local weighted performance
    target only; accepted-branch secrecy still depends solely on the hard cap
    and external entropy certificate.
    """
    lo,hi=calibration_domain(H)
    if pilot_bmr_upper>hi*1.05:
        return CapChoice(0,0.0,0,0.0,'bolt-lcb-preabort',0)
    z=min(max(pilot_bmr_upper,lo),hi)
    rows,w,leak,acc=_local(H,z)
    caps=np.unique(leak)
    best=CapChoice(0,0.0,0,-1.0,'bolt-lcb',len(rows))
    for cap in caps:
        event=(acc & (leak<=cap)).astype(float)
        p_lb=_weighted_hoeffding_lower(event,w,alpha)
        ell=delivered_length_for_cap(n_payload,h_min,int(cap),n_payload*H,eps_pa)
        J=p_lb*ell
        if J>best.objective_value:
            best=CapChoice(int(cap),float(p_lb),int(ell),float(J),'bolt-lcb',len(rows))
    return best

def optimal_cap(n_payload:int,H:int,pilot_bmr_upper:float,h_min:float,eps_pa:float=DEFAULT_EPS_PA)->CapChoice:
    lo,hi=calibration_domain(H)
    if pilot_bmr_upper>hi*1.05:
        return CapChoice(0,0.0,0,0.0,'bolt-preabort',0)
    rows,w,leak,acc=_local(H,min(max(pilot_bmr_upper,lo),hi))
    # Objective changes only when cap crosses an observed transcript length.
    caps=np.unique([int(_weighted_quantile(leak,w,q)) for q in np.linspace(.05,1.0,96)])
    best=CapChoice(0,0.0,0,-1.0,'bolt',len(rows))
    for cap in caps:
        pacc=float(np.sum(w*(acc & (leak<=cap))))
        ell=delivered_length_for_cap(n_payload,h_min,int(cap),n_payload*H,eps_pa)
        J=pacc*ell
        if J>best.objective_value:
            best=CapChoice(int(cap),pacc,int(ell),float(J),'bolt',len(rows))
    return best
