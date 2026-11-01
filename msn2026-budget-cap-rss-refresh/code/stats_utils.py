from __future__ import annotations
import numpy as np
from scipy.stats import beta


def bootstrap_mean_ci(values, confidence=0.95, resamples=4000, seed=12345):
    x=np.asarray(values,float).reshape(-1)
    if len(x)==0:return (float('nan'),float('nan'))
    if len(x)==1:return (float(x[0]),float(x[0]))
    rng=np.random.default_rng(seed)
    idx=rng.integers(0,len(x),size=(int(resamples),len(x)))
    means=x[idx].mean(axis=1)
    a=(1-confidence)/2
    return float(np.quantile(means,a)),float(np.quantile(means,1-a))


def clopper_pearson(k,n,confidence=0.95):
    if n<=0:return (float('nan'),float('nan'))
    alpha=1-confidence
    lo=0.0 if k==0 else float(beta.ppf(alpha/2,k,n-k+1))
    hi=1.0 if k==n else float(beta.ppf(1-alpha/2,k+1,n-k))
    return lo,hi
