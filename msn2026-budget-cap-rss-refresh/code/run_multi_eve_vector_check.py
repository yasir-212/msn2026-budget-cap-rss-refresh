from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
from scipy.stats import ncx2
from rayleigh_entropy import EveGaussianSpec, effective_eve_posterior_snr, exponential_quantizer_edges, continuous_eve_guessing_probability
ROOT=Path(__file__).resolve().parent
H=2; S_TARGET=0.5625; C_TARGET=S_TARGET/(1+S_TARGET); N=200_000

def make_spec(d:int):
    if d==1:
        Sigma=np.eye(1,dtype=complex)
    else:
        rho=.30
        Sigma=np.full((d,d),rho,dtype=complex); np.fill_diagonal(Sigma,1.0)
    u=np.ones(d,dtype=complex)
    base=float(np.real(np.vdot(u,np.linalg.solve(Sigma,u))))
    a=u*math.sqrt(S_TARGET/base)
    return EveGaussianSpec.from_arrays(a,Sigma,label=f'{d}-antenna vector check')

def mc_pguess(spec,seed):
    a,Sigma=spec.arrays(); s=effective_eve_posterior_snr(spec); v=1/(1+s)
    rng=np.random.default_rng(seed)
    ha=(rng.normal(size=N)+1j*rng.normal(size=N))/math.sqrt(2)
    L=np.linalg.cholesky(Sigma)
    z=(rng.normal(size=(N,len(a)))+1j*rng.normal(size=(N,len(a))))/math.sqrt(2)
    E=ha[:,None]*a[None,:]+z@L.conj().T
    Sinv_a=np.linalg.solve(Sigma,a)
    # posterior mean a^H Sigma^-1 E /(1+s)
    m=(E@np.conj(Sinv_a))/(1+s)
    nc=2*np.abs(m)**2/v
    edges=exponential_quantizer_edges(H); probs=[]
    for lo,hi in zip(edges[:-1],edges[1:]):
        flo=np.zeros(N) if lo<=0 else ncx2.cdf(2*lo/v,df=2,nc=nc)
        fhi=np.ones(N) if not np.isfinite(hi) else ncx2.cdf(2*hi/v,df=2,nc=nc)
        probs.append(np.maximum(fhi-flo,0))
    q=np.max(np.vstack(probs),axis=0)
    return float(q.mean()),float(q.std(ddof=1)/math.sqrt(N))

def main():
    rho=math.sqrt(C_TARGET)
    analytic=continuous_eve_guessing_probability(H,rho)[0]
    vals={}
    for d,seed in [(1,7101),(2,7102),(4,7104)]:
        spec=make_spec(d); p,se=mc_pguess(spec,seed)
        vals[str(d)]={'p_guess':p,'mc_standard_error':se,'posterior_snr':effective_eve_posterior_snr(spec)}
    out={'version':'multi-eve-vector-check-v1','H':H,'c_E':C_TARGET,'s_E':S_TARGET,
         'scalar_prediction_p_guess':analytic,'vector_mc':vals,
         'max_abs_discrepancy':max(abs(v['p_guess']-analytic) for v in vals.values())}
    (ROOT/'results_multi_eve_vector_check.json').write_text(json.dumps(out,indent=2))
    print(json.dumps(out,indent=2))
if __name__=='__main__': main()
