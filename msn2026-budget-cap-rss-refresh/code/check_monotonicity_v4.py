"""Empirical check of Assumption A2 (G_H(Lambda|z) nonincreasing in z) on the library."""
import json, numpy as np
from pathlib import Path
from scipy.stats import kendalltau
d=json.loads(Path(__file__).with_name('protocol_calibration_v4.json').read_text())['records']
out={}
for H in (1,2):
    R=[r for r in d if r['H']==H]; z=np.array([r['pilot_bmr_upper'] for r in R]); C=np.array([r['converged'] for r in R],bool); L=np.array([r['required_cap'] for r in R])
    qs=np.quantile(L[C],[.1,.25,.5,.75,.9]); rows=[]
    for lam in qs:
        X=(C&(L<=lam)).astype(float)
        t,p=kendalltau(z,X)
        edges=np.quantile(z,np.linspace(0,1,7)); g=[float(X[(z>=a)&(z<=b)].mean()) for a,b in zip(edges[:-1],edges[1:])]
        viol=sum(max(0,g[i+1]-g[i]) for i in range(len(g)-1))
        rows.append({'Lambda':float(lam),'kendall_tau_z_vs_accept':float(t),'p':float(p),'sextile_G':g,'total_upward_violation':viol})
    out[str(H)]=rows
    print(H,[(round(r['kendall_tau_z_vs_accept'],3),round(r['total_upward_violation'],3)) for r in rows])
Path(__file__).with_name('results_monotonicity_v4.json').write_text(json.dumps(out,indent=1))
