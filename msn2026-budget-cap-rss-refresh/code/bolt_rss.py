from __future__ import annotations

from dataclasses import dataclass, asdict
import hashlib
import json
import math
import os
from pathlib import Path
import struct
from typing import Any

import numpy as np
from scipy.signal import fftconvolve
from scipy.stats import beta

from auth import AUTH_TAG_BITS, forgery_bound, parse_auth_material, tag_to_bytes, wc_tag, wc_verify
from bolt_policy import CalibrationLibrary, PolicyParams, choose_cap, select as policy_select
from cascade import cascade_reconcile
from rayleigh_entropy import (
    AnalogEveCertificate, EveGaussianSpec, analog_eve_certificate,
    quantize_power, symbols_to_bits,
)
from security import (
    AUTH_REPLENISH_BITS, DEFAULT_EPS_PA, DEFAULT_MAX_REFRESH_SESSIONS,
    abort_aware_conditional_distance, delivered_application_length,
    extracted_length, lifetime_failure_bound, pa_margin_bits,
    total_failure_bound, variable_transcript_support_bits,
)

H_CANDIDATES=(1,2); H_MAX=max(H_CANDIDATES)


@dataclass(frozen=True)
class SourceModelCertificate:
    """Deployment qualification established outside the live payload session.

    The scales map raw linear receiver powers to the normalized Exp(1) model.
    `failure_probability` is an externally justified bound for the joint event
    that the stated block-i.i.d./Rayleigh/scale certificate is wrong during the
    qualification epoch.  Simulation with a known generator uses zero.
    """
    independent_blocks: bool
    alice_power_scale: float
    bob_power_scale: float
    failure_probability: float = 0.0
    max_refresh_sessions: int = DEFAULT_MAX_REFRESH_SESSIONS
    epoch_id: str = "simulation-known-generator"
    label: str = "external source/scale certificate"
    alice_scale_ratio_lower: float = 1.0
    alice_scale_ratio_upper: float = 1.0

    def validate(self) -> None:
        if not self.independent_blocks:
            raise ValueError("certified path requires independent diversity blocks")
        if not (self.alice_power_scale > 0 and self.bob_power_scale > 0):
            raise ValueError("positive Alice/Bob power scales are required")
        if not (0.0 <= self.failure_probability < 1.0):
            raise ValueError("source-certificate failure probability must lie in [0,1)")
        if int(self.max_refresh_sessions) < 1:
            raise ValueError("max_refresh_sessions must be positive")
        if not (0.0 < float(self.alice_scale_ratio_lower) <= float(self.alice_scale_ratio_upper)):
            raise ValueError("invalid certified Alice scale-ratio interval")





def _cp_upper(k:int,n:int,alpha:float)->float:
    if n<=0 or k>=n:return 1.0
    return float(beta.ppf(1.0-alpha,k+1,n-k))


def _pack_symbols_fixed(symbols:np.ndarray,bits:int)->bytes:
    vals=np.asarray(symbols,dtype=np.uint8)
    raw=((vals[:,None]>>np.arange(bits-1,-1,-1))&1).astype(np.uint8)
    return np.packbits(raw.reshape(-1)).tobytes()


def _nested_symbols_from_max(qmax:np.ndarray,H:int)->np.ndarray:
    return (np.asarray(qmax,int)>>(H_MAX-int(H))).astype(int)


def _fresh_bits(n:int,rng:np.random.Generator|None=None)->np.ndarray:
    if n<=0:return np.zeros(0,np.uint8)
    if rng is not None:return rng.integers(0,2,size=n,dtype=np.uint8)
    raw=np.frombuffer(os.urandom((n+7)//8),dtype=np.uint8)
    return np.unpackbits(raw)[:n].astype(np.uint8)


def _toeplitz_hash(bits:np.ndarray,out_len:int,seed_bits:np.ndarray)->np.ndarray:
    x=np.asarray(bits,np.uint8)
    if out_len<=0:return np.zeros(0,np.uint8)
    need=len(x)+out_len-1; s=np.asarray(seed_bits,np.uint8)
    if len(s)!=need:raise ValueError('Toeplitz seed must contain n+ell-1 bits')
    conv=fftconvolve(s.astype(float),x.astype(float),mode='full')
    vals=np.rint(conv[len(x)-1:len(x)-1+out_len]).astype(np.int64)
    return (vals&1).astype(np.uint8)


def pilot_bmr_table(a_pilot:np.ndarray,b_pilot:np.ndarray,confidence:float=0.99):
    if len(a_pilot)!=len(b_pilot):raise ValueError('pilot arrays must have equal length')
    qa_max=quantize_power(a_pilot,H_MAX); qb_max=quantize_power(b_pilot,H_MAX)
    alpha=(1.0-float(confidence))/len(H_CANDIDATES)
    table={}; records=[b'PILOT-BOLT-RSS-v1',b'A-HMAX='+_pack_symbols_fixed(qa_max,H_MAX)]
    for H in H_CANDIDATES:
        qa=_nested_symbols_from_max(qa_max,H); qb=_nested_symbols_from_max(qb_max,H)
        ba=symbols_to_bits(qa,H); bb=symbols_to_bits(qb,H)
        mism=int(np.count_nonzero(ba!=bb)); nbits=int(len(ba))
        pu=_cp_upper(mism,nbits,alpha)
        table[H]={'mismatches':mism,'n_bits':nbits,'bmr':mism/max(nbits,1),
                  'bmr_upper':float(min(pu,0.5))}
        records.append(struct.pack('>BII',H,mism,nbits))
    return table,b'\n'.join(records)


_DEFAULT_LIB=None


def default_library():
    global _DEFAULT_LIB
    if _DEFAULT_LIB is None:
        _DEFAULT_LIB=CalibrationLibrary.load()
    return _DEFAULT_LIB


def select_H(n_payload:int,table:dict[int,dict[str,float]],eve_spec:EveGaussianSpec,
             fixed_H:int|None=None,objective:str='bolt',
             alice_scale_ratio_bounds:tuple[float,float]=(1.0,1.0),
             lib:CalibrationLibrary|None=None,params:PolicyParams|None=None,
             fixed_caps:dict|None=None):
    """Select (H, Lambda) from pilot data only, through the unified policy module.

    objective in {'bolt','bolt_lcb','q95','q99','fixedcap'}; 'certificate' is a
    legacy alias of 'q99'.  Every policy uses the same calibration
    neighbourhood and is scored with the same external entropy certificate.
    The choice never enters the secrecy theorem.
    """
    lib=lib or default_library(); params=params or PolicyParams()
    obj={'certificate':'q99'}.get(objective,objective)
    certs={H:analog_eve_certificate(H,eve_spec=eve_spec,alice_scale_ratio_bounds=alice_scale_ratio_bounds) for H in H_CANDIDATES}
    h_by_H={H:float(certs[H].h_min_per_block) for H in H_CANDIDATES}
    z_by_H={H:float(table[H]['bmr_upper']) for H in H_CANDIDATES}
    if fixed_H is not None and fixed_H not in H_CANDIDATES: raise ValueError('fixed_H must be 1 or 2')
    choice=policy_select(lib,z_by_H,h_by_H,n_payload,obj,params,fixed_H=fixed_H,fixed_caps=fixed_caps)
    candidates={}
    for H in H_CANDIDATES:
        c=choose_cap(lib,H,z_by_H[H],h_by_H[H],n_payload,obj,params,fixed_caps)
        candidates[H]={'certificate':certs[H],'predicted_leak_cap_bits':int(c.cap_bits),
                       'predicted_accept_probability':float(c.predicted_accept),
                       'point_accept_probability':float(c.point_accept),
                       'predicted_delivered_bits':int(c.delivered_if_accept),
                       'objective':float(c.objective),'policy':c.policy}
    return int(choice.H),candidates


def bolt_rss_session(
    alice_power:np.ndarray,bob_power:np.ndarray,eve_power:np.ndarray|None=None,*,
    source_certificate:SourceModelCertificate|None,
    eve_certificate:EveGaussianSpec|None,
    auth_key_material:bytes,
    active_health_passed:bool,
    session_index:int=0,
    n_pilot:int=256,
    eps_pa:float=DEFAULT_EPS_PA,
    public_seed:int|None=None,
    fixed_H:int|None=None,
    selector_objective:str='bolt',
    materialize_keys:bool=True,
    p_accept_lower:float|None=None,
    lib:CalibrationLibrary|None=None,
    policy_params:PolicyParams|None=None,
    fixed_caps:dict|None=None,
    pilot_confidence:float=0.99,
)->dict[str,Any]:
    a_raw=np.asarray(alice_power,float); b_raw=np.asarray(bob_power,float)
    e=None if eve_power is None else np.asarray(eve_power,float)
    if len(a_raw)!=len(b_raw) or len(a_raw)<n_pilot+128:raise ValueError('equal Alice/Bob arrays with enough blocks required')
    if np.any(~np.isfinite(a_raw)) or np.any(~np.isfinite(b_raw)) or np.any(a_raw<0) or np.any(b_raw<0):raise ValueError('linear finite nonnegative power required')
    if e is not None and len(e)!=len(a_raw):raise ValueError('optional Eve diagnostic length mismatch')
    if len(auth_key_material)!=32:raise ValueError('fresh 256-bit authentication material required')
    if source_certificate is None or eve_certificate is None:
        # Diagnostic-only fail closed.  Still return a machine-readable result.
        return {'version':'bolt-rss-v1','certificate_applicable':False,'claim_scope':'diagnostic only; missing external certificate',
                'delivered_length':0,'certified_bits_per_block':0.0,'refresh_sustainable':False,
                'source_certificate_present':source_certificate is not None,'eve_certificate_present':eve_certificate is not None}
    source_certificate.validate()
    if session_index<0 or session_index>=source_certificate.max_refresh_sessions:
        raise ValueError('session_index outside certified qualification epoch')

    a=a_raw/float(source_certificate.alice_power_scale)
    b=b_raw/float(source_certificate.bob_power_scale)
    reproducible=public_seed is not None
    rng=np.random.default_rng(int(public_seed)) if reproducible else None
    if public_seed is None:public_seed=int.from_bytes(os.urandom(8),'big')

    pilot_table,pilot_transcript=pilot_bmr_table(a[:n_pilot],b[:n_pilot],confidence=pilot_confidence)
    n_payload=len(a)-n_pilot
    H,candidates=select_H(n_payload,pilot_table,eve_certificate,fixed_H=fixed_H,objective=selector_objective,
                          alice_scale_ratio_bounds=(source_certificate.alice_scale_ratio_lower,source_certificate.alice_scale_ratio_upper),
                          lib=lib,params=policy_params,fixed_caps=fixed_caps)
    qa=quantize_power(a[n_pilot:],H); qb=quantize_power(b[n_pilot:],H)
    a_bits=symbols_to_bits(qa,H); b_bits=symbols_to_bits(qb,H)
    raw_bmr=float(np.mean(a_bits!=b_bits))

    p_u=pilot_table[H]['bmr_upper']
    leak_cap=int(candidates[H]['predicted_leak_cap_bits'])
    if leak_cap<=0:
        recon={'out':b_bits.copy(),'leak_bits':0,'rounds':0,'final_bmr':raw_bmr,
               'converged':False,'confirm_collision_prob':0.0,'budget_abort':False,
               'transcript':b'RECON-PREABORT|pilot-upper='+struct.pack('>d',p_u)}
    else:
        cseed=_fresh_bits(len(a_bits)+64-1,rng)
        recon=cascade_reconcile(a_bits,b_bits,est_bmr=p_u,seed=int(public_seed)&0x7fffffff,
                                confirm_bits=64,confirm_seed_bits=cseed,
                                max_source_leak_bits=leak_cap)
    b_corr=np.asarray(recon['out'],np.uint8); n_raw=min(len(a_bits),len(b_corr))
    a_bits=a_bits[:n_raw]; b_corr=b_corr[:n_raw]

    cert=analog_eve_certificate(H,eve_spec=eve_certificate,
                               alice_scale_ratio_bounds=(source_certificate.alice_scale_ratio_lower,source_certificate.alice_scale_ratio_upper))
    # The extractor length used on an accepted branch is fixed by the external
    # certificate, independent pilot and precommitted transcript cap; it does
    # NOT depend on payload reconciliation success.  This is the virtual
    # all-branches extractor length used by the abort-aware proof.
    support_leak=variable_transcript_support_bits(leak_cap)
    candidate_pre_auth_len=extracted_length(n_payload,cert.h_min_per_block,support_leak,n_raw,eps_pa)
    cert_ok=bool(cert.valid and source_certificate.independent_blocks and active_health_passed
                 and recon['converged'] and not recon.get('budget_abort',False))
    pre_auth_len=candidate_pre_auth_len if cert_ok else 0

    tp_need=n_raw+pre_auth_len-1 if pre_auth_len>0 else 0; tp_seed=_fresh_bits(tp_need,rng)
    if materialize_keys and pre_auth_len:
        ka_full=_toeplitz_hash(a_bits,pre_auth_len,tp_seed); kb_full=_toeplitz_hash(b_corr,pre_auth_len,tp_seed)
        agree_full=bool(np.array_equal(ka_full,kb_full))
    else:
        ka_full=kb_full=np.zeros(0,np.uint8); agree_full=None

    # Authenticate everything needed to bind the security context and prevent
    # replay across refresh sessions/qualification epochs.
    src_blob=json.dumps(asdict(source_certificate),sort_keys=True,separators=(',',':')).encode()
    eve_blob=json.dumps(asdict(eve_certificate),sort_keys=True,separators=(',',':')).encode()
    transcript=b'\n'.join([
        b'BOLT-RSS-v1',b'epoch='+source_certificate.epoch_id.encode(),
        b'session-index='+struct.pack('>Q',int(session_index)),
        b'source-cert-sha256='+hashlib.sha256(src_blob).hexdigest().encode(),
        b'eve-cert-sha256='+hashlib.sha256(eve_blob).hexdigest().encode(),
        b'health='+bytes([int(bool(active_health_passed))]),b'H='+bytes([H]),
        b'leak-cap='+struct.pack('>I',leak_cap),pilot_transcript,recon['transcript'],
        b'pa-seed='+np.packbits(tp_seed).tobytes(),b'ell='+struct.pack('>I',pre_auth_len)])
    # Authentication material is one-time.  Do not spend it on a session that
    # already fails closed before authentication.  If a candidate session reaches
    # authentication, the material is consumed whether verification succeeds or
    # fails; a failed verification therefore requires re-bootstrap rather than
    # unsafe reuse of the same Wegman--Carter pad.
    auth_attempted=bool(cert_ok and pre_auth_len>=AUTH_REPLENISH_BITS)
    auth_material_consumed=False
    requires_rebootstrap=False
    if auth_attempted:
        hk,pad=parse_auth_material(auth_key_material)
        tag=wc_tag(hk,pad,transcript)
        auth_material_consumed=True
        auth_verified=wc_verify(hk,pad,transcript,tag)
        eps_auth=forgery_bound(len(transcript))
        requires_rebootstrap=not auth_verified
    else:
        tag=0
        auth_verified=False
        eps_auth=0.0

    sustainable=bool(auth_attempted and auth_verified)
    delivered=delivered_application_length(pre_auth_len) if sustainable else 0
    if materialize_keys and sustainable:
        next_a=ka_full[:AUTH_REPLENISH_BITS]; next_b=kb_full[:AUTH_REPLENISH_BITS]
        app_a=ka_full[AUTH_REPLENISH_BITS:AUTH_REPLENISH_BITS+delivered]
        app_b=kb_full[AUTH_REPLENISH_BITS:AUTH_REPLENISH_BITS+delivered]
        key_agree=bool(np.array_equal(next_a,next_b) and np.array_equal(app_a,app_b))
        if not key_agree:delivered=0; app_a=app_b=np.zeros(0,np.uint8)
    else:
        key_agree=agree_full if materialize_keys else None; app_a=app_b=np.zeros(0,np.uint8)

    eps_cert=min(1.0,float(source_certificate.failure_probability)+float(eve_certificate.failure_probability))
    crypt_eps=(total_failure_bound(eps_pa=eps_pa,eps_confirm=float(recon['confirm_collision_prob']),
                                   eps_auth=float(eps_auth),eps_certificate=0.0)
               if cert_ok else None)
    per_eps=(min(1.0,crypt_eps+eps_cert) if crypt_eps is not None else None)
    life_eps=(lifetime_failure_bound(crypt_eps,source_certificate.max_refresh_sessions,eps_cert)
              if crypt_eps is not None else None)
    cond_dist=abort_aware_conditional_distance(eps_pa,p_accept_lower)

    # Optional Eve trace is diagnostic only and may be multi-column power.
    eve_match=None
    if e is not None and e.ndim==1:
        # Only meaningful after caller has normalized a compatible diagnostic Eve trace.
        try:
            eb=symbols_to_bits(quantize_power(e[n_pilot:],H),H)
            eve_match=float(np.mean(a_bits==eb[:len(a_bits)]))
        except Exception:
            eve_match=None

    comm_bits=8*len(transcript)+(AUTH_TAG_BITS if auth_attempted else 0)
    return {
        'version':'bolt-rss-v1','claim_scope':('model-conditional abort-aware certified output' if cert_ok else 'diagnostic/fail-closed'),
        'source_certificate':asdict(source_certificate),'eve_certificate':asdict(eve_certificate),
        'certificate':cert.to_dict(),'certificate_applicable':cert_ok,'active_health_passed':bool(active_health_passed),
        'session_index':int(session_index),'H':H,'selector_objective':selector_objective,
        'n_total_blocks':len(a),'n_pilot_blocks':n_pilot,'n_payload_blocks':n_payload,'n_raw_bits':n_raw,
        'pilot_table':pilot_table,'candidate_table':{str(h):{k:(v.to_dict() if isinstance(v,AnalogEveCertificate) else v) for k,v in d.items()} for h,d in candidates.items()},
        'raw_bmr':raw_bmr,'final_bmr':float(recon['final_bmr']),'recon_converged':bool(recon['converged']),
        'recon_rounds':int(recon['rounds']),'realized_source_dependent_leak_bits':int(recon['leak_bits']),
        'precommitted_leak_cap_bits':leak_cap,'transcript_support_leak_bits':support_leak,
        'leak_budget_exceeded':bool(recon.get('budget_abort',False)),
        'candidate_pre_auth_length':int(candidate_pre_auth_len),'pre_auth_length':pre_auth_len,
        'auth_replenishment_bits':AUTH_REPLENISH_BITS,'refresh_sustainable':sustainable,
        'delivered_length':int(delivered),'certified_bits_per_block':float(delivered/len(a)),
        'certified_bits_per_payload_block':float(delivered/n_payload),
        'eps_pa':float(eps_pa),'eps_confirm':float(recon['confirm_collision_prob']),'eps_auth':float(eps_auth),
        'eps_external_certificate':float(eps_cert),'per_session_cryptographic_bound':crypt_eps,
        'per_session_failure_bound':per_eps,
        'max_refresh_sessions':int(source_certificate.max_refresh_sessions),'lifetime_failure_bound':life_eps,
        'abort_aware_statement':'p_accept * trace_distance_accepted <= eps_pa',
        'conditional_accepted_distance_bound':cond_dist,
        'auth_attempted':auth_attempted,'auth_material_consumed':auth_material_consumed,
        'requires_rebootstrap':requires_rebootstrap,'auth_verified':bool(auth_verified),
        'auth_tag_hex':(tag_to_bytes(tag).hex() if auth_attempted else None),
        'communication_bits':int(comm_bits),'transcript_bytes':len(transcript),
        'public_bits_per_delivered_bit':(float(comm_bits/delivered) if delivered>0 else None),
        'transcript_sha256':hashlib.sha256(transcript).hexdigest(),
        'public_randomness_mode':('recorded deterministic test vector' if reproducible else 'fresh OS randomness'),
        'key_materialized':bool(materialize_keys),'key_agree':key_agree,
        'eve_gray_bit_match_diagnostic':eve_match,'final_key_alice':app_a,'final_key_bob':app_b,
    }
