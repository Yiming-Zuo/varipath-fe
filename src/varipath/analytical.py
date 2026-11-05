from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from numpy.polynomial.hermite import hermgauss
from numpy.polynomial.legendre import leggauss
from scipy.special import logsumexp
import torch

from .benchmarks import DoubleWellSystem
from .interventions import ConjugateRoute, IdentityLoop, ProtectedTranslation
from .protection import RadialProtection


class _IdentityBase:
    def __init__(self, n_steps):
        self.n_steps = n_steps

    def single_step(self, z, k):
        return z

    single_inverse_step = single_step


def _a(t):
    return t*t*(1-t)*(1-t)


def _route(c, sigma=1., epsilon=0., n_steps=32, conjugated=False):
    guard = RadialProtection([0., 0.], 32., 34.)
    times = torch.arange(n_steps+1, dtype=torch.float64)/n_steps
    loops = [IdentityLoop()]
    loops.extend(ProtectedTranslation(guard, torch.tensor([c*_a(float(t)), 0.], dtype=torch.float64)) for t in times[1:-1])
    loops.append(IdentityLoop())
    eta = times[:-1]*(1-times[:-1])/n_steps
    return ConjugateRoute(_IdentityBase(n_steps), loops, guard, torch.zeros(2, dtype=torch.float64), sigma, .5, eta, epsilon=epsilon, conjugated_kicks=conjugated)


def _shifts(route):
    return np.asarray([np.zeros(2) if isinstance(loop, IdentityLoop) else loop.displacement.detach().cpu().numpy() for loop in route.loops])


def _plateau_map(route, points, *, inverse=False):

    x = np.array(points, dtype=float, copy=True)
    shifts, sigma = _shifts(route), route.sigma
    eta = route.eta.detach().cpu().numpy()
    logdet = np.zeros(len(x))
    if inverse:

        for k in reversed(range(route.n_steps)):
            shifted = x-shifts[k+1]
            base = shifted if route.conjugated_kicks else shifted+shifts[k]
            old = base.copy()
            for _ in range(100):
                diff=old-route.center.detach().cpu().numpy()
                psi=np.exp(-np.square(diff).sum(1)/(2*sigma*sigma))
                increment=-route.epsilon*eta[k]*route.amplitude*psi[:,None]*diff/(sigma*sigma)
                new=base-increment
                if np.max(np.abs(new-old))<1e-13:
                    old=new; break
                old=new
            else:
                raise RuntimeError("E0平台kick逆解未收敛")
            x=old+shifts[k] if route.conjugated_kicks else old
        return x, -_plateau_map(route,x)[1]
    for k in range(route.n_steps):
        if route.conjugated_kicks:
            x=x-shifts[k]
        diff=x-route.center.detach().cpu().numpy()
        psi=np.exp(-np.square(diff).sum(1)/(2*sigma*sigma))
        strength=route.epsilon*eta[k]*route.amplitude
        increment=-strength*psi[:,None]*diff/(sigma*sigma)
        kicked=x+increment
        tangential=-strength*psi/sigma**2
        radial=strength*psi*(np.square(diff).sum(1)/sigma**4-1/sigma**2)
        if not ((1+tangential>0)&(1+radial>0)).all():
            raise RuntimeError("E0实际kick行列式非正")
        logdet+=np.log1p(tangential)+np.log1p(radial)

        before=kicked if route.conjugated_kicks else kicked-shifts[k]
        after=before+shifts[k+1]
        if max(np.linalg.norm(x,axis=1).max(),np.linalg.norm(kicked,axis=1).max(),np.linalg.norm(before,axis=1).max(),np.linalg.norm(after,axis=1).max())>=route.protection.inner_radius:
            raise ValueError("E0专门化算子遇到非平台构型，必须使用一般保护图")
        x=after
    return x,logdet


def _tilt(x):
    return DoubleWellSystem.tilt(torch.as_tensor(x,dtype=torch.float64)).numpy()


def _quadrature(order=96):
    a,w=leggauss(order)
    cuts=[-3.,-.5,0.,.5,3.]
    x=np.concatenate([.5*(hi-lo)*a+.5*(hi+lo) for lo,hi in zip(cuts[:-1],cuts[1:])])
    wx=np.concatenate([.5*(hi-lo)*w for lo,hi in zip(cuts[:-1],cuts[1:])])
    gx,gw=hermgauss(order)
    y=np.sqrt(2)*gx
    wy=gw/np.sqrt(math.pi)
    zx=float(np.sum(wx*np.exp(-(x*x-1)**2)))
    px=wx*np.exp(-(x*x-1)**2)/zx
    points=np.stack(np.meshgrid(x,y,indexing="ij"),-1).reshape(-1,2)
    probability=(px[:,None]*wy[None,:]).ravel()
    return points,probability,zx


def _energy(points, tilt=.05, k=1.):
    return (points[:,0]**2-1)**2+.5*k*points[:,1]**2-tilt*_tilt(points[:,0])


def _response(points, times, weights, c, sigma=1., amplitude=.5):
    g=np.zeros_like(points)
    div=np.zeros(len(points))
    for t,w in zip(times,weights):
        d=points+np.array([c*_a(t),0.])
        r2=np.square(d).sum(1)
        psi=np.exp(-r2/(2*sigma*sigma))
        g-=amplitude*w*psi[:,None]*d/sigma**2
        div+=amplitude*w*psi*(r2/sigma**4-2/sigma**2)
    gradient=np.stack([4*points[:,0]*(points[:,0]**2-1),points[:,1]],1)
    score=(gradient*g).sum(1)-div
    return g,score


def _derivatives(c,sigma=1.,order=96,n_steps=None,target_tilt=.05,target_k=1.):
    points,p,zx=_quadrature(order)
    if n_steps is None:
        t,w=leggauss(48)
        times=.5*(t+1)
        dose=.5*w*times*(1-times)
    else:
        times=np.arange(n_steps)/n_steps
        dose=times*(1-times)/n_steps
    g,score=_response(points,times,dose,c,sigma)
    logw=_energy(points,0.,1.)-_energy(points,target_tilt,target_k)
    logk=logsumexp(np.log(p)+logw)
    r2=np.exp(2*(logw-logk))
    mass=float(np.sum(p[points[:,0]>0]*score[points[:,0]>0]))
    risk=float(-np.sum(p*r2*score))
    log_risk=risk/float(np.sum(p*r2))
    conditional=score[points[:,0]>0]
    conditional_p=p[points[:,0]>0]/p[points[:,0]>0].sum()
    conditional_mean=np.sum(conditional_p*conditional)
    shape=float(np.sum(conditional_p*(conditional-conditional_mean)**2)/8)
    boundary=(-.5*c/(sigma*sigma))*math.exp(-1)/zx*float(np.sum(dose*np.array([_a(t) for t in times])*np.exp(-c*c*np.array([_a(t) for t in times])**2/(2*sigma*sigma))))/math.sqrt(1+1/sigma**2)
    return dict(mass=mass,risk=risk,log_risk=log_risk,conditional_h2_coefficient=shape,
                score_mean=float(np.sum(p*score)),mass_flux=boundary,log_partition_ratio=float(logk))


def _finite_mass(route, zx, order=96):
    y,w=hermgauss(order)
    ys=np.sqrt(2)*y
    lo=np.full(order,-.1);hi=np.full(order,.1)
    for _ in range(46):
        mid=(lo+hi)/2
        mapped,_=_plateau_map(route,np.stack([mid,ys],1))
        positive=mapped[:,0]>0
        hi=np.where(positive,mid,hi);lo=np.where(positive,lo,mid)
    root=(lo+hi)/2
    q,weights=leggauss(12)
    x=.5*root[:,None]*(q[None,:]+1)

    change=-.5*root*np.sum(weights[None,:]*np.exp(-(x*x-1)**2),axis=1)/zx
    return float(np.sum(w*change)/math.sqrt(math.pi))


def _finite_metrics(route,target_tilt=.05,target_k=1.,order=96):
    points,p,zx=_quadrature(order)
    moved,ld=_plateau_map(route,points)
    loga=_energy(points,0.,1.)-_energy(moved,target_tilt,target_k)+ld
    logk=float(logsumexp(np.log(p)+_energy(points,0.,1.)-_energy(points,target_tilt,target_k)))
    second=float(logsumexp(np.log(p)+2*loga)-2*logk)
    fitted_loga=_energy(points,0.,1.)-_energy(moved,0.,1.)+ld
    affinity=float(np.sum(p*np.exp(.5*fitted_loga)))
    return dict(mass_change=_finite_mass(route,zx,order),risk=math.expm1(second),
                log_risk=second,change_of_variables_error=float(math.expm1(logsumexp(np.log(p)+loga)-logk)),
                hellinger_squared=1-affinity)


def _native_finite_verification(n_steps=32):
    points=torch.tensor([[-1.,-.6],[-.02,.2],[.03,-.4],[1.2,.3]],dtype=torch.float64)
    endpoints=[];weights=[];native_error=0.;inverse_error=0.;inverse_logdet_error=0.
    for c in (-.5,.5):
        off=_route(c,n_steps=n_steps)
        off_value,off_ld=off.forward(points)
        native_error=max(native_error,float((off_value-points).abs().max()),float(off_ld.abs().max()))
        for conjugated in (False,True):
            route=_route(c,epsilon=.02,n_steps=n_steps,conjugated=conjugated)
            actual,ld=route.forward(points)
            vectorized,vld=_plateau_map(route,points.numpy())
            native_error=max(native_error,float(np.max(np.abs(vectorized-actual.numpy()))),float(np.max(np.abs(vld-ld.numpy()))))
            restored,ild=route.inverse(actual)
            inverse_error=max(inverse_error,float((restored-points).abs().max()))
            inverse_logdet_error=max(inverse_logdet_error,float((ld+ild).abs().max()))
            if conjugated:
                endpoints.append(actual)
                weights.append(-DoubleWellSystem().effective_u(actual,1)+DoubleWellSystem().effective_u(points,0)+ld)
    conjugated_endpoint_error=float((endpoints[0]-endpoints[1]).abs().max())
    conjugated_weight_error=float((weights[0]-weights[1]).abs().max())
    absolute=[_plateau_map(_route(c,epsilon=.02,n_steps=n_steps),points.numpy())[0] for c in (-.5,.5)]
    absolute_separation=float(np.max(np.abs(absolute[0]-absolute[1])))
    return dict(n_steps=n_steps,native_ad_and_vectorized_error=native_error,inverse_error=inverse_error,
                inverse_logdet_error=inverse_logdet_error,conjugated_endpoint_error=conjugated_endpoint_error,
                conjugated_logweight_error=conjugated_weight_error,absolute_endpoint_separation=absolute_separation)


def run_analytical(output=None):

    epsilon=1e-4;n_steps=32
    continuous=[];finite=[]
    for sigma in (1.,.6):
        for c in (-.5,0.,.5):
            prediction=_derivatives(c,sigma)
            discrete=_derivatives(c,sigma,n_steps=n_steps)
            minus=_finite_metrics(_route(c,sigma,-epsilon,n_steps))
            plus=_finite_metrics(_route(c,sigma,epsilon,n_steps))
            derivative={key:(plus[key]-minus[key])/(2*epsilon) for key in ("risk","log_risk","mass_change")}
            continuous.append(dict(c=c,sigma=sigma,**prediction))
            finite.append(dict(c=c,sigma=sigma,epsilon=epsilon,n_steps=n_steps,
                               measured_derivative=derivative,discrete_prediction=discrete,
                               mass_derivative_error=abs(derivative["mass_change"]-discrete["mass"]),
                               risk_derivative_error=abs(derivative["risk"]-discrete["risk"]),
                               continuous_discrete_mass_error=abs(discrete["mass"]-prediction["mass"]),
                               maximum_change_of_variables_error=max(abs(plus["change_of_variables_error"]),abs(minus["change_of_variables_error"]))))
    same_side=[]
    for c in (-.25,-.5):
        base=_finite_metrics(_route(c,epsilon=0.,n_steps=n_steps))
        actual=_finite_metrics(_route(c,epsilon=epsilon,n_steps=n_steps))
        half=_finite_metrics(_route(c,epsilon=epsilon/2,n_steps=n_steps))
        same_side.append(dict(c=c,epsilon=epsilon,prediction=_derivatives(c,n_steps=n_steps),
                              finite_mass_change=actual["mass_change"],finite_risk_change=actual["risk"]-base["risk"],
                              mass_half_dose_residual=actual["mass_change"]-2*half["mass_change"],
                              risk_half_dose_residual=(actual["risk"]-base["risk"])-2*(half["risk"]-base["risk"])))
    a_finite=_finite_metrics(_route(0.,epsilon=.02,n_steps=n_steps))
    a_prediction=_derivatives(0.,n_steps=n_steps)

    points,p,_=_quadrature()
    times=np.arange(n_steps)/n_steps;dose=times*(1-times)/n_steps
    k1=.75
    logw=_energy(points,0.,1.)-_energy(points,0.,k1)
    logk=float(logsumexp(np.log(p)+logw))
    psi=np.exp(-np.square(points).sum(1)/2)
    b_formula=float(2*(1-k1)*.5*dose.sum()*np.sum(p*points[:,1]**2*psi*np.exp(2*(logw-logk))))
    b_minus=_finite_metrics(_route(0.,epsilon=-epsilon,n_steps=n_steps),0.,k1)
    b_plus=_finite_metrics(_route(0.,epsilon=epsilon,n_steps=n_steps),0.,k1)
    b_measured=(b_plus["risk"]-b_minus["risk"])/(2*epsilon)
    fitted_full=_finite_metrics(_route(0.,epsilon=.02,n_steps=n_steps),0.,1.)
    fitted_half=_finite_metrics(_route(0.,epsilon=.01,n_steps=n_steps),0.,1.)
    fitted_prediction=_derivatives(0.,n_steps=n_steps,target_tilt=0.)

    m,c,d,beta=2,1.,.25,1.
    def entropic_energy(z):
        return .5*z[0]**2+.5*torch.exp(-c*z[0]**2+d*z[0]**4)*z[1:].square().sum()
    origin=torch.zeros(m+1,dtype=torch.float64)
    hessian=torch.func.hessian(entropic_energy)(origin)
    def marginal_free_energy(x):
        return .5*x*x+(m/(2*beta))*(-c*x*x+d*x**4)
    h=1e-3
    marginal_curvature=(marginal_free_energy(h)+marginal_free_energy(-h)-2*marginal_free_energy(0.))/h**2
    stationary_minima=math.sqrt((m*c-beta)/(2*m*d))
    negative_controls={
        "A_invariant_boundary":dict(exact_mass_change=0.,measured_mass_change=a_finite["mass_change"],
            measured_conditional_h2=a_finite["hellinger_squared"],predicted_conditional_h2=.02**2*a_prediction["conditional_h2_coefficient"]),
        "B_irrelevant_true_saddle":dict(source_saddle_hessian=[-4.,1.],target_saddle_hessian=[-4.,k1],
            predicted_positive_risk_derivative=b_formula,measured_risk_derivative=b_measured,absolute_error=abs(b_formula-b_measured)),
        "C_entropic_bottleneck":dict(stationary_point_gradient_norm=float(torch.func.grad(entropic_energy)(origin).norm()),
            energy_hessian_eigenvalues=torch.linalg.eigvalsh(hessian).tolist(),energy_stationary_points="Only origin: dU/dy=a(x)y=0, a(x)>0, then dU/dx=x=0",
            free_energy_origin_curvature=float(marginal_curvature),free_energy_minima=[-stationary_minima,stationary_minima],
            free_energy_barrier=marginal_free_energy(0.)-marginal_free_energy(stationary_minima)),
        "D_fitted_endpoint":dict(risk_before=0.,risk_full=fitted_full["risk"],risk_half=fitted_half["risk"],
            predicted_quadratic_coefficient=8*fitted_prediction["conditional_h2_coefficient"],
            measured_quadratic_coefficient=fitted_full["risk"]/.02**2,
            quadratic_half_dose_residual=fitted_full["risk"]-4*fitted_half["risk"]),
    }
    native=_native_finite_verification(n_steps)
    convergence=0.
    for sigma in (1.,.6):
        for offset in (-.5,-.25,0.,.5):
            low=_derivatives(offset,sigma,order=64)
            high=_derivatives(offset,sigma,order=96)
            convergence=max(convergence,max(abs(low[k]-high[k]) for k in ("mass","risk","log_risk","conditional_h2_coefficient")))
    gates={
        "opposite_mass_responses":all(row["mass"]*(-row["c"])>0 for row in continuous if row["c"]!=0),
        "flux_matches_score":max(abs(row["mass"]-row["mass_flux"]) for row in continuous)<2e-9,
        "finite_map_predictions":max(max(row["mass_derivative_error"],row["risk_derivative_error"]) for row in finite)<2e-8,
        "same_side_mass_opposite_risk":all(row["finite_mass_change"]>0 for row in same_side) and same_side[0]["finite_risk_change"]>0 and same_side[1]["finite_risk_change"]<0,
        "invariant_mass_nonzero_shape":abs(a_finite["mass_change"])<1e-12 and a_finite["hellinger_squared"]>1e-9,
        "irrelevant_true_saddle_harmful":b_measured>0 and abs(b_measured-b_formula)<2e-8,
        "entropic_bottleneck_no_energy_saddle":bool((torch.linalg.eigvalsh(hessian)>0).all()) and marginal_curvature<0,
        "fitted_endpoint_quadratic_harm":fitted_full["risk"]>0 and fitted_half["risk"]>0 and abs(fitted_full["risk"]-4*fitted_half["risk"])<.05*fitted_full["risk"],
        "native_finite_inverse_and_volume":max(native["native_ad_and_vectorized_error"],native["inverse_error"],native["inverse_logdet_error"])<1e-10,
        "conjugated_negative_control":max(native["conjugated_endpoint_error"],native["conjugated_logweight_error"])<1e-10 and native["absolute_endpoint_separation"]>1e-6,
        "quadrature_convergence":convergence<2e-8,
    }
    result=dict(status="passed" if all(gates.values()) else "failed",scope="double_well_controls",
                gates=gates,continuous_responses=continuous,finite_step_responses=finite,same_side_pair=same_side,
                negative_controls=negative_controls,native_finite_map=native,quadrature_order64_order96_maximum_change=convergence,
                source_x_tail_probability_upper_at_3=2*math.exp(-64)/96/_quadrature()[2],
                field_amplitude=.5,field_envelope="t(1-t)",time_grid_nodes=n_steps,
                quadrature_rule="split_legendre_x_hermite_y")
    if output is not None:
        destination=Path(output)
        if destination.suffix.lower()!=".json":
            destination=destination/"result.json"
        if destination.exists():
            raise FileExistsError("E0输出已存在，不能静默覆盖")
        destination.parent.mkdir(parents=True,exist_ok=True)
        temporary=destination.with_suffix(destination.suffix+".pending")
        temporary.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+"\n")
        temporary.replace(destination)
    return result
