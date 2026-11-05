from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from mpmath import iv

from .interventions import GeometryQualification, IdentityLoop, ProtectedTranslation
from .transport import Transport
from .geometry import MolecularProtection


class CertificationFailure(RuntimeError):
    pass


def _i(x):
    return x if hasattr(x,"_mpi_") else iv.mpf(float(x))


def _lo(x):
    return float(np.nextafter(float(_i(x).a),-math.inf))


def _hi(x):
    return float(np.nextafter(float(_i(x).b),math.inf))


def _abs(x):
    return max(abs(_lo(x)),abs(_hi(x)))


def _dot(a,b):
    return sum((x*y for x,y in zip(a,b)),0)


def _cross(a,b):
    return [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]


def _sub(a,b):
    return [x-y for x,y in zip(a,b)]


def _norm(a):
    return iv.sqrt(sum((x**2 for x in a),_i(0)))


def _matvec(matrix,vector,transpose=False):
    matrix=np.asarray(matrix)
    if transpose:
        matrix=matrix.T
    return [_dot([_i(v) for v in row],vector) for row in matrix]


def _frobenius(matrix):
    return _hi(iv.sqrt(sum((_i(v)**2 for v in np.asarray(matrix).ravel()),_i(0))))


def _tanh(x):

    return 2/(1+iv.exp(-2*x))-1


@dataclass
class _Dual:
    value: object
    gradient: list

    @classmethod
    def constant(cls,value,dim):
        return cls(_i(value),[_i(0)]*dim)

    def cast(self,value):
        return value if isinstance(value,_Dual) else self.constant(value,len(self.gradient))

    def __add__(self,other):
        other=self.cast(other)
        return _Dual(self.value+other.value,[a+b for a,b in zip(self.gradient,other.gradient)])

    __radd__=__add__

    def __neg__(self):
        return _Dual(-self.value,[-a for a in self.gradient])

    def __sub__(self,other):
        return self+-self.cast(other)

    def __rsub__(self,other):
        return self.cast(other)+-self

    def __mul__(self,other):
        other=self.cast(other)
        return _Dual(self.value*other.value,[a*other.value+self.value*b for a,b in zip(self.gradient,other.gradient)])

    __rmul__=__mul__

    def __truediv__(self,other):
        other=self.cast(other)
        return _Dual(self.value/other.value,[(a*other.value-self.value*b)/(other.value**2) for a,b in zip(self.gradient,other.gradient)])

    def __pow__(self,n):
        if n!=2:
            raise ValueError("区间一阶代数仅需平方")
        return self*self

    def exp(self):
        value=iv.exp(self.value)
        return _Dual(value,[value*g for g in self.gradient])

    def sqrt(self):
        value=iv.sqrt(self.value)
        return _Dual(value,[g/(2*value) for g in self.gradient])


def _dual_atan2(y,x):
    denominator=x.value*x.value+y.value*y.value
    if _lo(denominator)<=0:
        raise CertificationFailure("atan2点区间包含原点")
    return _Dual(iv.atan2(y.value,x.value),[(x.value*dy-y.value*dx)/denominator for dy,dx in zip(y.gradient,x.gradient)])


def _raw_system(system):
    return getattr(system,"system",system)


def _raw_coordinates(system,z):
    if hasattr(system,"raw"):
        return [c*_i(b)+_i(a) for a,b,c in zip(system.location.tolist(),system.scale.tolist(),z)]
    return list(z)


def _heavy_positions(system,z,dual=False):
    raw=_raw_system(system);chart=raw.chart
    native=_raw_coordinates(system,z)
    zero=native[0]*0
    exp=(lambda a:a.exp()) if dual else iv.exp
    result=[None]*chart.n_atoms
    origin,first,second=chart.triad
    result[origin]=[zero,zero,zero]
    result[first]=[exp(native[0]),zero,zero]
    result[second]=[native[1],exp(native[2]),zero]
    for k,atom in enumerate(chart.heavy):
        result[atom]=list(native[3+3*k:6+3*k])
    return result,native


def _full_positions(system,z):
    result,native=_heavy_positions(system,z)
    chart=_raw_system(system).chart
    for k,(h,p) in enumerate(zip(chart.hydrogens,chart.parents)):
        u=native[chart.split+2*k:chart.split+2*k+2]
        s=sum((v**2 for v in u),_i(0))
        v=iv.exp(-s/2)
        if _lo(s)>0:
            factor=2*iv.sqrt(v*(1-v)/s)
        else:


            factor=iv.mpf([0,_hi(iv.sqrt(2))])
        direction=[factor*u[0],factor*u[1],2*v-1]
        length=_i(float(chart.lengths[k]))
        result[h]=[a+length*b for a,b in zip(result[p],direction)]
    return result


def _clearance_interval(protection,z):
    if not isinstance(protection,MolecularProtection):
        raise CertificationFailure("实际分子证书需要MolecularProtection的明确几何支撑")
    x=_full_positions(protection.system,z)
    margins=[]
    for value,lower,upper in zip(z,protection.lower.tolist(),protection.upper.tolist()):
        margins.extend((value-_i(lower),_i(upper)-value))
    distances=[]
    for pair,denominator in zip(protection.pairs.tolist(),protection.pair_lipschitz.tolist()):
        distance=_norm(_sub(x[pair[0]],x[pair[1]]))
        distances.append(distance)
        margins.append((distance-_i(protection.collision_nm))/_i(denominator))
    for triple,denominator in zip(protection.angle_triples.tolist(),protection.angle_lipschitz.tolist()):
        a,b,c=triple
        u,v=_sub(x[a],x[b]),_sub(x[c],x[b])
        value=_norm(_cross(u,v))-_i(protection.sine_min)*_norm(u)*_norm(v)
        margins.append(value/_i(denominator))
    for q,sign,denominator in zip(protection.chiral_quads.tolist(),protection.chiral_signs.tolist(),protection.chiral_lipschitz.tolist()):
        vectors=[_sub(x[q[k]],x[q[3]]) for k in range(3)]
        signed=_dot(_cross(vectors[0],vectors[1]),vectors[2])*_i(sign)
        margins.append((signed-_i(protection.chiral_margin))/_i(denominator))
    if protection.has_neck:
        for distance,neck,denominator in zip(distances,protection.neck.tolist(),protection.pair_lipschitz.tolist()):
            value=distance-_i(neck)
            absolute=iv.mpf([0,max(abs(_lo(value)),abs(_hi(value)))]) if _lo(value)<=0<=_hi(value) else abs(value)
            margins.append((absolute-_i(protection.neck_exclusion_nm))/_i(denominator))
    tau=_i(protection.temperature)

    shift=min(_lo(v) for v in margins)
    return _i(shift)-tau*iv.log(sum((iv.exp(-(_i(v)-_i(shift))/tau) for v in margins),_i(0)))


def _clearance_lipschitz(protection):
    system=protection.system;chart=_raw_system(system).chart
    scale=[_i(v) for v in protection.scale.tolist()]
    lo=_raw_coordinates(system,[_i(v) for v in protection.lower.tolist()])
    hi=_raw_coordinates(system,[_i(v) for v in protection.upper.tolist()])
    B=[_i(0)]*chart.n_atoms;L=[_i(0)]*chart.n_atoms
    origin,first,second=chart.triad
    B[first]=iv.exp(hi[0]);L[first]=B[first]*scale[0]
    B[second]=iv.sqrt(_i(max(_abs(lo[1]),_abs(hi[1])))**2+iv.exp(2*hi[2]))
    L[second]=_i(max(_hi(scale[1]),_hi(iv.exp(hi[2])*scale[2])))
    for k,atom in enumerate(chart.heavy):
        indices=range(3+3*k,6+3*k)
        B[atom]=iv.sqrt(sum((_i(max(_abs(lo[j]),_abs(hi[j])))**2 for j in indices),_i(0)))
        L[atom]=_i(max(_hi(scale[j]) for j in indices))


    for k,(h,p) in enumerate(zip(chart.hydrogens,chart.parents)):
        length=_i(float(chart.lengths[k]))
        B[h]=B[p]+length
        L[h]=L[p]+iv.sqrt(2)*length*_i(max(_hi(v) for v in scale[chart.split+2*k:chart.split+2*k+2]))
    upper=1.
    for (a,b),denominator in zip(protection.pairs.tolist(),protection.pair_lipschitz.tolist()):
        upper=max(upper,_hi((L[a]+L[b])/_i(denominator)))
    for (a,b,c),denominator in zip(protection.angle_triples.tolist(),protection.angle_lipschitz.tolist()):
        lu,lv=L[a]+L[b],L[c]+L[b];bu,bv=B[a]+B[b],B[c]+B[b]
        upper=max(upper,_hi((1+_i(protection.sine_min))*(lu*bv+lv*bu)/_i(denominator)))
    for q,sign,denominator in zip(protection.chiral_quads.tolist(),protection.chiral_signs.tolist(),protection.chiral_lipschitz.tolist()):
        lips=[L[q[k]]+L[q[3]] for k in range(3)];mags=[B[q[k]]+B[q[3]] for k in range(3)]
        derivative=sum((lips[k]*mags[(k+1)%3]*mags[(k+2)%3] for k in range(3)),_i(0))
        upper=max(upper,_hi(abs(_i(sign))*derivative/_i(denominator)))
    return upper


def _velocity_bounds(base,k):
    U,V,W=[a.detach().cpu().numpy() for a in (base.U,base.V,base.W)]
    c=base.spline_coefficients(k).detach().cpu().numpy()
    un,vn=_frobenius(U),_frobenius(V)
    wcn=_hi(iv.sqrt(sum(((_i(W[i,j])*_i(c[j]))**2 for i in range(len(W)) for j in range(64)),_i(0))))
    speed=_hi(_i(_frobenius(W))*iv.sqrt(sum((_i(v)**2 for v in c),_i(0))))
    lip=_hi(_i(wcn)*_i(vn)*_i(un))
    curvature=_hi(_i(wcn)*(4/(3*iv.sqrt(3)))*_i(un)**2*(_i(vn)**2+_i(vn)))
    t=_i(k)/_i(base.n_steps);envelope=t*(1-t)
    for alpha,amplitude,sigma in zip(base.alpha.detach().tolist(),base.amplitudes.tolist(),base.bandwidths.tolist()):
        strength=_i(alpha)*_i(amplitude)*envelope
        speed=_hi(_i(speed)+strength*iv.exp(-iv.mpf('.5'))/_i(sigma))
        lip=_hi(_i(lip)+strength/_i(sigma)**2)
        third=(3*iv.sqrt(3)*iv.exp(-iv.mpf('1.5'))+3*iv.exp(-iv.mpf('.5')))/_i(sigma)**3
        curvature=_hi(_i(curvature)+strength*third)
    return speed,lip,curvature


def _forward_interval_step(base,z,k):
    c=base.spline_coefficients(k).detach().cpu().numpy()
    if not np.any(c) and not bool((base.alpha!=0).any()):
        return z,None
    U,V,W=[a.detach().cpu().numpy() for a in (base.U,base.V,base.W)]
    first=[_tanh(v) for v in _matvec(U,z)]
    hidden=[_tanh(v) for v in _matvec(V,first)]
    velocity=_matvec(W,[a*_i(b) for a,b in zip(hidden+[_i(1)],c)])
    fields=[]
    centers=base.centers if base.time_centers is None else base.time_centers[k]
    for alpha,amplitude,sigma,center in zip(base.alpha.detach().tolist(),base.amplitudes.tolist(),base.bandwidths.tolist(),centers.tolist()):
        strength=_i(alpha)*_i(amplitude)*_i(k)/_i(base.n_steps)*(1-_i(k)/_i(base.n_steps))
        delta=[a-_i(b) for a,b in zip(z,center)]
        psi=iv.exp(-_dot(delta,delta)/(2*_i(sigma)**2))
        velocity=[v-strength*psi*d/_i(sigma)**2 for v,d in zip(velocity,delta)]
        fields.append((strength,psi,delta,_i(sigma)))
    moved=[a+b/_i(base.n_steps) for a,b in zip(z,velocity)]
    return moved,(first,hidden,c,U,V,W,fields)


def _vjp_interval(gradient,cache,n_steps):
    if cache is None:
        return gradient
    first,hidden,c,U,V,W,fields=cache
    pulled=_matvec(W,gradient,True)
    pulled=[a*_i(b)*(1-h*h) for a,b,h in zip(pulled[:64],c[:64],hidden)]
    pulled=_matvec(V,pulled,True)
    pulled=[a*(1-h*h) for a,h in zip(pulled,first)]
    pulled=_matvec(U,pulled,True)
    for strength,psi,delta,sigma in fields:
        dot=_dot(delta,gradient)
        pulled=[v+strength*psi*(d*dot/sigma**4-g/sigma**2) for v,d,g in zip(pulled,delta,gradient)]
    return [g+v/_i(n_steps) for g,v in zip(gradient,pulled)]


@dataclass(frozen=True)
class _Bound:
    magnitude: float
    first: float
    second: float


def _mul(a,b):
    return _Bound(_hi(_i(a.magnitude)*_i(b.magnitude)),
        _hi(_i(a.first)*_i(b.magnitude)+_i(a.magnitude)*_i(b.first)),
        _hi(_i(a.second)*_i(b.magnitude)+2*_i(a.first)*_i(b.first)+_i(a.magnitude)*_i(b.second)))


def _add(a,b):
    return _Bound(*[_hi(_i(x)+_i(y)) for x,y in zip((a.magnitude,a.first,a.second),(b.magnitude,b.first,b.second))])


def _heavy_atom_derivatives(system,center,radius):
    chart=_raw_system(system).chart
    scale=system.scale.detach().tolist() if hasattr(system,"scale") else [1.]*system.dim
    raw=_raw_coordinates(system,center)
    L=[0.]*chart.n_atoms;H=[0.]*chart.n_atoms
    origin,first,second=chart.triad
    e0=iv.exp(raw[0]+_i(scale[0])*_i(radius));e2=iv.exp(raw[2]+_i(scale[2])*_i(radius))
    L[first]=_hi(e0*_i(scale[0]));H[first]=_hi(e0*_i(scale[0])**2)
    L[second]=max(scale[1],_hi(e2*_i(scale[2])));H[second]=_hi(e2*_i(scale[2])**2)
    for k,atom in enumerate(chart.heavy):
        L[atom]=max(scale[3+3*k:6+3*k])
    return L,H


def _angle_bound(system,center,radius,quad,positions,L,H):
    def vector(a,b):
        value=_sub(positions[a],positions[b])
        lip=_hi(_i(L[a])+_i(L[b]));curvature=_hi(_i(H[a])+_i(H[b]))
        mag=_hi(_norm(value)+_i(lip)*_i(radius))
        return value,_Bound(mag,lip,curvature)
    a,A=vector(quad[0],quad[1]);b,B=vector(quad[2],quad[1]);c,C=vector(quad[3],quad[2])
    norm_b_lower=_lo(_norm(b)-_i(B.first)*_i(radius))
    cross_ab=_mul(A,B);cross_bc=_mul(B,C)
    cross_a_lower=_lo(_norm(_cross(a,b))-_i(cross_ab.first)*_i(radius))
    cross_c_lower=_lo(_norm(_cross(b,c))-_i(cross_bc.first)*_i(radius))
    if min(norm_b_lower,cross_a_lower,cross_c_lower)<=0:
        raise CertificationFailure("完整邻域无法排除二面角键长或定义平面退化")

    xbound=_add(_mul(_mul(A,C),_mul(B,B)),_mul(_mul(A,B),_mul(C,B)))
    norm_b=_Bound(B.magnitude,B.first,_hi(_i(B.second)+_i(B.first)**2/_i(norm_b_lower)))
    ybound=_mul(norm_b,_mul(_mul(B,A),C))
    lw=_hi(iv.sqrt(_i(xbound.first)**2+_i(ybound.first)**2))
    hw=_hi(iv.sqrt(_i(xbound.second)**2+_i(ybound.second)**2))
    denominator=_lo(_i(cross_a_lower)*_i(cross_c_lower))
    deg=180/iv.pi
    return _Bound(180.,_hi(deg*_i(lw)/_i(denominator)),
                  _hi(deg*(_i(hw)/_i(denominator)+_i(lw)**2/_i(denominator)**2)))


def _level_duals_and_bounds(system,center,radius):
    dim=system.dim
    dual=[]
    for k,value in enumerate(center):
        gradient=[_i(0)]*dim;gradient[k]=_i(1)
        dual.append(_Dual(value,gradient))
    positions,_=_heavy_positions(system,dual,True)
    plain,_=_heavy_positions(system,center)
    L,H=_heavy_atom_derivatives(system,center,radius)
    angles=[];bounds=[]
    for quad in system.torsion_quads:
        a=_sub(positions[quad[0]],positions[quad[1]])
        b=_sub(positions[quad[2]],positions[quad[1]])
        c=_sub(positions[quad[3]],positions[quad[2]])
        x=_dot(a,c)*_dot(b,b)-_dot(a,b)*_dot(c,b)
        y=_dot(_cross(b,a),c)*_dot(b,b).sqrt()
        angles.append(_dual_atan2(y,x)*(180/iv.pi))
        bounds.append(_angle_bound(system,center,radius,quad,plain,L,H))
    levels=[];level_bounds=[]
    for i in range(0,len(angles),2):
        shifted=[];maxima=[]
        for angle,bound,reference in zip(angles[i:i+2],bounds[i:i+2],(-70.,-40.)):
            midpoint=.5*(_lo(angle.value)+_hi(angle.value))
            winding=math.floor((midpoint-reference+180)/360)
            difference=angle-reference-360*winding
            maximum=_hi(_i(_abs(difference.value))+_i(bound.first)*_i(radius))
            if maximum>=180:
                raise CertificationFailure("整个邻域无法排除周期角的wrap分支切口")
            shifted.append(difference);maxima.append(maximum)
        levels.append((shifted[0]**2+shifted[1]**2)/1600-1)
        lp=_hi((2*_i(maxima[0])*_i(bounds[i].first)+2*_i(maxima[1])*_i(bounds[i+1].first))/1600)
        hp=_hi(2*(_i(bounds[i].first)**2+_i(bounds[i+1].first)**2+_i(maxima[0])*_i(bounds[i].second)+_i(maxima[1])*_i(bounds[i+1].second))/1600)
        level_bounds.append(_Bound(0.,lp,hp))
    return levels,level_bounds


def certify_geometry(route,system,boundary,normals,sigma,active_patch,eta,check_budget=None):

    result={"method":"interval_point_gradient_and_analytic_global_H_on_plateau_images", "nodes":[],"certified":False,"reasons":[]}
    previous=iv.dps;iv.dps=35
    maximum_angle=0.;maximum_height=0.;all_unique=True;all_plateau=True
    try:
        if not isinstance(route.base,Transport) or not isinstance(route.protection,MolecularProtection):
            raise CertificationFailure("此实现只认证实际Transport及实际MolecularProtection，不接受手填几何oracle")
        route.base.validate_parameters()
        if any(parameter.dtype!=torch.float64 for parameter in route.base.parameters()):
            raise CertificationFailure("真实输运证书要求float64冻结参数")
        if hasattr(system,"mechanism_boundary_levels"):
            raise CertificationFailure("当前分子证书不接受自定义或手填分界oracle")
        first=route.base.protection.state_dict();second=route.protection.state_dict()
        if route.base.protection.specification()!=route.protection.specification() or first.keys()!=second.keys() or any(not torch.equal(first[key],second[key]) for key in first):
            raise CertificationFailure("基流与活动管的保护/物理图状态不匹配")
        declared=system.state_dict();actual=route.protection.system.state_dict()
        if declared.keys()!=actual.keys() or any(not torch.equal(declared[key],actual[key]) for key in declared):
            raise CertificationFailure("分界目标与真实保护图的坐标/物理状态不匹配")
        if system.name not in ("ALDP","ALA4") or len(system.torsion_quads) not in (2,4):
            raise CertificationFailure("没有当前分子真实αR圆盘的可核实拓扑")
        if not math.isfinite(sigma) or sigma<=0:
            raise CertificationFailure("带宽必须有限且严格为正")
        active=sorted(int(k) for k,v in eta.items() if float(v)>0) if isinstance(eta,dict) else torch.nonzero(torch.as_tensor(eta)>0).flatten().tolist()
        if any(k<0 or k>=route.n_steps for k in active):
            raise CertificationFailure("活动节点越出实际固定网格")
        clearance_lip=_i(_clearance_lipschitz(route.protection))
        identity=not bool((route.base.C!=0).any()) and not bool((route.base.alpha!=0).any())
        result["active_nodes_expected"]=len(active)
        if not active:
            raise CertificationFailure("时窗没有活动节点")
        for k in active:
            if check_budget is not None:check_budget()
            shifted=torch.as_tensor(boundary[k],dtype=torch.float64)
            n=torch.as_tensor(normals[k],dtype=torch.float64)
            loop=route.loops[k]
            displacement=torch.zeros_like(shifted) if isinstance(loop,IdentityLoop) else loop.displacement
            if not isinstance(loop,(IdentityLoop,ProtectedTranslation)):
                raise CertificationFailure("回路不是已实现的实际保护平移")
            if isinstance(loop,ProtectedTranslation) and loop.protection.specification()!=route.protection.specification():
                raise CertificationFailure("回路平移与真实活动场的共同保护不同")
            point=shifted-displacement
            anchor_distance=_hi(_norm([_i(a)-_i(b) for a,b in zip(shifted.tolist(),route.center.tolist())]))
            tangent_radius=_hi(2*_i(sigma)+_i(anchor_distance))
            domain_radius=_hi(_i(tangent_radius)+_i('.1')*_i(sigma))
            z=[_i(a)-_i(b) for a,b in zip(shifted.tolist(),displacement.tolist())]


            tube_lower=_lo(_clearance_interval(route.protection,z)-clearance_lip*(_norm([_i(v) for v in displacement.tolist()])+_i(domain_radius)))
            if tube_lower<route.protection.width:
                raise CertificationFailure(f"节点{k}完整2σ及平移活动管未获平台证书：lower={tube_lower:g}<width={route.protection.width:g}")
            radius=domain_radius;flow_lip=1.;flow_hessian=0.;caches=[];minimum_clearance=math.inf
            for j in range(k,route.n_steps) if not identity else (k,):
                if check_budget is not None:check_budget()
                lower=_lo(_clearance_interval(route.protection,z)-clearance_lip*_i(radius))
                minimum_clearance=min(minimum_clearance,lower)
                if lower<route.protection.width:
                    raise CertificationFailure(f"节点{k}剩余图第{j}步完整像球离开平台充分域：lower={lower:g},width={route.protection.width:g},radius={radius:g}")
                if identity:
                    break
                _,lv,hv=_velocity_bounds(route.base,j)
                ls=1+_i(lv)/_i(route.n_steps);hs=_i(hv)/_i(route.n_steps)
                flow_hessian=_hi(ls*_i(flow_hessian)+hs*_i(flow_lip)**2)
                flow_lip=_hi(ls*_i(flow_lip));radius=_hi(ls*_i(radius))
                if not all(math.isfinite(value) for value in (radius,flow_lip,flow_hessian)):
                    raise CertificationFailure("剩余图导数界溢出")
                z,cache=_forward_interval_step(route.base,z,j)
                caches.append(cache)
            levels,bounds=_level_duals_and_bounds(system,z,radius)
            if not 0<=active_patch<len(levels):
                raise CertificationFailure("活动真实圆盘片编号错误")
            for other,(level,bound) in enumerate(zip(levels,bounds)):
                if other!=active_patch and _hi(level.value+_i(bound.first)*_i(radius))>=0:
                    raise CertificationFailure(f"节点{k}不能排除ALA4另一圆盘片边界，patch={other}")
            level=levels[active_patch];endpoint_bound=bounds[active_patch]
            hf=_hi(_i(endpoint_bound.second)*_i(flow_lip)**2+_i(endpoint_bound.first)*_i(flow_hessian))
            gradient=level.gradient
            for cache in reversed(caches):gradient=_vjp_interval(gradient,cache,route.n_steps)
            ni=[_i(v) for v in n.tolist()];norm=_norm(ni)
            if _lo(norm)<=0:raise CertificationFailure("追踪法向未定义")
            ni=[v/norm for v in ni]
            parallel=_dot(gradient,ni)
            tangent=_norm([g-parallel*v for g,v in zip(gradient,ni)])
            m=_lo(parallel-_i(hf)*_i(domain_radius))
            tangent_bound=_hi(tangent+_i(hf)*_i(domain_radius))
            angle=_hi(2*iv.atan2(_i(tangent_bound),_i(m))) if m>0 else math.inf
            height=_hi((_i(_abs(level.value))+tangent*_i(tangent_radius)+_i('.5')*_i(hf)*_i(tangent_radius)**2)/_i(m)) if m>0 else math.inf
            unique=m>0 and height<_lo(iv.mpf(".1")*_i(sigma))
            record=dict(node=k,full_anchor_neighborhood_radius=2*sigma,boundary_center_domain_radius=domain_radius,
                        remaining_map_lipschitz_upper=flow_lip,remaining_map_hessian_upper=flow_hessian,
                        endpoint_level_hessian_upper=endpoint_bound.second,pulled_level_hessian_upper=hf,
                        normal_derivative_lower=m,center_tangent_gradient_upper=_hi(tangent),
                        maximum_normal_pair_angle_upper=angle,maximum_vertical_departure_upper=height,
                        root_residual_upper=_abs(level.value),minimum_plateau_clearance_lower=minimum_clearance,
                        plane_graph_exists_and_unique=unique,other_patch_excluded=True,entire_activity_tube_on_plateau=True)
            result["nodes"].append(record)
            maximum_angle=max(maximum_angle,angle);maximum_height=max(maximum_height,height);all_unique&=unique
        result["certified"]=True
    except (CertificationFailure,ValueError,RuntimeError,OverflowError,ZeroDivisionError) as exc:
        result["reasons"].append(str(exc));all_unique=False;all_plateau=False
    finally:
        iv.dps=previous
    passed=result["certified"] and all_unique and all_plateau and maximum_angle<.1 and maximum_height<.1*sigma
    if not passed and not result["reasons"]:
        result["reasons"].append(f"解析充分界未满足门：normal_angle<0.1得到{maximum_angle:g}，height<0.1σ得到{maximum_height:g}，unique={all_unique}")
    result["qualified"]=passed
    result["active_nodes_certified"]=len(result["nodes"])
    result["arithmetic_scope"]="Directed intervals enclose the mathematical finite map at its frozen float64 constants; implementation floating-point/inverse residual bounds remain part of the independent numerical validation used by P/I inference."
    result["protection_clearance_lipschitz_upper"]=_hi(clearance_lip) if "clearance_lip" in locals() else None
    result["derivation"]="Every remaining image ball lies on b=1. Compose analytic network/Gaussian L,H. True heavy-atom exp/linear chart and dot/cross/norm/atan2 bounds give global basin-level H. Directed interval values and reverse derivatives bound the center gradient. n·gradF>=m>0 and |F0|+|grad_tan|r+Hr²/2<m*(0.1σ) certify the unique implicit graph on the whole tangential disk."
    reason="通过真实分子全邻域充分条件" if passed else "; ".join(result["reasons"])
    qualification=GeometryQualification(all_unique,maximum_angle if result["certified"] else math.inf,
        maximum_height if result["certified"] else math.inf,sigma,all_plateau,certified=result["certified"],reason=reason)
    return qualification,result
