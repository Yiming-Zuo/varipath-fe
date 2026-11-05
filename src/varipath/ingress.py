from __future__ import annotations

import torch

from .physics import R_KJ


class IngressError(FloatingPointError):
    pass


def _physical_u(raw,positions,endpoint):
    temperature=raw.state(endpoint) if raw.name=="ALDP" else 300.
    s=1. if raw.name=="ALDP" else float(endpoint)
    result=torch.vmap(lambda x:raw.energy(x,s))(positions)/(R_KJ*temperature)
    if not torch.isfinite(result).all():
        raise IngressError("原生物理端点能量非有限，整批失败")
    return result


def _validate(raw,positions):
    if positions.ndim!=3 or positions.shape[1:]!=(raw.chart.n_atoms,3) or positions.dtype!=torch.float64 or not len(positions):
        raise IngressError("全坐标入口需要非空 float64 [batch,atoms,3]")
    if not torch.isfinite(positions).all():
        raise IngressError("全坐标含非有限值，整批失败")
    if raw.chart.constraint_error(positions)>1e-10:
        raise IngressError("原始构型违反含氢键长约束；入口不能投影")
    if not raw.in_domain_positions(positions).all():
        raise IngressError("原始或生成构型不属于全 L 型条件域")
    regions=raw.region_positions(positions)
    if not regions["valid"].all():
        raise IngressError("物理盆地标签不可判定，不能将退化点归入 remainder")
    return regions


def _frame_and_poles(raw,positions):
    chart=raw.chart
    origin,first,second=chart.triad
    delta=positions-positions[:,origin:origin+1]
    first_vector=delta[:,first]
    length=first_vector.norm(dim=-1,keepdim=True)
    if (length<=1e-12).any():
        raise IngressError("原始构型体固框长度退化，不能认证球面南极恒等扩展")
    ex=first_vector/length
    perpendicular=delta[:,second]-(delta[:,second]*ex).sum(-1,keepdim=True)*ex
    height=perpendicular.norm(dim=-1,keepdim=True)
    if (height<=1e-12).any():
        raise IngressError("原始构型体固框共线，不能认证球面南极恒等扩展")
    ey=perpendicular/height
    rotation=torch.stack((ex,ey,torch.linalg.cross(ex,ey)),-2)
    body=delta@rotation.transpose(-1,-2)
    directions=body[:,chart.hydrogens]-body[:,chart.parents]

    poles=((directions[...,:2]==0).all(-1)&(directions[...,2]<0)).any(-1)
    return rotation,poles


@torch.no_grad()
def transport_configurations(system,model,full_positions,*,check_budget=None):

    raw=getattr(system,"system",system)
    positions=torch.as_tensor(full_positions)
    if positions.ndim==2:
        positions=positions[None]
    if getattr(model,"dim",None)!=system.dim or not getattr(model.protection,"compact_support",False):
        raise IngressError("入口要求同维、具有固定紧支撑保护域的 Transport")
    try:
        model.validate_parameters()
    except (ValueError,RuntimeError,FloatingPointError) as exc:
        raise IngressError("输运参数或固定网格不满足已认证定义，整批失败") from exc
    if check_budget is not None:
        check_budget()
    try:
        source_regions=_validate(raw,positions)
        source_u0=_physical_u(raw,positions,0)
        source_u1=_physical_u(raw,positions,1)
        rotations,poles=_frame_and_poles(raw,positions)
    except (ValueError,RuntimeError,FloatingPointError) as exc:
        if isinstance(exc,IngressError):
            raise
        raise IngressError("原始物理构型核对失败，整批失败") from exc
    target=positions.clone()
    loga=source_u0-source_u1
    mass_logdet=loga.new_zeros(len(positions))
    ordinary_logdet=mass_logdet.clone()
    inverse_residual=mass_logdet.clone()
    identity=torch.ones(len(positions),dtype=torch.bool,device=positions.device)
    reasons=["sphere_chart_south_pole" if bool(pole) else "zero_field" for pole in poles]
    regular=torch.nonzero(~poles).flatten()
    if len(regular):
        try:
            encoded=system.encode(positions[regular])
            b=model.protection(encoded)
            if b.shape!=(len(regular),) or not torch.isfinite(b).all() or ((b<0)|(b>1)).any():
                raise IngressError("保护因子无效，不能认证恒等子集")
            active_mask=b>0
            active=regular[active_mask]
            if len(active):
                shape=encoded[active_mask]
                if check_budget is not None:
                    check_budget()
                moved,ordinary=model(shape,check_budget=check_budget)
                restored,inverse_ld=model.inverse(moved)
                residual=(restored-shape).abs().amax(-1)
                relative_tolerance=10*float(model.inverse_tolerance)*(1+shape.abs().amax(-1))
                if not torch.isfinite(restored).all() or not torch.isfinite(inverse_ld).all() or (residual>relative_tolerance).any():
                    raise IngressError("完整实际逆图未恢复源构型，整批失败")
                com=(positions[active]*raw.chart.masses[:,None]).sum(-2)/raw.chart.masses.sum()
                moved_positions=system.positions(moved)@rotations[active]+com[:,None]
                _validate(raw,moved_positions)
                target_u1=_physical_u(raw,moved_positions,1)
                volume_ratio=system.log_volume(moved)-system.log_volume(shape)
                actual_logdet=ordinary+volume_ratio
                if not torch.isfinite(actual_logdet).all():
                    raise IngressError("活跃子集的实际质量 Jacobian 非有限")
                target[active]=moved_positions
                loga[active]=source_u0[active]-target_u1+actual_logdet
                ordinary_logdet[active]=ordinary
                mass_logdet[active]=actual_logdet
                inverse_residual[active]=residual
                identity[active]=False
                for index in active.tolist():
                    reasons[index]=None
        except (ValueError,RuntimeError,FloatingPointError) as exc:
            if isinstance(exc,IngressError):
                raise
            raise IngressError("常规形状图、实际变换或逆解失败，整批失败") from exc
    if not torch.isfinite(loga).all():
        raise IngressError("整批权重非有限")
    if check_budget is not None:
        check_budget()
    return dict(positions=target,log_weights=loga,log_mass_jacobian=mass_logdet,
                ordinary_logdet=ordinary_logdet,identity=identity,identity_reason=reasons,
                inverse_residual=inverse_residual,rotations=rotations,
                source_regions=source_regions,target_regions=_validate(raw,target),
                scope="full_coordinate_transport")
