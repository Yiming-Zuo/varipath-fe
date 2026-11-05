from __future__ import annotations

from dataclasses import asdict
import json
import math
from pathlib import Path

import numpy as np
import torch

from .baselines.mbar import fit_mbar
from .inference import (Interval, RoleSeal, absolute_mse_interval, bonferroni_t,
                        combine_gates, paired_mse_interval, undefined)
from .readouts import R_KJ
from .records import METHODS


QUANTITIES=("delta_f","preference","target_a")
UNITS={"delta_f":"dimensionless","preference":"dimensionless","target_a":"probability"}


def _array(value):
    return value.detach().cpu().numpy() if isinstance(value,torch.Tensor) else np.asarray(value)


def _serialize(value):
    if isinstance(value,Interval):
        return _serialize(asdict(value))
    if isinstance(value,dict):
        return {key:_serialize(v) for key,v in value.items()}
    if isinstance(value,(tuple,list)):
        return [_serialize(v) for v in value]
    if isinstance(value,np.ndarray):
        return _serialize(value.tolist())
    if isinstance(value,(np.integer,np.floating,np.bool_)):
        return _serialize(value.item())
    if isinstance(value,float) and not math.isfinite(value):
        return None
    return value


def _registry(value):
    return value if isinstance(value,RoleSeal) else RoleSeal(value) if value is not None else None


def _load(value):
    return json.loads(Path(value).read_text()) if isinstance(value,(str,Path)) else value


def _interval(value):
    if isinstance(value,Interval):
        return value
    if not isinstance(value,dict) or value.get("lower") is None or value.get("upper") is None:
        return undefined("参考区间缺失或未定义")
    return Interval(float(value["lower"]),float(value["upper"]),value.get("status","valid"),value.get("reason",""))


def _assert_reveal(registry,reference_units=None):
    if registry is None:
        return False
    roles=registry.state["roles"]
    if any(roles.get(role,{}).get("status")!="sealed" for role in ("P","I","F")):
        raise ValueError("R揭盲要求真实P、I、F输出全部先封存")
    if any(not Path(path).is_file() for role in ("P","I","F") for path in roles[role].get("output_paths",[])):
        raise ValueError("R揭盲前的P/I/F封存产物已缺失，不能仅凭状态标记继续")
    if reference_units is not None:
        existing=roles.get("R")
        if existing is None:
            registry.open("R",list(reference_units))
        elif not set(reference_units).issubset(set(existing["unit_ids"])):
            raise ValueError("本次R单元不在预先登记的完整R集合中")
    return True


def _bias_bounds(config,bank,system,registry):

    path=config.get("reference",{}).get("numerical_bias_certificate")
    if not path:
        return None,["没有独立数值偏差认证；不能默认偏差为零"]
    path=Path(path)
    if not path.is_file():
        return None,["独立数值偏差认证文件不存在"]
    certificate=_load(path)
    reasons=[]
    if certificate.get("kind")!="reference_numerical_bias" or certificate.get("status")!="passed" or certificate.get("population_bound") is not True:
        reasons.append("数值证据不是通过的人口偏差界认证；有限点最大误差不能自动替代人口界")
    if certificate.get("molecule")!=system.name or certificate.get("units")!=UNITS:
        reasons.append("数值偏差认证的体系或单位不匹配")
    bounds=certificate.get("bounds",{})
    if set(bounds)!=set(QUANTITIES) or any(not isinstance(v,(int,float)) or not math.isfinite(v) or v<0 for v in bounds.values()):
        reasons.append("数值偏差界必须完整、有限、非负")
    ids=certificate.get("source_unit_ids",[])
    occupied={unit for entry in registry.state["roles"].values() for unit in entry["unit_ids"]}
    if not ids or len(set(ids))!=len(ids) or occupied.intersection(ids):
        reasons.append("数值偏差核验的独立单元缺失、重复或泄漏至正式数据")
    protocol=bank.get("qualification_result",{}).get("protocol",{}).get("protocol",{})
    expected={key:protocol.get(key) for key in ("kernel","step_size","leapfrog_steps")}
    expected["states"]=[float(s) for s in bank["states"]]
    if certificate.get("protocol")!=expected:
        reasons.append("数值偏差认证不覆盖本次冻结的实际采样协议")
    if not isinstance(certificate.get("derivation"),str) or not certificate["derivation"].strip():
        reasons.append("数值偏差认证缺少界的推导与适用条件")
    checks=certificate.get("checks",[])
    if not checks:
        reasons.append("数值偏差认证没有可核实的实际检查产物")
    for check in checks:
        evidence=Path(check.get("path",""))
        if not evidence.is_absolute():
            evidence=path.parent/evidence
        if not evidence.is_file() or evidence.resolve()==path.resolve():
            reasons.append("数值偏差检查产物缺失或循环自证")
            continue
        observed=_load(evidence)
        if observed.get("status") not in ("passed","PASS") or observed.get("molecule")!=system.name:
            reasons.append("独立数值检查实际未通过或体系不符")
        actual_ids=observed.get("source_unit_ids",[])
        measurements=observed.get("measured_errors",{})
        if not actual_ids or not set(actual_ids).issubset(set(ids)) or len(set(actual_ids))!=len(actual_ids):
            reasons.append("独立数值检查没有对应实际独立单元")
        if set(measurements)!=set(QUANTITIES):
            reasons.append("独立数值检查没有三项实际误差数组，不能凭PASS标记放行")
            continue
        for quantity in QUANTITIES:
            error=np.asarray(measurements[quantity],dtype=float)
            if error.shape!=(len(actual_ids),) or not np.isfinite(error).all() or quantity not in bounds or np.max(np.abs(error))>bounds[quantity]:
                reasons.append("独立数值检查的实际误差未被所给偏差界覆盖")
    return (None if reasons else bounds),reasons


def reference_summary(bank,system,config,look=1,registry=None):

    if type(look) is not int or look not in (1,2,3,4):
        raise ValueError("参考只允许冻结的四次查看")
    units=list(bank.get("source_unit_ids",[]))
    reference_id=config.get("reference",{}).get("id") or f"{system.name}:{units[0] if units else 'missing'}:look-{look}"
    report=dict(reference_id=reference_id,molecule=system.name,role="R",look=look,
                source_unit_ids=units,units=UNITS.copy(),status="unresolved",eligible_for_formal=False,
                error_family="R",family_alpha=.01,interval_events=24,reasons=[],centers={},intervals={},
                normalization="ALA4 preference divided by R*300K; ALDP signed cross-temperature log-odds; delta_f=-logK")
    registry=_registry(registry)
    if not _assert_reveal(registry):
        report["reasons"].append("没有RoleSeal，未读取R位置、交叉能量或结果")
        return report
    if bank.get("role")!="R":
        raise ValueError("物理参考必须来自独立R角色")
    if not units or len(set(units))!=len(units):
        raise ValueError("R的真实独立ladder身份缺失或重复")
    _assert_reveal(registry,units)
    points=_array(bank["positions"])
    cross=_array(bank["u_kn"])
    states=np.asarray(bank["states"],dtype=float)
    if points.ndim!=4 or points.shape[0]!=len(units) or points.shape[2]!=len(states) or points.shape[3]!=system.dim or cross.shape!=(points.shape[0],points.shape[1],len(states),len(states)):
        raise ValueError("R实际位置/交叉能量数组维数不一致")
    if not np.isfinite(points).all() or not np.isfinite(cross).all() or not np.isfinite(states).all():
        report["reasons"].append("R有非有限原始输入，整组未决且不删帧")
        return report
    if len(units)<4:
        report["reasons"].append("R至少需要四条独立完整ladder")
    for i in range(len(units)):
        if any(np.array_equal(points[i],points[j]) for j in range(i)):
            raise ValueError("R不同ID下复制了完全相同轨迹")
    qualification=bank.get("qualification_result",{})
    if bank.get("mode")!="formal" or bank.get("eligible_for_formal") is not True or qualification.get("eligible_for_formal") is not True:
        report["reasons"].append("本次R缺少完整有效的平衡、混合和核资格；ESS不授予资格")
    planned=config.get("reference",{}).get("count")
    if type(planned) is not int or planned<=0 or points.shape[1]!=planned*2**(look-1):
        report["reasons"].append("R保存长度不匹配冻结的nR、2nR、4nR、8nR查看计划")
    endpoint_indices=[]
    for endpoint in (0,1):
        matching=np.flatnonzero(np.isclose(states,float(system.state(endpoint)),rtol=0,atol=1e-10))
        if len(matching)!=1:
            raise ValueError("R窗口缺少或重复物理端点")
        endpoint_indices.append(int(matching[0]))
    rows=[];failures=[]
    for ladder in range(len(units)):
        row=[np.nan]*3
        try:
            batch_size=config.get("runtime",{}).get("production_batch_size",32)
            if type(batch_size) is not int or batch_size<1:
                raise ValueError("参考计算批量必须为正整数")
            flags=[]


            for state_index in range(len(states)):
                for start in range(0,points.shape[1],batch_size):
                    x=torch.tensor(points[ladder,start:start+batch_size,state_index],dtype=torch.float64)
                    if hasattr(system,"in_domain") and not bool(system.in_domain(x).all()):
                        raise ValueError("R存在物理支持域失败")
                    regions=system.regions(x)
                    if "valid" in regions and not bool(regions["valid"].all()):
                        raise ValueError("R存在不可判定的硬区域")
                    flags.append(torch.stack([regions["a"],regions["b"]],-1))
            flags=torch.cat(flags)
            actual=cross[ladder].transpose(1,2,0).reshape(len(states),-1)
            result=fit_mbar(actual,[points.shape[1]]*len(states),endpoint_u=actual[endpoint_indices],
                            region_flags=flags,min_overlap=config.get("mbar",{}).get("min_overlap",1e-8))
            logs=np.asarray(result["endpoint_log_probabilities"])
            row[0]=result["endpoint_delta_f"]
            row[2]=float(np.exp(logs[1,0]))
            if np.isfinite(logs).all():
                odds=(logs[1,1]-logs[1,0])-(logs[0,1]-logs[0,0])
                row[1]=float(-odds if system.name=="ALA4" else odds)
            else:
                failures.append({"unit_id":units[ladder],"reason":"R原生区域零命中，物理参考不使用地板"})
        except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError) as exc:
            failures.append({"unit_id":units[ladder],"reason":str(exc)})
        rows.append(row)
    observations=np.asarray(rows)
    diagnostic=bonferroni_t(observations,"R",independent_unit_ids=units)
    bounds,bias_reasons=_bias_bounds(config,bank,system,registry)
    report["reasons"].extend(bias_reasons)
    if failures:
        report["reasons"].append("至少一个完整R单元有失败或原生零命中；该单元保留")
    report.update(observations=observations.tolist(),ladder_failures=failures,
                  diagnostic_intervals={key:interval for key,interval in zip(QUANTITIES,diagnostic)},
                  numerical_bias_bounds=bounds,qualification=qualification)
    for index,key in enumerate(QUANTITIES):
        report["centers"][key]=float(observations[:,index].mean()) if np.isfinite(observations[:,index]).all() else None
        report["intervals"][key]=diagnostic[index].widen(float(bounds[key])) if bounds is not None and not report["reasons"] else undefined("参考资格、偏差界或原生读出未确认")
    eligible=not report["reasons"] and all(interval.finite for interval in report["intervals"].values())
    report["eligible_for_formal"]=eligible
    report["status"]="ready" if eligible else "unresolved"
    precision=float(config.get("experiment",{}).get("reference_half_width",.01))
    report["precision_gates"]={key:("supported" if interval.finite and (interval.upper-interval.lower)/2<=precision else "unresolved") for key,interval in report["intervals"].items() if key!="target_a"}
    return _serialize(report)


def _normalized(readout,molecule,key,native=False):
    if not isinstance(readout,dict):
        return np.nan
    expected="kJ/mol" if molecule=="ALA4" else "dimensionless"
    if readout.get("units")!=expected:
        return np.nan
    if key=="delta_f":
        value=readout.get("delta_f")
    else:
        value=readout.get("native_preference" if native else "preference")
        if value is not None and molecule=="ALA4":
            value=value/(R_KJ*300)
    return float(value) if isinstance(value,(int,float)) and math.isfinite(value) else np.nan


def _data_units(record,directory):
    supplied=record.get("data_unit_ids")
    if supplied is not None:
        if not isinstance(supplied,dict):
            raise ValueError("部署独立单元记录必须按真实数据角色给出")
        return supplied
    result={}
    for role in ("fallback","fit","select","production"):
        path=directory/role/"bank.pt"
        if path.is_file():
            bank=torch.load(path,map_location="cpu",weights_only=False)
            result[role]=list(bank.get("source_unit_ids",[]))
    return result


def aggregate_results(run_root,references,config,registry=None):

    root=Path(run_root)
    registry=_registry(registry)
    if not _assert_reveal(registry):
        return dict(status="unresolved",reasons=["缺少P/I/F先封存与R后揭盲登记，未读取参考"],comparisons={},absolute_precision={},expected_deployments=2048)
    references=_load(references)
    refs={name:_load(references.get(name,{})) for name in ("ALA4","ALDP")}
    rids=set();reference_units=set()
    for name,ref in refs.items():
        if ref.get("molecule")!=name or ref.get("role")!="R" or ref.get("units")!=UNITS or not ref.get("reference_id"):
            raise ValueError("独立物理参考的ID、角色、体系或规范化单位不明确")
        if ref["reference_id"] in rids:
            raise ValueError("两个体系不能复用同一个物理参考ID")
        rids.add(ref["reference_id"])
        ids=set(ref.get("source_unit_ids",[]))
        if not ids or reference_units.intersection(ids):
            raise ValueError("参考独立单元缺失或跨体系复用")
        reference_units.update(ids)
    registered=set(registry.state["roles"].get("R",{}).get("unit_ids",[]))
    if not reference_units.issubset(registered):
        raise ValueError("参考独立单元不在已揭盲的R登记中")
    experiment=config.get("experiment",{})
    if int(experiment.get("replicas",128))!=128 or tuple(experiment.get("methods",METHODS))!=METHODS:
        raise ValueError("正式E3固定每体系八法各128次完整部署")
    expected=128
    records={};global_reasons=[];seen_units={};hardware=set();md_diagnostics=0
    for path in sorted(root.rglob("result.json")):
        item=_load(path)
        if not all(key in item for key in ("method","replica","molecule")):
            continue
        name,method,replica=item["molecule"],item["method"],item["replica"]
        if (name=="ALA4" and method=="endpoint-MD" and experiment.get("direct_md_replicas")==128
                and type(replica) is int and 0<=replica<expected):
            md_diagnostics+=1
            continue
        if name not in refs or method not in METHODS or type(replica) is not int or not 0<=replica<expected:
            raise ValueError("部署记录不属于冻结的两个体系/八法/128编号")
        key=(name,method,replica)
        if key in records:
            raise ValueError("同一个完整部署编号有多份结果，不能挑选较有利的一份")
        reasons=[]
        if item.get("eligible_for_formal") is not True or item.get("mode")!="formal":
            reasons.append("该实际部署未获正式资格")
        if item.get("status") not in ("completed","fallback") or not isinstance(item.get("deployed"),dict):
            reasons.append("部署没有合法完整输出")
        if item.get("status")=="fallback" and item.get("fallback_used") is not True:
            reasons.append("后备状态缺少实际触发记录")
        if not math.isclose(float(item.get("budget_ratio",float("nan"))),float(config.get("budget_ratio",1.)),rel_tol=0,abs_tol=1e-12):
            reasons.append("预算档不匹配或未记录")
        if not math.isclose(float(item.get("budget_seconds",float("nan"))),float(config.get("budget_seconds",14400)),rel_tol=0,abs_tol=1e-8):
            reasons.append("冷启动预算不匹配或未记录")
        execution=item.get("execution_unit")
        if execution is None:
            reasons.append("缺少本次实际执行单元记录")
        else:
            hardware.add(json.dumps(execution,sort_keys=True))
        by_role=_data_units(item,path.parent)
        required_roles={"fallback"}
        if item.get("status")=="completed":
            required_roles.add("production")
            if method not in ("E-MBAR","EB-MBAR"):
                required_roles.update(("fit","select"))
        for role in required_roles:
            if not by_role.get(role):
                reasons.append(f"缺少实际必需的{role}数据身份")
        phases=item.get("cost",{}).get("phases",[])
        if not any(phase.get("name")=="paid-identity-fallback" and phase.get("status")=="complete" for phase in phases):
            reasons.append("没有实际完成付费后备的费用记录")
        flattened=[]
        for role,ids in by_role.items():
            if not isinstance(ids,list) or len(set(ids))!=len(ids):
                reasons.append("独立采样单元记录无效或重复")
                continue
            if not ids:
                if role in required_roles:
                    reasons.append("必需的独立采样单元缺失")
                continue
            flattened.extend(ids)
        if not flattened:
            reasons.append("没有本次真实采样单元身份")
        if len(set(flattened))!=len(flattened):
            raise ValueError("同一部署的不同数据角色复用了完整ladder")
        if reference_units.intersection(flattened):
            raise ValueError("独立R参考泄漏至方法部署")
        for unit in flattened:
            owner=(name,replica)
            if unit in seen_units and seen_units[unit]!=owner:
                raise ValueError("不同独立重复复用了实际采样ladder")
            seen_units[unit]=owner
        item["evaluation_reasons"]=reasons
        item["result_path"]=str(path)
        records[key]=item
    if len(hardware)!=1:
        global_reasons.append("执行单元缺失或硬件/并行度不一致，等成本正式比较未确认")
    f_units=set(registry.state["roles"]["F"]["unit_ids"])
    if not set(seen_units).issubset(f_units):
        global_reasons.append("部署实际采样单元未全部列入冻结的F登记")
    comparisons={};absolute={};diagnostics={};missing=[]
    paired_ids=[f"deployment-{i}" for i in range(expected)]
    for name in ("ALA4","ALDP"):
        ref=refs[name]
        for key in ("delta_f","preference"):
            interval=_interval(ref.get("intervals",{}).get(key))
            center=ref.get("centers",{}).get(key)
            reference_ready=ref.get("eligible_for_formal") is True and interval.finite and isinstance(center,(int,float)) and math.isfinite(center)
            if not reference_ready:
                interval=undefined("独立参考未确认")
                center=0. if not isinstance(center,(int,float)) or not math.isfinite(center) else center
            vectors={};method_diagnostics={}
            for method in METHODS:
                values=np.full(expected,np.nan);raw=np.full(expected,np.nan)
                native=[];fallbacks=0;floors=0;invalid=0
                for replica in range(expected):
                    item=records.get((name,method,replica))
                    if item is None:
                        if key=="delta_f":missing.append({"molecule":name,"method":method,"replica":replica})
                        invalid+=1
                        continue
                    raw[replica]=_normalized(item.get("deployed"),name,key)
                    if not item["evaluation_reasons"] and not global_reasons:
                        values[replica]=raw[replica]
                    if not math.isfinite(values[replica]):invalid+=1
                    fallbacks+=item.get("status")=="fallback"
                    floors+=bool((item.get("deployed") or {}).get("floor_triggered",False))
                    native_value=_normalized(item.get("native"),name,key,native=key=="preference")
                    if math.isfinite(native_value):native.append(native_value)
                vectors[method]=values
                method_diagnostics[method]=dict(expected=expected,legal_formal_outputs=int(np.isfinite(values).sum()),
                    missing_or_invalid=invalid,paid_fallback_count=fallbacks,floor_trigger_count=floors,
                    deployed_point_mse=float(np.mean((raw-center)**2)) if np.isfinite(raw).all() and reference_ready else None,
                    native_conditionally_defined_count=len(native),
                    native_conditional_rmse=float(np.sqrt(np.mean((np.asarray(native)-center)**2))) if native and reference_ready else None,
                    native_error_scope="conditional on native-defined output; no unconditional native MSE is inferred")
            for method in METHODS[1:]:
                comparisons[f"{name}/{key}/V-vs-{method}"]=paired_mse_interval(vectors["V"],vectors[method],interval,reference_center=center,independent_unit_ids=paired_ids)
            absolute[f"{name}/{key}/V"]=absolute_mse_interval(vectors["V"],interval,reference_center=center,independent_unit_ids=paired_ids)
            diagnostics[f"{name}/{key}"]=method_diagnostics
    statuses=[result["status"] for result in comparisons.values()]+[result["status"] for result in absolute.values()]
    overall=combine_gates(statuses)
    result=dict(status=overall,reasons=global_reasons,comparisons=comparisons,absolute_precision=absolute,
                deployment_diagnostics=diagnostics,expected_deployments=2048,observed_deployments=len(records),missing_deployments=missing,
                comparison_count=28,absolute_precision_count=4,moment_interval_events=64,alpha=.02,
                direct_md_diagnostic_records=md_diagnostics,
                reference_ids={name:ref["reference_id"] for name,ref in refs.items()},
                reference_precision_gates={name:ref.get("precision_gates",{}) for name,ref in refs.items()},
                reference_projection="same independent physical reference interval enters each linear/quadratic projection once; it is not reduced by deployment replication",
                dataset_scope="Every fixed deployment position retained; failure/missing/nonformal records yield unresolved intervals, never success-only formal RMSE")
    return _serialize(result)
