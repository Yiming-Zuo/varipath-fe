from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t

from .evaluation import (_assert_reveal,_data_units,_interval,_load,_normalized,
                         _registry,_serialize,UNITS)
from .inference import FAMILY_ALPHA,Interval,undefined
from .records import METHODS,write_json


EXPECTED=128
METHODS_MD=("V","endpoint-MD")


def _reference(value):
    loaded=_load(value)
    if "references" in loaded:
        loaded=loaded["references"]
    return _load(loaded["ALA4"]) if "ALA4" in loaded else loaded


def _moment_intervals(observations):
    if observations.shape!=(EXPECTED,2) or not np.isfinite(observations).all():
        return [undefined("128 个完整配对部署存在缺失、失败或不合格输出") for _ in range(2)]

    critical=student_t.ppf(1-.04/(2*2),EXPECTED-1)
    means=observations.mean(0)
    half=critical*observations.std(0,ddof=1)/math.sqrt(EXPECTED)
    return [Interval(float(center-radius),float(center+radius)) for center,radius in zip(means,half)]


def endpoint_md_summary(run_root,reference,config,registry=None):

    registry=_registry(registry)
    base=dict(status="unresolved",molecule="ALA4",quantity="preference/(R*300K)",
              expected_per_method=EXPECTED,expected_deployments=2*EXPECTED,
              eligible_for_formal=False,excluded_from_formal_families=True,
              added_E3_events=0,diagnostic_coverage=.95,moment_alpha=.04,
              reference_alpha=FAMILY_ALPHA["R"],moment_interval_events=2)
    if not _assert_reveal(registry):
        return dict(base,reasons=["缺少 P/I/F 先封存与 R 后揭盲登记；未读取参考"],methods={})
    config=_load(config)
    experiment=config.get("experiment",{})
    if experiment.get("direct_md_replicas")!=EXPECTED or int(experiment.get("replicas",EXPECTED))!=EXPECTED:
        raise ValueError("端态 MD 诊断须预先声明 ALA4 的 V 与 endpoint-MD 各128次")
    ref=_reference(reference)
    if ref.get("molecule")!="ALA4" or ref.get("role")!="R" or ref.get("units")!=UNITS or not ref.get("reference_id"):
        raise ValueError("ALA4 独立参考的 ID、角色或规范化单位不明确")
    if float(ref.get("family_alpha",FAMILY_ALPHA["R"]))!=FAMILY_ALPHA["R"]:
        raise ValueError("95% 诊断要求复用现有 R 族 .01 覆盖界")
    reference_units=list(ref.get("source_unit_ids",[]))
    if not reference_units or len(set(reference_units))!=len(reference_units):
        raise ValueError("参考的真实独立单元缺失或重复")
    r_role=registry.state["roles"].get("R",{})
    if not set(reference_units).issubset(set(r_role.get("unit_ids",[]))):
        raise ValueError("参考不属于已揭盲登记的 R 集合")
    interval=_interval(ref.get("intervals",{}).get("preference"))
    center=ref.get("centers",{}).get("preference")
    reference_ready=ref.get("eligible_for_formal") is True and interval.finite and isinstance(center,(int,float)) and math.isfinite(center)
    reasons=[]
    if not reference_ready:
        reasons.append("独立 R 的偏好参考或其覆盖资格尚未确认")
        interval=undefined("独立 R 参考未确认")
    if not isinstance(center,(int,float)) or not math.isfinite(center):
        center=None
    f_role=registry.state["roles"]["F"]
    f_units=set(f_role["unit_ids"])
    sealed_paths={str(Path(path).resolve()) for path in f_role.get("output_paths",[])}
    records={};owners={};hardware=set()
    for path in sorted(Path(run_root).rglob("result.json")):
        item=_load(path)
        if not all(key in item for key in ("method","replica","molecule")):
            continue
        method,name,replica=item["method"],item["molecule"],item["replica"]
        if method not in METHODS and method!="endpoint-MD":
            raise ValueError("运行目录出现未声明方法，不能自动纳入 MD 诊断")
        if method not in METHODS_MD or (name=="ALDP" and method=="V"):
            continue
        if name!="ALA4" or type(replica) is not int or not 0<=replica<EXPECTED:
            raise ValueError("MD 诊断部署须为 ALA4 的固定128编号")
        key=(method,replica)
        if key in records:
            raise ValueError("同一配对部署有多份记录，不能择优读取")
        invalid=[]
        if str(path.resolve()) not in sealed_paths:
            invalid.append("本次实际输出文件未列入 F 封存")
        if item.get("mode")!="formal" or item.get("eligible_for_formal") is not True:
            invalid.append("实际部署没有取得冻结协议的有效输出资格")
        if item.get("status") not in ("completed","fallback") or not isinstance(item.get("deployed"),dict):
            invalid.append("没有合法完整的部署输出")
        if item.get("status")=="fallback" and item.get("fallback_used") is not True:
            invalid.append("后备状态缺少实际触发记录")
        for field,default,tolerance in (("budget_ratio",1.,1e-12),("budget_seconds",14400.,1e-8)):
            actual=item.get(field)
            if not isinstance(actual,(int,float)) or not math.isfinite(actual) or not math.isclose(actual,float(config.get(field,default)),rel_tol=0,abs_tol=tolerance):
                invalid.append("预算档或完整冷启动预算不匹配")
        execution=item.get("execution_unit")
        if not isinstance(execution,dict) or not execution:
            invalid.append("缺少实际执行单元记录")
        else:
            execution=dict(execution)
            native_platform=execution.pop("native_md_platform",None)
            if method=="endpoint-MD" and native_platform not in (None,"CPU","Reference"):
                invalid.append("原生 MD 平台未确认与共同 CPU 执行单元相同")
            hardware.add(json.dumps(execution,sort_keys=True))
        cost=item.get("cost",{})
        if not any(phase.get("name")=="paid-identity-fallback" and phase.get("status")=="complete" for phase in cost.get("phases",[])):
            invalid.append("没有实际完成付费后备的成本记录")
        elapsed=cost.get("wall_seconds")
        if not isinstance(elapsed,(int,float)) or not math.isfinite(elapsed) or elapsed<0:
            invalid.append("缺少完整实际墙钟费用")
        elif elapsed>float(config.get("budget_seconds",14400.))+1e-8:
            invalid.append("完整实际墙钟费用超过共同预算")
        by_role=_data_units(item,path.parent)
        required={"fallback"}
        if item.get("status")=="completed":
            required.add("production")
            if method=="V":
                required.update(("fit","select"))
        flattened=[]
        for role in required:
            if not by_role.get(role):
                invalid.append("缺少实际必需的 "+role+" 独立数据身份")
        for role,ids in by_role.items():
            if not isinstance(ids,list) or any(not isinstance(unit,str) or not unit for unit in ids) or len(set(ids))!=len(ids):
                invalid.append("独立采样单元记录无效或重复")
            else:
                flattened.extend(ids)
        if len(set(flattened))!=len(flattened):
            raise ValueError("同一部署的不同角色复用了实际采样单元")
        if set(reference_units).intersection(flattened):
            raise ValueError("R 参考泄漏到 V 或端态 MD 部署")
        if not flattened or not set(flattened).issubset(f_units):
            invalid.append("实际采样单元没有完整列入 F 封存")
        for unit in flattened:
            if unit in owners and owners[unit]!=key:
                raise ValueError("不同方法或独立重复复用了实际采样单元")
            owners[unit]=key
        item=dict(item,evaluation_reasons=invalid,result_path=str(path))
        records[key]=item
    if len(hardware)!=1:
        reasons.append("实际硬件或并行度不一致，等成本诊断未确认")
    vectors={};methods={};missing=[];failures=[]
    for method in METHODS_MD:
        values=np.full(EXPECTED,np.nan)
        native=[];fallbacks=floors=unknown_floor=0
        for replica in range(EXPECTED):
            item=records.get((method,replica))
            if item is None:
                missing.append(dict(method=method,replica=replica));unknown_floor+=1
                continue
            readout=item.get("deployed") or {}
            estimate=_normalized(readout,"ALA4","preference")
            if not item["evaluation_reasons"] and len(hardware)==1:
                values[replica]=estimate
            if not math.isfinite(estimate):
                item["evaluation_reasons"].append("部署偏好缺失、非有限或物理单位错误")
            if item["evaluation_reasons"]:
                failures.append(dict(method=method,replica=replica,reasons=item["evaluation_reasons"]))
            fallbacks+=int(item.get("status")=="fallback" and item.get("fallback_used") is True)
            if type(readout.get("floor_triggered")) is bool:
                floors+=int(readout["floor_triggered"])
            else:
                unknown_floor+=1
            native_value=_normalized(item.get("native"),"ALA4","preference",native=True)
            if (item.get("native") or {}).get("native_defined") is True and math.isfinite(native_value):
                native.append(native_value)
        complete=np.isfinite(values).all() and reference_ready
        mse=float(np.mean((values-center)**2)) if complete else None
        methods[method]=dict(expected=EXPECTED,valid_complete_outputs=int(np.isfinite(values).sum()),
            missing_or_invalid_count=int((~np.isfinite(values)).sum()),deployed_point_mse=mse,
            deployed_point_rmse=None if mse is None else math.sqrt(mse),
            native_conditionally_defined_count=len(native),native_undefined_count=EXPECTED-len(native),
            native_undefined_rate=(EXPECTED-len(native))/EXPECTED,
            native_conditional_rmse=float(np.sqrt(np.mean((np.asarray(native)-center)**2))) if native and reference_ready else None,
            floor_trigger_count=floors,floor_trigger_rate=floors/EXPECTED,floor_flag_unknown_count=unknown_floor,
            paid_fallback_count=fallbacks,paid_fallback_rate=fallbacks/EXPECTED,
            native_error_scope="Conditional on native-defined outputs only; no unconditional native MSE",
            rates_scope="All rates use planned denominator 128; missing floor flags remain explicit unknowns")
        vectors[method]=values
    if center is not None:
        v,m=vectors["V"]-center,vectors["endpoint-MD"]-center
        moments=_moment_intervals(np.column_stack((v*v-m*m,v-m)))
    else:
        moments=[undefined("参考中心缺失") for _ in range(2)]
    comparison=undefined("128 个配对部署或共同参考未确认")
    if reference_ready and all(moment.finite for moment in moments):
        delta=(interval.lower-center,interval.upper-center)
        product=[d*b for d in delta for b in (moments[1].lower,moments[1].upper)]
        comparison=Interval(moments[0].lower-2*max(product),moments[0].upper-2*min(product))
    mse_v,mse_md=(methods[method]["deployed_point_mse"] for method in METHODS_MD)
    ratio=mse_v/mse_md if mse_v is not None and mse_md is not None and mse_md>0 else None
    base.update(status="diagnostic_complete" if comparison.finite else "unresolved",reasons=reasons,
        observed_deployments=len(records),missing_deployments=missing,invalid_deployments=failures,methods=methods,
        paired_replica_ids=list(range(EXPECTED)),normalized_outputs={key:value.tolist() for key,value in vectors.items()},
        point_mse_difference=None if mse_v is None or mse_md is None else mse_v-mse_md,
        point_mse_ratio_V_over_MD=ratio,point_rmse_ratio_V_over_MD=None if ratio is None else math.sqrt(ratio),
        mse_difference_interval=comparison,moment_intervals=dict(centered_squared_error_difference=moments[0],output_difference=moments[1]),
        comparison_orientation="MSE(V) minus MSE(endpoint-MD)",reference_id=ref["reference_id"],
        reference_source_unit_ids=reference_units,reference_interval=interval,reference_center=center,
        reference_projection="The same independent normalized R interval enters the linear projection once; replication does not shrink it",
        coverage_scope="Two Student-t intervals share total alpha .04 by Bonferroni; the existing common R event has alpha .01. Coverage requires their scientific assumptions. This is descriptive diagnosis outside the 28 comparisons and 64 E3 moment events.")
    return _serialize(base)


def main(argv=None):
    parser=argparse.ArgumentParser(description="ALA4 V/端态 MD 的128配对诊断，不进入 E3 正式族")
    parser.add_argument("--runs",type=Path,required=True)
    parser.add_argument("--reference",type=Path,required=True)
    parser.add_argument("--registry",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--config",type=Path)
    args=parser.parse_args(argv)
    config=_load(args.config) if args.config else dict(budget_seconds=14400.,budget_ratio=1.,experiment=dict(replicas=128,direct_md_replicas=128))
    if args.output.exists():
        raise FileExistsError("MD 诊断输出已存在，不覆盖原记录")
    result=endpoint_md_summary(args.runs,args.reference,config,args.registry)
    write_json(args.output,result)
    return result


if __name__=="__main__":
    main()
