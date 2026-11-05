from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.stats import beta, norm, t

from .inference import (DELTA_LOG_RISK, DELTA_MASS, Interval, absolute_mse_interval,
    bonferroni_t, combine_gates, equivalence_gate, intervention_interval_names,
    negative_gate, paired_mse_interval, panel_agreement, panel_manifest,
    positive_gate, prediction_gate, repair_intervals, select_same_side_pair,
    shape_claim_gate)
from .records import METHODS, read_json, run_lock, save_checkpoint, write_json


SYSTEMS = ("ALA4", "ALDP")
SCOPE = "inference_simulation"


def _prediction_names():
    fields = ("mass", "mass_b", "mass_remainder", "log_risk", "conditional_h2")
    names = [cell["id"]+"/"+field for cell in panel_manifest() for field in fields]
    return names+[cell["id"]+"/geometric_mass" for cell in panel_manifest() if cell["panel"] == "main"]


def _noise(rng, shape, correlation, distribution, degrees):
    if not 0 <= correlation < 1:
        raise ValueError("统计向量的相关系数须在 [0,1) 内")
    result = math.sqrt(correlation)*rng.normal(size=(*shape[:-1], 1))+math.sqrt(1-correlation)*rng.normal(size=shape)
    if distribution == "student_t":
        if degrees <= 2:
            raise ValueError("给定误差标准差的 Student-t 模型需要自由度大于二")
        result *= np.sqrt((degrees-2)/rng.chisquare(degrees, size=(*shape[:-1], 1)))
    elif distribution != "normal":
        raise ValueError("仅实现显式 normal 或 student_t 统计规划分布")
    return result


def _clipped_mean(center, scale, lower, upper, distribution, degrees):

    if scale == 0:
        return float(np.clip(center, lower, upper))
    width = scale if distribution == "normal" else scale*math.sqrt((degrees-2)/degrees)
    a, b = (lower-center)/width, (upper-center)/width
    if distribution == "normal":
        fa, fb = norm.cdf(a), norm.cdf(b)
        first = norm.pdf(a)-norm.pdf(b)
    else:
        fa, fb = t.cdf(a, degrees), t.cdf(b, degrees)
        first = ((degrees+a*a)*t.pdf(a, degrees)-(degrees+b*b)*t.pdf(b, degrees))/(degrees-1)
    return float(lower*fa+upper*(1-fb)+center*(fb-fa)+width*first)


def _case(config, case):
    result = copy.deepcopy(config["model"])
    result.update({key: value for key, value in case.items() if key not in ("name", "description")})
    for key in ("theta",):
        if np.asarray(result[key]).shape != (2, 2):
            raise ValueError("theta 须按 ALA4、ALDP 各两个规范化读出给定")
    for key in ("target_a", "baseline_a"):
        if np.asarray(result[key]).shape != (2,) or not np.all((np.asarray(result[key]) > 0)&(np.asarray(result[key]) < 1)):
            raise ValueError("两体系布居参数须严格在零与一之间")
    if len(result["method_standard_deviations"]) != 8 or len(result["method_biases"]) != 8:
        raise ValueError("F 情景必须包含全部八法")
    if min(result["method_standard_deviations"]) <= 0 or min(result["prediction_sd"], result["intervention_sd"], result["reference_frame_sd"]) <= 0:
        raise ValueError("生成模型的标准差须为正")
    for key in ("failure_probability", "P_failure_probability", "I_failure_probability", "R_failure_probability"):
        if not 0 <= result[key] <= 1:
            raise ValueError("失效率必须在零与一之间")
    return result


def load_config(value):
    config = read_json(value) if not isinstance(value, dict) else copy.deepcopy(value)
    if config.get("scope") != SCOPE or config.get("seed", -1) < 0:
        raise ValueError("统计模拟需要 inference_simulation 类型及非负种子")
    if not config.get("generating_assumptions") or not config.get("cases"):
        raise ValueError("须显式声明生成假设及预定情景")
    if type(config.get("simulations")) is not int or config["simulations"] < 1:
        raise ValueError("情景重复数必须为正整数")
    if config.get("deployments", 128) != 128 or config.get("inference_blocks", 16) != 16 or config.get("reference_ladders", 4) != 4:
        raise ValueError("规划保留128部署、16独立块及每体系4条参考ladder")
    if type(config.get("reference_initial_draws")) is not int or config["reference_initial_draws"] < 1:
        raise ValueError("四次嵌套参考查看需要正的初始长度")
    names = [case["name"] for case in config["cases"]]
    if len(set(names)) != len(names):
        raise ValueError("情景名称重复")
    for case in config["cases"]:
        _case(config, case)
    return config


def generate_scenario(config, case, role_seeds):

    parameters = _case(config, case)
    rng_p, rng_i, rng_f, rng_r = [np.random.default_rng(seed) for seed in role_seeds]
    distribution, degrees = parameters["distribution"], parameters["degrees_of_freedom"]
    pnames, inames = _prediction_names(), intervention_interval_names()
    pindex, iindex = {name: i for i, name in enumerate(pnames)}, {name: i for i, name in enumerate(inames)}
    pmean, imean = np.zeros(552), np.zeros(418)
    for cell in panel_manifest():
        prefix, offset = cell["id"], cell["offset"]
        mass = (0. if offset == 0 else (1 if offset > 0 else -1)*parameters["mass_effect"]*(.75 if offset == .5 else 1.))
        risk = 0. if offset == 0 else parameters["risk_effect"]*(-1 if offset == 1 else 1)
        for field, value in (("mass", mass), ("mass_b", -mass), ("mass_remainder", 0.),
                             ("log_risk", risk), ("conditional_h2", parameters["shape_effect"] if offset == 0 else 0.)):
            pmean[pindex[prefix+"/"+field]] = value
        if cell["panel"] == "main":
            pmean[pindex[prefix+"/geometric_mass"]] = mass+parameters["geometry_bias"]
        for field, value in (("mass", mass+parameters["intervention_mass_bias"]), ("log_risk", risk+parameters["intervention_risk_bias"])):
            imean[iindex[prefix+"/"+field+"/full"]] = value
            imean[iindex[prefix+"/"+field+"/full_minus_twice_half"]] = parameters["half_dose_residual"]
    for system_index, system in enumerate(SYSTEMS):
        imean[iindex[system+"/baseline_a"]] = parameters["baseline_a"][system_index]
        for shape in range(3):
            imean[iindex[f"{system}/shape{shape}/h2"]] = parameters["shape_effect"]
    p = pmean+parameters["prediction_sd"]*_noise(rng_p, (16, 552), parameters["correlation"], distribution, degrees)
    i = imean+parameters["intervention_sd"]*_noise(rng_i, (16, 418), parameters["correlation"], distribution, degrees)
    for name, index in iindex.items():
        bounds = (0., 1.) if name.endswith("/baseline_a") else ((-1., 1.) if "/mass/" in name or "/mass_" in name else None)
        if bounds is not None:
            i[:, index] = np.clip(i[:, index], *bounds)
            imean[index] = _clipped_mean(imean[index], parameters["intervention_sd"], *bounds, distribution, degrees)
    theta = np.asarray(parameters["theta"], dtype=float)
    standard = np.asarray(parameters["method_standard_deviations"])
    bias = np.asarray(parameters["method_biases"])
    f_noise = _noise(rng_f, (128, 2, 16), parameters["correlation"], distribution, degrees).reshape(128,2,8,2)
    f = theta[None, :, None, :]+bias[None, None, :, None]+standard[None, None, :, None]*f_noise
    f[rng_f.random((128, 2, 8)) < parameters["failure_probability"]] = np.nan
    p[rng_p.random(16) < parameters["P_failure_probability"]] = np.nan
    i[rng_i.random(16) < parameters["I_failure_probability"]] = np.nan

    initial = config["reference_initial_draws"]
    lengths = [initial*2**look for look in range(4)]
    reference = np.empty((4, 4, 6))
    for system_index in range(2):
        values = theta[system_index]+parameters["reference_frame_sd"]*_noise(
            rng_r, (4, lengths[-1], 2), parameters["reference_correlation"], distribution, degrees)
        counts = np.zeros(4)
        previous = 0
        failed = rng_r.random(4) < parameters["R_failure_probability"]
        for look, length in enumerate(lengths):
            counts += rng_r.binomial(length-previous, parameters["target_a"][system_index], size=4)
            reference[look, :, 3*system_index:3*system_index+2] = values[:, :length].mean(1)
            reference[look, :, 3*system_index+2] = counts/length
            reference[look, failed, 3*system_index:3*system_index+3] = np.nan
            previous = length
    reference_true = np.array([value for s in range(2) for value in (*theta[s], parameters["target_a"][s])])
    return dict(parameters=parameters, P=p, I=i, F=f, R=reference,
                P_true=pmean, I_true=imean, R_true=reference_true,
                P_names=pnames, I_names=inames, role_seeds=role_seeds)


def _covered(intervals, truth):
    return all(interval.finite and interval.lower <= value <= interval.upper for interval, value in zip(intervals, truth))


def _boolean_interval(value):
    return Interval(float(value), float(value))


def _shape_candidate(predictions,system,window):
    candidates = [cell for cell in panel_manifest() if cell["system"] == system and cell["window"] == window and cell["panel"] == "main"]
    qualifying = [cell for cell in candidates if all(equivalence_gate(predictions[cell["id"]+"/"+field], DELTA_MASS)=="supported" for field in ("mass","mass_b","mass_remainder")) and positive_gate(predictions[cell["id"]+"/conditional_h2"],.01)=="supported"]
    return sorted(qualifying,key=lambda cell:(-predictions[cell["id"]+"/conditional_h2"].lower,cell["anchor"],cell["offset"]))[0] if qualifying else None


def _selected_shape_rows(draw,predictions):
    indices={name:index for index,name in enumerate(draw["I_names"])}
    for system in SYSTEMS:
        for window in range(3):
            selected=_shape_candidate(predictions,system,window)
            prefix=f"{system}/shape{window}/"
            if selected is None:
                for field in ("h2","h2_full_minus_four_half","mass_a","mass_b","mass_remainder"):
                    draw["I"][:,indices[prefix+field]]=np.nan
                continue
            actual=draw["I"][:,indices[selected["id"]+"/mass/full"]]
            target=draw["I_true"][indices[selected["id"]+"/mass/full"]]
            for field,factor in (("mass_a",1.),("mass_b",-.5),("mass_remainder",-.5)):
                draw["I"][:,indices[prefix+field]]=factor*actual
                draw["I_true"][indices[prefix+field]]=factor*target


def _geometry_rows(draw, pintervals):
    pi = {name: index for index, name in enumerate(draw["P_names"])}
    ii = {name: index for index, name in enumerate(draw["I_names"])}
    errors = {}
    for system in SYSTEMS:
        observations, truth, uncertainty = [], [], []
        for cell in panel_manifest():
            if cell["system"] != system or cell["panel"] != "main":
                continue
            group = [row for row in panel_manifest() if row["system"] == system and row["anchor"] == cell["anchor"] and row["window"] == cell["window"] and row["panel"] == "main"]
            g = pintervals[pi[cell["id"]+"/geometric_mass"]]
            bs = [pintervals[pi[row["id"]+"/mass"]] for row in group]
            mt = draw["I_true"][ii[cell["id"]+"/mass/full"]]
            gt = draw["P_true"][pi[cell["id"]+"/geometric_mass"]]
            bt = np.mean([draw["P_true"][pi[row["id"]+"/mass"]] for row in group])
            truth.append((mt-gt)**2-(mt-bt)**2)
            if not g.finite or not all(b.finite for b in bs):
                observations.append(np.full(16, np.nan));uncertainty.append(math.inf)
                continue
            gp, bp = (g.lower+g.upper)/2, sum((b.lower+b.upper)/2 for b in bs)/3
            dg, db = (g.upper-g.lower)/2, sum((b.upper-b.lower)/2 for b in bs)/3
            actual = draw["I"][:, ii[cell["id"]+"/mass/full"]]
            observations.append((actual-gp)**2-(actual-bp)**2)
            uncertainty.append(2*(dg+db)+dg*(2*abs(gp)+dg)+db*(2*abs(bp)+db))
        key = system+"/geometric_squared_error_difference"
        draw["I"][:, ii[key]] = np.asarray(observations).mean(0)
        draw["I_true"][ii[key]] = np.mean(truth)
        errors[key] = float(np.mean(uncertainty))
    return errors


def _e1_gates(p, i, reference, parameters):
    measured = {cell["id"]+"/"+readout: i[cell["id"]+"/"+readout+"/full"] for cell in panel_manifest() for readout in ("mass", "log_risk")}
    half = {cell["id"]+"/"+readout: i[cell["id"]+"/"+readout+"/full_minus_twice_half"] for cell in panel_manifest() for readout in ("mass", "log_risk")}
    agreement = panel_agreement(p, measured, half, numerical_error_bounds={name:0. for name in measured})
    gates = {"panel_main":agreement["main"]["status"], "panel_same_side":agreement["same_side"]["status"]}
    for system_index, system in enumerate(SYSTEMS):
        primary = [next(cell for cell in panel_manifest() if cell["system"]==system and cell["anchor"]==0 and cell["window"]==0 and cell["offset"]==offset) for offset in (-1.,1.)]
        primary_status=[]
        for cell in primary:
            key=cell["id"]+"/mass"
            primary_status += [negative_gate(p[key],-DELTA_MASS),negative_gate(measured[key],-DELTA_MASS)] if cell["offset"]<0 else [positive_gate(p[key],DELTA_MASS),positive_gate(measured[key],DELTA_MASS)]
        gates[system+"/primary_reversal"]=combine_gates(primary_status)
        harmful = next(cell for cell in panel_manifest() if cell["system"]==system and cell["anchor"]==0 and cell["window"]==0 and cell["offset"]==.5)
        key=harmful["id"]+"/log_risk"
        gates[system+"/harmful_risk"]=combine_gates([positive_gate(p[key],DELTA_LOG_RISK),positive_gate(measured[key],DELTA_LOG_RISK),prediction_gate(p[key],measured[key],DELTA_LOG_RISK)["status"]])
        candidates = []
        for anchor in range(4):
            for window in range(3):
                cells = [cell for cell in panel_manifest() if cell["system"] == system and cell["anchor"] == anchor and cell["window"] == window and cell["offset"] in (1., .5)]
                cells.sort(key=lambda cell: -cell["offset"])
                candidates.append(dict(data_role="P",anchor=anchor,window=window,cell_ids=[cell["id"] for cell in cells],
                    mass=[p[cell["id"]+"/mass"] for cell in cells],log_risk=[p[cell["id"]+"/log_risk"] for cell in cells]))
        deficit = parameters["target_a"][system_index]-parameters["baseline_a"][system_index]
        selected = select_same_side_pair(candidates, deficit) if deficit > 0 else None
        statuses = []
        if selected is not None:
            for route, identifier in enumerate(selected["cell_ids"]):
                mass = measured[identifier+"/mass"]
                repair = repair_intervals(reference[3*system_index+2], i[system+"/baseline_a"], mass)
                statuses += [positive_gate(repair["deficit"]),positive_gate(repair["gain"]),positive_gate(mass, DELTA_MASS)]
                risk = measured[identifier+"/log_risk"]
                statuses.append(negative_gate(risk, -DELTA_LOG_RISK) if route == selected["negative_risk_route"] else positive_gate(risk, DELTA_LOG_RISK))
                statuses += [prediction_gate(p[identifier+"/"+field], measured[identifier+"/"+field], delta)["status"] for field,delta in (("mass",DELTA_MASS),("log_risk",DELTA_LOG_RISK))]
                statuses += [equivalence_gate(half[identifier+"/"+field], delta/2) for field,delta in (("mass",DELTA_MASS),("log_risk",DELTA_LOG_RISK))]
        gates[system+"/repair_opposite_risk"] = combine_gates(statuses) if statuses else "unresolved"
        for shape in range(3):
            prefix=f"{system}/shape{shape}/"


            selected_shape = _shape_candidate(p,system,shape)
            if selected_shape is not None:
                gates[prefix+"claim"] = shape_claim_gate([i[prefix+"mass_"+region] for region in ("a","b","remainder")],i[prefix+"h2"],p[selected_shape["id"]+"/conditional_h2"],i[prefix+"h2_full_minus_four_half"])["status"]
            else:
                gates[prefix+"claim"] = "unresolved"
        gates[system+"/geometry_advantage"] = negative_gate(i[system+"/geometric_squared_error_difference"])
    return gates


def simulate_once(config, case, role_seeds):
    draw = generate_scenario(config,case,role_seeds)
    parameters=draw["parameters"]
    pintervals=bonferroni_t(draw["P"],"P",events=552)
    _selected_shape_rows(draw,dict(zip(draw["P_names"],pintervals)))
    geometry_error=_geometry_rows(draw,pintervals)
    iintervals=bonferroni_t(draw["I"],"I")
    for name,error in geometry_error.items():
        index=draw["I_names"].index(name)
        if math.isfinite(error):iintervals[index]=iintervals[index].widen(error)
    references=[bonferroni_t(values,"R") for values in draw["R"]]
    precision=config.get("reference_half_width",.01)
    selected_look=3
    for look,intervals in enumerate(references):
        if all(intervals[index].finite and (intervals[index].upper-intervals[index].lower)/2<=precision for index in (0,1,3,4)):
            selected_look=look;break
    active=references[selected_look]
    p=dict(zip(draw["P_names"],pintervals));i=dict(zip(draw["I_names"],iintervals))
    gates=_e1_gates(p,i,active,parameters)
    truth_gates=_e1_gates(dict(zip(draw["P_names"],map(_boolean_interval,draw["P_true"]))),
                         dict(zip(draw["I_names"],map(_boolean_interval,draw["I_true"]))),
                         list(map(_boolean_interval,draw["R_true"])),parameters)
    mse=np.asarray(parameters["method_standard_deviations"])**2+np.asarray(parameters["method_biases"])**2
    e3_intervals=[];e3_truth=[];reference_uses=[]
    for system_index,system in enumerate(SYSTEMS):
        for readout in range(2):
            refindex=3*system_index+readout
            ref=active[refindex]
            center=(ref.lower+ref.upper)/2 if ref.finite else 0.
            for baseline in range(1,8):
                report=paired_mse_interval(draw["F"][:,system_index,0,readout],draw["F"][:,system_index,baseline,readout],ref,reference_center=center)
                key=f"{system}/readout{readout}/V-{METHODS[baseline]}"
                gates[key]=report["status"];truth_gates[key]="supported" if mse[0]<mse[baseline] else "unsupported"
                e3_intervals.append(report["interval"]);e3_truth.append(mse[0]-mse[baseline]);reference_uses.append(f"R/{system}/readout{readout}/look{selected_look+1}")
            report=absolute_mse_interval(draw["F"][:,system_index,0,readout],ref,reference_center=center)
            key=f"{system}/readout{readout}/absolute"
            gates[key]=report["status"];truth_gates[key]="supported" if mse[0]<=.01 else "unsupported"
            e3_intervals.append(report["interval"]);e3_truth.append(mse[0]);reference_uses.append(f"R/{system}/readout{readout}/look{selected_look+1}")
    gate_count=len(gates)
    simultaneous=dict(P=_covered(pintervals,draw["P_true"]),I=_covered(iintervals,draw["I_true"]),
        R=all(_covered(intervals,draw["R_true"]) for intervals in references),E3=_covered(e3_intervals,e3_truth))
    alternative=all(value=="supported" for value in truth_gates.values())
    accepted=all(value=="supported" for value in gates.values())
    false_accept=any(value=="supported" and truth_gates[key]!="supported" for key,value in gates.items())
    conditions=parameters["distribution"]=="normal" or parameters["degrees_of_freedom"]>4
    return dict(simultaneous_coverage=all(simultaneous.values()),family_coverage=simultaneous,
        false_acceptance=false_accept,joint_success=accepted,alternative=alternative,
        all_gates_defined=all(value!="unresolved" for value in gates.values()),
        gate_statuses=gates,gate_truth=truth_gates,gate_count=gate_count,
        selected_reference_look=selected_look+1,reference_vectors_generated=1,
        reference_applications=len(reference_uses),reference_ids_used=reference_uses,
        reference_intervals=[[(v.lower,v.upper) for v in row] for row in references],
        reference_truth=draw["R_true"].tolist(),mse_truth=mse.tolist(),
        marginal_conditions_confirmed_for_generating_model=conditions,
        dimensions=dict(P=[16,552],I=[16,418],F=[128,2,8,2],R=[4,4,6]))


def _binomial(successes,total):
    lower=0. if successes==0 else float(beta.ppf(.05,successes,total-successes+1))
    upper=1. if successes==total else float(beta.ppf(.95,successes+1,total-successes))
    return dict(successes=successes,trials=total,estimate=successes/total if total else None,
                one_sided_95_lower=lower,one_sided_95_upper=upper)


def _summarize_case(state,config):
    count=state["completed"]
    report={key:_binomial(state[key],count) for key in ("simultaneous_coverage","false_acceptance","joint_success","all_gates_defined")}
    report["family_coverage"]={key:_binomial(value,count) for key,value in state["family_coverage"].items()}
    report["alternative"]=state["alternative"]
    report["conditions_confirmed_for_generating_model"]=state["conditions"]
    power=report["joint_success"]
    sufficient=count>=1000 and state["conditions"]
    report["joint_power_gate"]=("supported" if state["alternative"] and sufficient and power["one_sided_95_lower"]>=.8 else
                                ("unsupported" if state["alternative"] and sufficient and power["one_sided_95_upper"]<.8 else "unresolved"))
    report["coverage_gate"]=("supported" if sufficient and report["simultaneous_coverage"]["one_sided_95_lower"]>=.95 else "unresolved")
    report["false_acceptance_gate"]=("supported" if sufficient and report["false_acceptance"]["one_sided_95_upper"]<=.05 else "unresolved")
    report["planned_simulations"]=config["simulations"]
    return report


def planning_workload(config):
    config=load_config(config)
    simulations=len(config["cases"])*config["simulations"]
    return dict(scope=SCOPE,independent_simulations=simulations,cases=len(config["cases"]),
        per_simulation=dict(P_intervals=552,I_intervals=418,E3_moment_intervals=64,R_look_intervals=24,
            complete_F_outputs=4096,R_ladders_per_system=4,R_looks=4),
        total_scalar_interval_events=simulations*(552+418+64+24),
        cost="逐情景执行实际联合区间与门；不运行物理训练。墙钟取决于本机统计吞吐，checkpoint保留随机流和计数，不能以少量情景替代配置的全部重复。")


def run_calibration(config,output,resume=False,stop_after=None):
    config=load_config(config);output=Path(output)
    with run_lock(output):
        path=output/"checkpoint.pt"
        if resume:
            if not path.exists():raise FileNotFoundError("没有统计规划检查点")
            import torch
            saved=torch.load(path,weights_only=False)
            if saved["config"]!=config:raise ValueError("恢复不能改变预定生成模型或重复数")
            state=saved["state"];rng=np.random.default_rng();rng.bit_generator.state=saved["rng"]
        else:
            if path.exists() or (output/"result.json").exists():raise FileExistsError("统计规划输出已存在，请显式恢复")
            rng=np.random.default_rng(config["seed"])
            state={"case_index":0,"cases":{},"total_completed":0,"elapsed_seconds":0.}
            write_json(output/"config.json",config)
        started=time.monotonic()
        while state["case_index"]<len(config["cases"]):
            case=config["cases"][state["case_index"]]
            current=state["cases"].setdefault(case["name"],dict(completed=0,simultaneous_coverage=0,false_acceptance=0,joint_success=0,
                all_gates_defined=0,family_coverage={key:0 for key in ("P","I","R","E3")},alternative=True,conditions=True))
            while current["completed"]<config["simulations"]:
                if stop_after is not None and state["total_completed"]>=stop_after:
                    state["elapsed_seconds"]+=time.monotonic()-started
                    save_checkpoint(path,dict(config=config,state=state,rng=rng.bit_generator.state))
                    return dict(status="checkpointed",scope=SCOPE,completed=state["total_completed"])
                seeds=rng.integers(0,2**63,size=(4,4),dtype=np.uint64).tolist()
                result=simulate_once(config,case,seeds)
                for key in ("simultaneous_coverage","false_acceptance","joint_success","all_gates_defined"):
                    current[key]+=int(result[key])
                for key,value in result["family_coverage"].items():current["family_coverage"][key]+=int(value)
                current["alternative"] &= result["alternative"]
                current["conditions"] &= result["marginal_conditions_confirmed_for_generating_model"]
                current["completed"]+=1;state["total_completed"]+=1
                current["last_trace"]={key:result[key] for key in ("selected_reference_look","reference_vectors_generated","reference_applications","dimensions")}
                if state["total_completed"]%config.get("checkpoint_every",10)==0:
                    state["elapsed_seconds"]+=time.monotonic()-started;started=time.monotonic()
                    save_checkpoint(path,dict(config=config,state=state,rng=rng.bit_generator.state))
            state["case_index"]+=1
        state["elapsed_seconds"]+=time.monotonic()-started
        save_checkpoint(path,dict(config=config,state=state,rng=rng.bit_generator.state))
        report=dict(status="completed_planning_validation",scope=SCOPE,
            generating_assumptions=config["generating_assumptions"],parameters=config["model"],
            cases={name:_summarize_case(value,config) for name,value in state["cases"].items()},
            total_simulations=state["total_completed"],elapsed_seconds=state["elapsed_seconds"])
        write_json(output/"result.json",report)
        return report


def main(argv=None):
    parser=argparse.ArgumentParser(description="Evaluate simultaneous inference with simulated unit-level models.")
    parser.add_argument("config",type=Path);parser.add_argument("--output",type=Path);parser.add_argument("--resume",action="store_true");parser.add_argument("--plan",action="store_true")
    args=parser.parse_args(argv)
    if not args.plan and args.output is None:parser.error("执行需要 --output；仅查看成本可用 --plan")
    result=planning_workload(args.config) if args.plan else run_calibration(args.config,args.output,args.resume)
    print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))


if __name__=="__main__":main()
