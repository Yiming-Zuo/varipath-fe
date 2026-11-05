import argparse
import json
from pathlib import Path

from .records import METHODS
from .workflow import (doctor, evaluate, intervene, load_config, predict, prepare,
                       prepare_mechanism, reference, run_deployment, run_md, sample, seal_deployments)


def _named_paths(values):
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError("路径参数须写作 ALDP=PATH 或 ALA4=PATH")
        name, path = value.split("=", 1)
        if name not in ("ALDP", "ALA4") or name in result or not path:
            raise ValueError("路径参数的分子名称重复或无效")
        result[name] = Path(path)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(prog="varipath")
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("doctor", help="核对依赖、分子维数和原生物理能量")
    check.add_argument("config", type=Path)
    prep = commands.add_parser("prepare", help="用独立 D 数据冻结坐标与保护几何")
    prep.add_argument("config", type=Path)
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--resume", action="store_true")
    sampling = commands.add_parser("sample", help="生成独立 P、I 或 R 分子采样池")
    sampling.add_argument("config", type=Path)
    sampling.add_argument("--role", choices=("P", "I", "R"), required=True)
    sampling.add_argument("--output", type=Path, required=True)
    sampling.add_argument("--resume", action="store_true")
    sampling.add_argument("--look", type=int, choices=(1, 2, 3, 4), default=1,
                          help="R 保存 nR、2nR、4nR 或 8nR；其他角色只允许1")
    run = commands.add_parser("run", help="运行一个完整冷启动部署，先付费完成共同后备")
    run.add_argument("config", type=Path)
    run.add_argument("--preparation", type=Path, required=True)
    run.add_argument("--method", choices=METHODS, required=True)
    run.add_argument("--replica", type=int, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--resume", action="store_true")
    md = commands.add_parser("endpoint-md", help="同一冷启动预算内运行付费后备与真实 ALA4 端态 MD")
    md.add_argument("config", type=Path)
    md.add_argument("--preparation", type=Path, required=True)
    md.add_argument("--replica", type=int, required=True)
    md.add_argument("--output", type=Path, required=True)
    md.add_argument("--resume", action="store_true")
    smoke = commands.add_parser("smoke", help="运行解析示例")
    smoke.add_argument("--output", type=Path, required=True)
    smoke.add_argument("--resume", action="store_true")
    analytical = commands.add_parser("analytical", help="运行 E0 解析、有限步与阴性控制")
    analytical.add_argument("--output", type=Path, required=True)
    panel = commands.add_parser("mechanism-prepare", help="从真实冻结路线及认证目录准备 E1 固定面板")
    panel.add_argument("config", type=Path)
    panel.add_argument("--preparation", type=Path, required=True)
    panel.add_argument("--deployment", type=Path, required=True)
    panel.add_argument("--catalog", type=Path)
    panel.add_argument("--output", type=Path, required=True)
    panel.add_argument("--resume", action="store_true")
    prediction = commands.add_parser("predict", help="双体系 P 全面板零干预导数与盲预测封存")
    prediction.add_argument("config", type=Path)
    prediction.add_argument("--panel", action="append", required=True, help="ALDP=panel.pt；重复提供 ALA4")
    prediction.add_argument("--bank", action="append", required=True, help="ALDP=bank.pt；重复提供 ALA4")
    prediction.add_argument("--output", type=Path, required=True)
    prediction.add_argument("--resume", action="store_true")
    intervention = commands.add_parser("intervene", help="P 封存后执行独立 I 全量及半剂量干预")
    intervention.add_argument("config", type=Path)
    intervention.add_argument("--predictions", type=Path, required=True)
    intervention.add_argument("--bank", action="append", required=True)
    intervention.add_argument("--output", type=Path, required=True)
    intervention.add_argument("--resume", action="store_true")
    seal = commands.add_parser("seal-f", help="全量封存八法正式部署及已配置的直接端态 MD")
    seal.add_argument("config", type=Path)
    seal.add_argument("--runs", type=Path, required=True)
    seal.add_argument("--registry", type=Path, required=True)
    ref = commands.add_parser("reference", help="P/I/F 封存后显式揭盲双体系 R 并计算参考联合区间")
    ref.add_argument("config", type=Path)
    ref.add_argument("--bank", action="append", required=True)
    ref.add_argument("--registry", type=Path, required=True)
    ref.add_argument("--look", type=int, choices=(1, 2, 3, 4), default=1)
    ref.add_argument("--output", type=Path, required=True)
    evaluation = commands.add_parser("evaluate", help="传播独立参考不确定性并全量计算 E3 联合门")
    evaluation.add_argument("config", type=Path)
    evaluation.add_argument("--runs", type=Path, required=True)
    evaluation.add_argument("--reference", action="append", required=True)
    evaluation.add_argument("--registry", type=Path, required=True)
    evaluation.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "analytical":
            from .analytical import run_analytical
            result = run_analytical(args.output)
        elif args.command == "smoke":
            from .smoke import run_smoke
            result = run_smoke(args.output, resume=args.resume)
        else:
            config = load_config(args.config)
            if args.command == "doctor":
                result = doctor(config)
            elif args.command == "prepare":
                result = prepare(config, args.output, args.resume)
            elif args.command == "sample":
                result = sample(config, args.role, args.output, args.resume, args.look)
            elif args.command == "endpoint-md":
                result = run_md(config, args.preparation, args.replica, args.output, args.resume)
            elif args.command == "mechanism-prepare":
                result = prepare_mechanism(config, args.preparation, args.deployment, args.output, args.catalog, args.resume)
            elif args.command == "predict":
                result = predict(config, _named_paths(args.panel), _named_paths(args.bank), args.output, args.resume)
            elif args.command == "intervene":
                result = intervene(config, args.predictions, _named_paths(args.bank), args.output, args.resume)
            elif args.command == "seal-f":
                result = seal_deployments(config, args.runs, args.registry)
            elif args.command == "reference":
                result = reference(config, _named_paths(args.bank), args.registry, args.output, args.look)
            elif args.command == "evaluate":
                result = evaluate(config, args.runs, _named_paths(args.reference), args.registry, args.output)
            else:
                result = run_deployment(config, args.preparation, args.method, args.replica, args.output, args.resume)
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
        if result.get("status") == "failed":
            return 1
        return 2 if result.get("status") == "no_legal_output" else 0
    except (ValueError, RuntimeError, FileNotFoundError, FileExistsError) as exc:
        parser.exit(2, f"{type(exc).__name__}: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
