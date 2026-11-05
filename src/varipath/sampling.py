from __future__ import annotations

import json
import copy
import math
from pathlib import Path
import time

import numpy as np
import torch

RANDOM_STREAM_VERSION = 2
MODE_NAMESPACES = {"smoke": 0, "development": 1, "formal": 2}


def frozen_system_definition(system):

    layers, visited = [], set()
    current = system
    while id(current) not in visited:
        visited.add(id(current))
        layer = dict(kind=f"{type(current).__module__}.{type(current).__qualname__}",
                     name=current.name, dimension=int(current.dim))
        if hasattr(current, "state"):
            layer["endpoint_states"] = [float(current.state(i)) for i in (0, 1)]
        if hasattr(current, "metadata"):
            layer["metadata"] = copy.deepcopy(current.metadata)
        chart = getattr(current, "chart", None)
        if chart is not None:
            layer["chart_definition"] = {name:copy.deepcopy(getattr(chart, name)) for name in
                ("triad", "heavy", "parents", "hydrogens", "groups", "split", "n_atoms", "dim") if hasattr(chart, name)}
        energy = getattr(current, "energy", None)
        if energy is not None:
            layer["energy_definition"] = {name:getattr(energy, name) for name in ("gb_kind", "offset", "n_atoms") if hasattr(energy, name)}
        for name in ("windows", "encoded", "bonds", "torsion_quads", "triad"):
            if hasattr(current, name):
                layer[name] = copy.deepcopy(getattr(current, name))
        if hasattr(current, "asset_path"):
            xml = Path(current.asset_path)/"system.xml"
            layer["system_xml"] = xml.read_text(encoding="utf-8") if xml.is_file() else None
        layers.append(layer)
        if hasattr(current, "base"):
            current = current.base
        elif hasattr(current, "system"):
            current = current.system
        else:
            break
    return dict(schema_version=1, layers=layers)


def freeze_system(system):
    if not hasattr(system, "state_dict"):
        raise ValueError("采样银行需要可冻结的完整物理 state_dict")
    return dict(schema_version=1, random_stream_version=RANDOM_STREAM_VERSION,
                name=system.name, shape=[int(system.dim)],
                state_dict={key:value.detach().cpu().clone() for key,value in system.state_dict().items()},
                physical_definition=frozen_system_definition(system))


def verify_frozen_system(system, frozen):
    if frozen.get("schema_version") != 1 or frozen.get("random_stream_version") != RANDOM_STREAM_VERSION:
        raise ValueError("采样银行的物理冻结定义或随机流版本不兼容")
    if frozen.get("name") != system.name or frozen.get("shape") != [int(system.dim)]:
        raise ValueError("采样银行的物理体系或坐标形状发生变化")
    current = system.state_dict()
    saved = frozen.get("state_dict", {})
    if current.keys() != saved.keys() or any(current[key].dtype != saved[key].dtype or
            not torch.equal(current[key].detach().cpu(), saved[key].detach().cpu()) for key in current):
        raise ValueError("采样银行的完整 Hamiltonian 或仿射坐标参数发生变化")
    if frozen.get("physical_definition") != frozen_system_definition(system):
        raise ValueError("采样银行的 GB 模型、端态、约束/拓扑或固定物理定义发生变化")


def verify_bank_system(system, bank, directory):

    frozen = bank.get("frozen_system")
    if frozen is None:
        path = Path(directory)/"ladder-0.pt"
        if not path.is_file():
            raise ValueError("旧采样银行缺少完整物理冻结证据，只能读取旧输出，不能继续执行")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        kernels = checkpoint.get("sampler", {}).get("kernels", [])
        if checkpoint.get("random_stream_version") != RANDOM_STREAM_VERSION or not kernels:
            raise ValueError("旧银行的 ladder 检查点没有兼容的物理证明")
        kernel = kernels[0]
        if "physics_state" not in kernel or "physics_definition" not in kernel:
            raise ValueError("旧检查点缺少 GB/拓扑等非 tensor 物理定义，只能读取旧输出，不能继续执行")
        frozen = dict(schema_version=1, random_stream_version=RANDOM_STREAM_VERSION,
                      name=kernel.get("system"), shape=[kernel.get("dimension")],
                      state_dict=kernel["physics_state"], physical_definition=kernel["physics_definition"])
    verify_frozen_system(system, frozen)
    if bank.get("random_stream_version") != RANDOM_STREAM_VERSION or bank["positions"].shape[-1] != system.dim:
        raise ValueError("采样银行的随机流版本或实际坐标形状与冻结定义不符")
    return frozen

def _atomic_save(path, state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+".pending")
    with temporary.open("wb") as stream:
        torch.save(state, stream)
    temporary.replace(path)


def _default_scale(system):
    if hasattr(system,"scale"):
        return _default_scale(system.system)/system.scale.detach().cpu()
    split = system.chart.split
    scale = torch.full((system.dim,),.002,dtype=torch.float64)
    scale[0],scale[2] = .02,.02
    scale[split:] = .03
    return scale


def haar_rotation(generator,device=None):
    q=torch.randn(4,dtype=torch.float64,generator=generator)
    w,x,y,z=q/q.norm()
    matrix=torch.stack((1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y))).reshape(3,3)
    return matrix.to(device=device)


class HMCSampler:

    def __init__(self,system,state,seed,step_size=.05,leapfrog_steps=8,coordinate_scale=None,initial=None):
        if not math.isfinite(step_size) or step_size<=0 or type(leapfrog_steps) is not int or leapfrog_steps<1:
            raise ValueError("HMC 步长及积分步数必须为正")
        if (system.name=="ALDP" and (not math.isfinite(state) or state<=0)) or (system.name=="ALA4" and not 0<=state<=1):
            raise ValueError("温度或极性耦合状态非法")
        self.system,self.state = system,float(state)
        self.step_size,self.leapfrog_steps = float(step_size),leapfrog_steps
        self.scale = _default_scale(system) if coordinate_scale is None else torch.as_tensor(coordinate_scale,dtype=torch.float64).clone()
        if self.scale.shape!=(system.dim,) or not torch.isfinite(self.scale).all() or (self.scale<=0).any():
            raise ValueError("HMC 坐标尺度需要同维有限正数")
        self.rng = np.random.default_rng(seed)
        self.torch_rng = torch.Generator().manual_seed(seed)
        template=system.initial_coordinates()
        self.z = template.detach().clone() if initial is None else torch.as_tensor(initial,dtype=torch.float64,device=template.device).detach().clone()
        self.scale=self.scale.to(self.z)
        if self.z.shape!=(system.dim,) or not torch.isfinite(self.z).all():
            raise ValueError("HMC 初态维数错误")
        self.value,self.gradient = self._value_gradient(self.z)
        if not torch.isfinite(self.value) or not torch.isfinite(self.gradient).all() or not bool(system.in_domain(self.z)):
            raise ValueError("HMC 初态不在有限全 L 型支持域")
        self.attempted,self.accepted,self.domain_rejections,self.numerical_rejections = 0,0,0,0

    def _value_gradient(self,z):
        gradient,value = torch.func.grad_and_value(lambda q:self.system.unrestricted_u_state(q,self.state))(z)
        return value.detach(),gradient.detach()

    def step(self):
        momentum = torch.from_numpy(self.rng.normal(size=self.system.dim)).to(self.z)/self.scale
        old_kinetic = .5*(momentum*self.scale).square().sum()
        z,gradient = self.z.clone(),self.gradient.clone()
        p = momentum-.5*self.step_size*gradient
        valid = True
        try:
            for index in range(self.leapfrog_steps):
                z = z+self.step_size*self.scale.square()*p
                value,gradient = self._value_gradient(z)
                if not torch.isfinite(value) or not torch.isfinite(gradient).all() or not torch.isfinite(z).all():
                    valid = False
                    break
                if index+1<self.leapfrog_steps:
                    p = p-self.step_size*gradient
            p = p-.5*self.step_size*gradient
        except (FloatingPointError,torch.linalg.LinAlgError):
            valid = False
        in_domain = valid and bool(self.system.in_domain(z))
        ratio = -float("inf")
        if valid and in_domain:
            kinetic = .5*(p*self.scale).square().sum()
            ratio = float(self.value+old_kinetic-value-kinetic)
            valid = math.isfinite(ratio)
        log_uniform = math.log(max(float(self.rng.random()),np.finfo(float).tiny))
        accepted = valid and in_domain and log_uniform<min(0.,ratio)
        self.attempted += 1
        self.accepted += int(accepted)
        self.numerical_rejections += int(not valid)
        self.domain_rejections += int(valid and not in_domain)
        if accepted:
            self.z,self.value,self.gradient = z.detach(),value.detach(),gradient.detach()
        return dict(accepted=bool(accepted),log_acceptance_ratio=ratio if math.isfinite(ratio) else None,in_domain=bool(in_domain),numerical_rejection=not valid)

    def sample(self,count,warmup=0,thin=1,check_budget=None):
        if type(count) is not int or count<=0 or type(warmup) is not int or warmup<0 or type(thin) is not int or thin<=0:
            raise ValueError("样本数、预热和抽稀设置非法")
        for _ in range(warmup):
            if check_budget is not None:
                check_budget()
            self.step()
        points=[]
        for _ in range(count):
            if check_budget is not None:
                check_budget()
            for _ in range(thin):
                self.step()
            points.append(self.z.clone())
        if check_budget is not None:
            check_budget()
        return torch.stack(points)

    def state_dict(self):
        physics_state={key:value.detach().clone() for key,value in self.system.state_dict().items()} if hasattr(self.system,"state_dict") else {}
        return dict(format_version=1,system=self.system.name,dimension=self.system.dim,physics_state=physics_state,physics_definition=frozen_system_definition(self.system),state=self.state,step_size=self.step_size,leapfrog_steps=self.leapfrog_steps,scale=self.scale,z=self.z,value=self.value,gradient=self.gradient,numpy_rng=self.rng.bit_generator.state,torch_rng=self.torch_rng.get_state(),attempted=self.attempted,accepted=self.accepted,domain_rejections=self.domain_rejections,numerical_rejections=self.numerical_rejections)

    def load_state_dict(self,data):
        if data.get("format_version")!=1 or data["system"]!=self.system.name or data["dimension"]!=self.system.dim or data["state"]!=self.state or data["step_size"]!=self.step_size or data["leapfrog_steps"]!=self.leapfrog_steps or not torch.equal(data["scale"],self.scale):
            raise ValueError("HMC 恢复配置与保存时不一致")
        current=self.system.state_dict() if hasattr(self.system,"state_dict") else {}
        saved=data.get("physics_state",{})
        if current.keys()!=saved.keys() or any(not torch.equal(current[key],saved[key]) for key in current):
            raise ValueError("HMC 检查点的完整 Hamiltonian 或坐标参数发生变化")
        if data.get("physics_definition") != frozen_system_definition(self.system):
            raise ValueError("HMC 检查点的固定物理定义缺失或发生变化")
        if not bool(self.system.in_domain(data["z"])):
            raise ValueError("HMC 检查点不在全 L 域")
        value,gradient = self._value_gradient(data["z"])
        if not torch.equal(value,data["value"]) or not torch.equal(gradient,data["gradient"]):
            raise ValueError("HMC 检查点与当前有效势不一致")
        self.z,self.value,self.gradient = data["z"].clone(),value,gradient
        self.rng.bit_generator.state = data["numpy_rng"]
        self.torch_rng.set_state(data["torch_rng"])
        for key in ("attempted","accepted","domain_rejections","numerical_rejections"):
            setattr(self,key,int(data[key]))

    def save(self,path):
        _atomic_save(path,self.state_dict())

    @classmethod
    def load(cls,system,path):
        data = torch.load(path,map_location=system.initial_coordinates().device,weights_only=False)
        sampler = cls(system,data["state"],0,data["step_size"],data["leapfrog_steps"],data["scale"],data["z"])
        sampler.load_state_dict(data)
        return sampler


class ReplicaExchangeSampler:
    def __init__(self,system,states,seed,step_size=.05,leapfrog_steps=8,coordinate_scale=None,initial=None):
        if len(states)<2 or any(a>=b for a,b in zip(states,states[1:])):
            raise ValueError("交换状态必须严格递增且至少两个")
        self.system,self.states = system,[float(s) for s in states]
        self.rng = np.random.default_rng(seed)
        self.kernels = [HMCSampler(system,state,seed+1+index,step_size,leapfrog_steps,coordinate_scale,initial) for index,state in enumerate(states)]
        self.walkers = np.arange(len(states))
        self.iteration = 0
        self.exchange_attempted = np.zeros(len(states)-1,dtype=np.int64)
        self.exchange_accepted = np.zeros(len(states)-1,dtype=np.int64)

    def cross_energies(self):
        return torch.stack([torch.stack([self.system.effective_u_state(kernel.z,state).detach() for kernel in self.kernels]) for state in self.states])

    def step(self,hmc_trials=1):
        if type(hmc_trials) is not int or hmc_trials<1:
            raise ValueError("每次交换前的 HMC 次数必须为正")
        for kernel in self.kernels:
            for _ in range(hmc_trials):
                kernel.step()
        cross = self.cross_energies()
        swaps=[]
        for i in range(self.iteration%2,len(self.states)-1,2):
            j=i+1
            ratio=float(-cross[i,j]-cross[j,i]+cross[i,i]+cross[j,j])
            accepted=math.log(max(float(self.rng.random()),np.finfo(float).tiny))<min(0.,ratio)
            self.exchange_attempted[i]+=1
            self.exchange_accepted[i]+=int(accepted)
            if accepted:
                left,right=self.kernels[i],self.kernels[j]
                left.z,right.z=right.z,left.z
                left.value,left.gradient=left._value_gradient(left.z)
                right.value,right.gradient=right._value_gradient(right.z)
                self.walkers[[i,j]]=self.walkers[[j,i]]
            swaps.append(dict(first=i,second=j,accepted=accepted,log_ratio=ratio))
        self.iteration+=1
        rotations=torch.stack([haar_rotation(kernel.torch_rng,kernel.z.device) for kernel in self.kernels])
        return dict(swaps=swaps,positions=torch.stack([kernel.z.clone() for kernel in self.kernels]),rotations=rotations,u_kn=self.cross_energies(),walkers=self.walkers.copy())

    def state_dict(self):
        return dict(format_version=1,system=self.system.name,states=self.states,kernels=[kernel.state_dict() for kernel in self.kernels],numpy_rng=self.rng.bit_generator.state,walkers=self.walkers.copy(),iteration=self.iteration,exchange_attempted=self.exchange_attempted.copy(),exchange_accepted=self.exchange_accepted.copy())

    def load_state_dict(self,data):
        if data.get("format_version")!=1 or data["system"]!=self.system.name or data["states"]!=self.states or len(data["kernels"])!=len(self.kernels):
            raise ValueError("交换采样检查点与当前配置不一致")
        for kernel,state in zip(self.kernels,data["kernels"]):
            kernel.load_state_dict(state)
        if sorted(data["walkers"].tolist())!=list(range(len(self.states))):
            raise ValueError("检查点 walker 不是完整排列")
        self.rng.bit_generator.state=data["numpy_rng"]
        for key in ("walkers","iteration","exchange_attempted","exchange_accepted"):
            setattr(self,key,data[key])

    def save(self,path):
        _atomic_save(path,self.state_dict())

    @classmethod
    def load(cls,system,path):
        data=torch.load(path,map_location=system.initial_coordinates().device,weights_only=False)
        first=data["kernels"][0]
        sampler=cls(system,data["states"],0,first["step_size"],first["leapfrog_steps"],first["scale"],first["z"])
        sampler.load_state_dict(data)
        return sampler


def _run_sampling(system,config,output,resume=False,check_budget=None):

    output=Path(output)
    if any(type(config.get(key)) is not int for key in ("count", "warmup", "ladders")):
        raise ValueError("count、warmup 和 ladders 必须是整数，不能截断小数或接受布尔值")
    count,warmup,ladders=(config[k] for k in ("count","warmup","ladders"))
    checkpoint_every=config.get("checkpoint_every",10)
    block_size=config.get("analysis_block_size",1)
    if min(count,ladders)<1 or warmup<0 or type(checkpoint_every) is not int or checkpoint_every<1 or type(block_size) is not int or block_size<1:
        raise ValueError("采样数、梯数或预热非法")
    if config.get("mode","smoke") not in ("smoke","development","formal"):
        raise ValueError("采样模式必须显式指定")
    states=config.get("states",[300.,338.7,382.4,431.8,487.5,550.] if system.name=="ALDP" else np.linspace(0,1,17).tolist())
    output.mkdir(parents=True,exist_ok=True)
    settings=output/"sampling-config.json"
    if resume:
        if not settings.exists() or json.loads(settings.read_text())!=config:
            raise ValueError("恢复配置与原采样不同")
    else:
        if settings.exists():
            raise FileExistsError("已有采样必须显式 --resume")
        settings.write_text(json.dumps(config,ensure_ascii=False,indent=2)+"\n")
    started=time.monotonic()
    records=[]
    role=config.get("role","D")
    if role not in ("D","D-fit","D-select","R","P","I","F","F-train","F-fit","F-select","F-production","fallback"):
        raise ValueError("未知的数据角色")
    mode = config.get("mode", "smoke")
    role_seed=np.random.SeedSequence([int(config["seed"]),RANDOM_STREAM_VERSION,MODE_NAMESPACES[mode],*role.encode("utf-8")]).generate_state(1,dtype=np.uint32)[0].item()
    source_units=[f"{system.name}:{mode}:{role}:v{RANDOM_STREAM_VERSION}:{role_seed}:ladder-{index}" for index in range(ladders)]
    for ladder in range(ladders):
        path=output/f"ladder-{ladder}.pt"
        seed=role_seed+10000*ladder
        sampler=ReplicaExchangeSampler(system,states,seed,step_size=config.get("step_size",.05),leapfrog_steps=config.get("leapfrog_steps",8),initial=system.initial_coordinates(ladder%len(system.metadata["initial_positions_nm"])))
        if resume and path.exists():
            checkpoint=torch.load(path,map_location=system.initial_coordinates().device,weights_only=False)
            if checkpoint.get("random_stream_version") != RANDOM_STREAM_VERSION:
                raise ValueError("旧随机流版本的采样检查点不能混入当前 phase 命名空间")
            sampler.load_state_dict(checkpoint["sampler"])
            points,cross,walkers,rotations=checkpoint["points"],checkpoint["cross"],checkpoint["walkers"],checkpoint["rotations"]
        else:
            points,cross,walkers,rotations=[],[],[],[]
        total=warmup+count
        while sampler.iteration<total:
            if check_budget is not None:
                check_budget()
            record=sampler.step(config.get("hmc_trials",1))
            if sampler.iteration>warmup:
                points.append(record["positions"])
                cross.append(record["u_kn"])
                walkers.append(record["walkers"])
                rotations.append(record["rotations"])
            if sampler.iteration%checkpoint_every==0 or sampler.iteration==total:
                _atomic_save(path,dict(random_stream_version=RANDOM_STREAM_VERSION,sampler=sampler.state_dict(),points=points,cross=cross,walkers=walkers,rotations=rotations))
        records.append(dict(points=torch.stack(points),rotations=torch.stack(rotations),u_kn=torch.stack(cross),walkers=np.stack(walkers),hmc_acceptance=[k.accepted/max(k.attempted,1) for k in sampler.kernels],exchange_acceptance=np.divide(sampler.exchange_accepted,sampler.exchange_attempted,out=np.zeros_like(sampler.exchange_accepted,dtype=float),where=sampler.exchange_attempted>0).tolist()))
    result=dict(status="completed_sampling",mode=config.get("mode","smoke"),role=role,role_seed=role_seed,source_unit_ids=source_units,analysis_block_ids=[[f"{unit}:block-{frame//block_size}" for frame in range(count)] for unit in source_units],eligible_for_formal=False,qualification="sampling completed; independent convergence and accuracy gates are required",mixing=dict(status="not_assessed",required="rank normalized split Rhat, bulk/tail ESS, endpoint region support, and reference precision; mode and sample count never confer qualification"),ladder_id=np.broadcast_to(np.arange(ladders)[:,None,None],(ladders,count,len(states))).copy(),analysis_block_id=np.broadcast_to((np.arange(count)//block_size)[None,:,None],(ladders,count,len(states))).copy(),elapsed_seconds=time.monotonic()-started,states=states,positions=torch.stack([r["points"] for r in records]),u_kn=torch.stack([r["u_kn"] for r in records]),walkers=np.stack([r["walkers"] for r in records]),hmc_acceptance=[r["hmc_acceptance"] for r in records],exchange_acceptance=[r["exchange_acceptance"] for r in records])
    result["rotations"]=torch.stack([r["rotations"] for r in records])
    result["random_stream_version"] = RANDOM_STREAM_VERSION
    result["mode_namespace"] = mode
    result["rotation_measure"]="independent normalized Haar SO(3); physical positions=system.positions(shape) @ rotation; transport carries rotation unchanged"
    return result


def _atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name+".pending")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n")
    temporary.replace(path)


def _same_except_count(first, second):
    return {key:value for key,value in first.items() if key != "count"} == {key:value for key,value in second.items() if key != "count"}


def _validate_reference_prefix(output, bank, config):
    old_count = int(bank["positions"].shape[1])
    if bank.get("role") != "R" or old_count >= config["count"]:
        raise ValueError("只有已完成的 R 银行可以延长到更大的保存长度")
    for ladder in range(config["ladders"]):
        path = output/f"ladder-{ladder}.pt"
        if not path.exists():
            raise FileNotFoundError("R 延长缺少原 ladder 检查点，不能重新暖机或起跑")
        checkpoint = torch.load(path, weights_only=False)
        if checkpoint.get("random_stream_version") != RANDOM_STREAM_VERSION:
            raise ValueError("R 检查点的随机流版本不兼容，保留旧验证产物而不重启")
        iteration = checkpoint["sampler"]["iteration"]
        if not config["warmup"]+old_count <= iteration <= config["warmup"]+config["count"]:
            raise ValueError("R 检查点的暖机/保存进度与固定前缀不符")
        for key, saved in (("points", "positions"), ("cross", "u_kn"), ("rotations", "rotations")):
            prefix = torch.stack(checkpoint[key][:old_count])
            if not torch.equal(prefix, bank[saved][ladder]):
                raise ValueError("R 延长检查点与既有银行的原始前缀不一致")
        if not np.array_equal(np.stack(checkpoint["walkers"][:old_count]), bank["walkers"][ladder]):
            raise ValueError("R 延长检查点的 walker 前缀发生变化")


def run_sampling(system, config, output, resume=False, check_budget=None, extend=False):

    config = dict(config)
    if config.setdefault("random_stream_version", RANDOM_STREAM_VERSION) != RANDOM_STREAM_VERSION:
        raise ValueError("当前入口只接受显式版本2的随机流协议")
    output = Path(output)
    settings = output/"sampling-config.json"
    bank_path = output/"bank.pt"
    progress_path = output/"sampling-progress.json"
    role = config.get("role", "D")
    if any(type(config.get(key)) is not int for key in ("count", "warmup", "ladders")):
        raise ValueError("count、warmup 和 ladders 必须是整数，不能截断小数或接受布尔值")
    if config["count"] < 1 or config["ladders"] < 1 or config["warmup"] < 0:
        raise ValueError("count、ladders 必须为正，warmup 不得为负")
    if settings.exists() and not resume:
        raise FileExistsError("已有采样必须显式 --resume")
    if resume and not settings.exists():
        raise ValueError("恢复运行缺少原采样配置")
    old_bank = torch.load(bank_path, weights_only=False) if resume and bank_path.exists() else None
    old_settings = json.loads(settings.read_text()) if settings.exists() else None
    if resume and old_settings is not None and old_settings.get("random_stream_version") != RANDOM_STREAM_VERSION:
        raise ValueError("检查点随机流版本不匹配，不能恢复")
    old_progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}
    initial_system = freeze_system(system)
    if old_bank is not None:
        verify_bank_system(system, old_bank, output)
    if extend:
        if not resume or role != "R" or old_settings is None or old_settings.get("role") != "R" or old_bank is None:
            raise ValueError("显式延长只适用于有完整银行与检查点的 R 恢复运行")
        if not _same_except_count(old_settings, config) or config["count"] <= old_settings["count"]:
            raise ValueError("R 延长仅允许增加 count；物理核、暖机、随机流与其余配置必须相同")
        _validate_reference_prefix(output, old_bank, config)
        stop_path = output/"reference-status.json"
        stop = json.loads(stop_path.read_text()) if stop_path.exists() else {}
        if stop.get("final") or stop.get("stopped") or old_bank.get("reference_final") or old_bank.get("reference_stopped"):
            raise ValueError("R 已明确停止或标为 final，不能继续延长或新增查看")

        _atomic_json(settings, config)
    elif resume and old_settings is not None and old_settings != config:
        raise ValueError("恢复配置与原采样不同；R 增长必须显式 extend")
    if resume and old_bank is not None and old_bank["positions"].shape[1] < config["count"]:
        if role != "R":
            raise ValueError("只有 R 的明确延长事务可以继续增长银行")
        _validate_reference_prefix(output, old_bank, config)
        stop_path = output/"reference-status.json"
        stopped = json.loads(stop_path.read_text()) if stop_path.exists() else {}
        if stopped.get("final") or stopped.get("stopped") or old_bank.get("reference_final") or old_bank.get("reference_stopped"):
            raise ValueError("R 已明确停止或标为 final，不能继续延长或新增查看")
    previous_elapsed = max(float(old_progress.get("elapsed_seconds", 0.)),
                           float((old_bank or {}).get("elapsed_seconds", 0.)))
    base_count = int(old_progress.get("base_count", config.get("reference_base_count", old_settings["count"] if old_settings else config["count"])))
    started = time.monotonic()
    progress = dict(base_count=base_count, target_count=int(config["count"]), role=role,
                    completed_count=int(old_bank["positions"].shape[1]) if old_bank is not None else 0,
                    status="running", elapsed_seconds=previous_elapsed, checkpoint_interval_seconds=1.)
    last_progress = -math.inf

    def record_progress(force=False):
        nonlocal last_progress
        now = time.monotonic()
        if not force and now-last_progress < 1.:
            return
        progress["elapsed_seconds"] = previous_elapsed+time.monotonic()-started
        _atomic_json(progress_path, progress)
        last_progress = now

    def checked():
        record_progress()
        if check_budget is not None:
            check_budget()

    try:
        result = _run_sampling(system, config, output, resume=resume, check_budget=checked)
        verify_frozen_system(system, initial_system)
        progress.update(status="completed_sampling", completed_count=int(config["count"]))
        record_progress(force=True)
        result["elapsed_seconds_current_call"] = time.monotonic()-started
        result["elapsed_seconds"] = progress["elapsed_seconds"]
        result["production_draws_per_ladder"] = int(config["count"])
        result["frozen_system"] = initial_system
        if role == "R":
            result["reference_base_count"] = base_count
        _atomic_save(bank_path, result)
        return result
    except BaseException:
        progress["status"] = "interrupted"
        raise
    finally:

        if output.exists():
            record_progress(force=True)
