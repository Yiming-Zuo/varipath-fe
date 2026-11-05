from __future__ import annotations

import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import openmm as mm
from openmm import unit
import torch

from .physics import MolecularSystem
from .readouts import R_KJ
from .records import Budget, read_json, run_lock, save_checkpoint, write_json


VERLET_UPSTREAM = "https://github.com/choderalab/openmmtools/blob/a42cbdc862a0a4fe996a579dd1b04611ce9916a4/openmmtools/integrators.py"


def require_platform(name):
    available = [mm.Platform.getPlatform(i).getName() for i in range(mm.Platform.getNumPlatforms())]
    if name not in available:
        raise RuntimeError(f"OpenMM {name} 平台不可用；可用平台：{', '.join(available)}")
    return mm.Platform.getPlatformByName(name)


class VelocityVerletIntegrator(mm.CustomIntegrator):


    def __init__(self, timestep):
        super().__init__(timestep)
        self.addPerDofVariable("x1", 0)
        self.addUpdateContextState()
        self.addComputePerDof("v", "v+0.5*dt*f/m")
        self.addComputePerDof("x", "x+dt*v")
        self.addComputePerDof("x1", "x")
        self.addConstrainPositions()
        self.addComputePerDof("v", "v+0.5*dt*f/m+(x-x1)/dt")
        self.addConstrainVelocities()


class CanonicalConstraintBridge:


    def __init__(self, system, auxiliary_mass=4.):
        self.masses = np.asarray([system.getParticleMass(i).value_in_unit(unit.dalton)
                                  for i in range(system.getNumParticles())])
        if (self.masses <= 0).any() or any(system.isVirtualSite(i) for i in range(len(self.masses))):
            raise ValueError("质量 bridge 只支持正质量且没有虚拟位点的体系")
        if auxiliary_mass <= 0 or not math.isfinite(auxiliary_mass):
            raise ValueError("辅助质量必须有限且为正")
        self.auxiliary_mass = float(auxiliary_mass)
        inverse = 1/self.auxiliary_mass
        reduced = .5/(2*inverse)
        if float(np.float32(inverse)) != inverse or float(np.float32(reduced)) != reduced:
            raise ValueError("辅助 SHAKE 的质量系数必须可由 float32 精确表示")
        constraints, graph = [], {}
        for index in range(system.getNumConstraints()):
            first, second, length = system.getConstraintParameters(index)
            constraints.append((first, second, length.value_in_unit(unit.nanometer)))
            graph.setdefault(first, []).append(second)
            graph.setdefault(second, []).append(first)
        self.groups, self.rows = [], []
        remaining = set(graph)
        while remaining:
            component, todo = set(), [min(remaining)]
            while todo:
                atom = todo.pop()
                if atom not in component:
                    component.add(atom)
                    todo.extend(graph[atom])
            remaining -= component
            centers = [atom for atom in component if len(graph[atom]) > 1]
            if len(component) == 2:
                parent = max(component, key=lambda atom: (self.masses[atom], -atom))
            elif len(centers) == 1 and len(graph[centers[0]]) <= 3:
                parent = centers[0]
            else:
                raise ValueError("质量 bridge 不支持非星图或超过三叶的约束组")
            leaves = sorted(component-{parent})
            if any(graph[leaf] != [parent] for leaf in leaves):
                raise ValueError("每个受约束叶只能连接一个父原子")
            lengths = []
            for leaf in leaves:
                index = next(i for i, (a, b, _) in enumerate(constraints) if {a, b} == {parent, leaf})
                length = constraints[index][2]
                auxiliary = float(np.sqrt(np.float64(np.float32(length*length))))
                if float(np.float32(auxiliary*auxiliary)) != float(np.float32(length*length)):
                    raise ValueError("辅助约束平方与固定 SHAKE 参数不一致")
                self.rows.append((index, parent, leaf, length, auxiliary))
                lengths.append(length)
            if np.ptp(lengths) > 1e-14:
                raise ValueError("同父星图内的约束长度必须相等")
            self.groups.append((parent, leaves))
        self.parents = np.asarray([row[1] for row in self.rows], dtype=int)
        self.leaves = np.asarray([row[2] for row in self.rows], dtype=int)
        self.native_lengths = np.asarray([row[3] for row in self.rows])
        self.auxiliary_lengths = np.asarray([row[4] for row in self.rows])

    def auxiliary_system(self, native):
        auxiliary = mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(native))
        for atom in range(auxiliary.getNumParticles()):
            auxiliary.setParticleMass(atom, self.auxiliary_mass*unit.dalton)
        for index, parent, leaf, _, length in self.rows:
            auxiliary.setConstraintParameters(index, parent, leaf, length*unit.nanometer)
        return auxiliary

    def _map(self, positions, scale):
        result = np.array(positions, dtype=float, copy=True)
        if len(self.rows):
            delta = result[..., self.leaves, :] - result[..., self.parents, :]
            result[..., self.leaves, :] = result[..., self.parents, :] + delta*scale[:, None]
        return result

    def to_native(self, positions):
        return self._map(positions, self.native_lengths/self.auxiliary_lengths)

    def to_auxiliary(self, positions):
        return self._map(positions, self.auxiliary_lengths/self.native_lengths)

    def log_metric_ratio(self, positions):
        result = np.zeros(np.asarray(positions).shape[:-2])
        for parent, leaves in self.groups:
            direction = np.asarray(positions)[..., leaves, :] - np.asarray(positions)[..., parent, None, :]
            direction = direction/np.linalg.norm(direction, axis=-1, keepdims=True)
            dots = direction@direction.swapaxes(-1, -2)
            native = np.diag(1/self.masses[leaves]) + dots/self.masses[parent]
            auxiliary = (np.eye(len(leaves))+dots)/self.auxiliary_mass
            sign0, ld0 = np.linalg.slogdet(native)
            sign1, ld1 = np.linalg.slogdet(auxiliary)
            if np.any(sign0 <= 0) or np.any(sign1 <= 0):
                raise ValueError("约束质量 Gram 非正定")
            result += .5*(ld0-ld1)
        return result

    def log_measure_ratio(self, positions):
        constant = 1.5*np.log(self.masses/self.auxiliary_mass).sum() + 2*np.log(self.native_lengths/self.auxiliary_lengths).sum()
        return constant + self.log_metric_ratio(positions)


def _recenter(positions, masses):
    return positions - (positions*masses[:, None]).sum(0)/masses.sum()


def _positions(context):
    return np.asarray(context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.nanometer)).copy()


def _energy(context):
    value = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    if not math.isfinite(value):
        raise FloatingPointError("原生 OpenMM 势能非有限")
    return float(value)


class MetropolizedEndpointMD:


    def __init__(self, system, endpoint, seed, *, steps_per_segment=100,
                 timestep_fs=1., platform="CPU", constraint_tolerance_nm=1e-10,
                 initial_positions=None, project_initialization=False):
        if endpoint not in (0, 1) or platform not in ("CPU", "Reference", "CUDA"):
            raise ValueError("端点或 OpenMM 平台无效")
        if type(steps_per_segment) is not int or steps_per_segment < 1 or not math.isfinite(timestep_fs) or timestep_fs <= 0:
            raise ValueError("每段 MD 步数和步长必须为正")
        if not 0 < constraint_tolerance_nm <= 1e-8:
            raise ValueError("约束入口容差必须在 (0,1e-8] nm")
        self.system, self.endpoint, self.platform_name = system, int(endpoint), platform
        self.steps_per_segment, self.timestep_fs = steps_per_segment, float(timestep_fs)
        self.constraint_tolerance_nm = float(constraint_tolerance_nm)
        self.temperature = float(system.state(endpoint)) if system.name == "ALDP" else 300.
        self.kT = R_KJ*self.temperature
        self.rng = np.random.default_rng(seed)
        raw = getattr(system, "system", system)
        native = mm.XmlSerializer.deserialize((raw.asset_path/"system.xml").read_text())
        if any(isinstance(force, mm.CMMotionRemover) or "Barostat" in type(force).__name__ for force in native.getForces()):
            raise ValueError("本提议核不接受额外状态更新器")
        if system.name == "ALA4":
            gb = next(force for force in native.getForces() if isinstance(force, mm.CustomGBForce))
            for atom in range(gb.getNumParticles()):
                parameters = list(gb.getParticleParameters(atom))
                parameters[0] *= math.sqrt(float(endpoint))
                gb.setParticleParameters(atom, parameters)
        self.native_xml = mm.XmlSerializer.serialize(native)
        self.bridge = CanonicalConstraintBridge(native)
        auxiliary = self.bridge.auxiliary_system(native)
        self.auxiliary_xml = mm.XmlSerializer.serialize(auxiliary)
        self.integrator = VelocityVerletIntegrator(self.timestep_fs*unit.femtosecond)
        self.integrator.setConstraintTolerance(1e-12)
        self.native_integrator = mm.VerletIntegrator(.001*unit.picosecond)
        self.native_integrator.setConstraintTolerance(1e-12)
        properties = {"Precision": "double", "DeterministicForces": "true"} if platform == "CUDA" else ({"Threads": "1"} if platform == "CPU" else {})
        self.auxiliary = mm.Context(auxiliary, self.integrator, require_platform(platform), properties)
        self.native = mm.Context(native, self.native_integrator, mm.Platform.getPlatformByName("Reference"))
        if platform == "CUDA" and self.auxiliary.getPlatform().getPropertyValue(self.auxiliary, "Precision") != "double":
            raise ValueError("实际 CUDA 提议平台没有使用 double")
        self.attempted = self.accepted = self.domain_rejections = self.numerical_rejections = 0
        self.initialization = {}
        initial_positions = raw.initial_positions[0].detach().cpu().numpy() if initial_positions is None else initial_positions
        self.set_positions(initial_positions, project_initialization=project_initialization)

    def _constraint_error(self, positions, *, auxiliary=False):
        lengths = self.bridge.auxiliary_lengths if auxiliary else self.bridge.native_lengths
        if len(lengths) == 0:
            return 0.
        return float(np.max(np.abs(np.linalg.norm(positions[self.bridge.leaves]-positions[self.bridge.parents], axis=-1)-lengths)))

    def _in_domain(self, positions):
        raw = getattr(self.system, "system", self.system)
        return bool(raw.in_domain_positions(torch.tensor(positions, dtype=torch.float64)))

    def set_positions(self, positions, *, project_initialization=False):
        previous = None
        if hasattr(self, "positions"):
            previous = (self.native.createCheckpoint(), self.auxiliary.createCheckpoint(),
                        self.positions.copy(), self.auxiliary_positions.copy(),
                        self.native_energy, self.metric, copy.deepcopy(self.initialization))
        try:
            self._set_positions(positions, project_initialization=project_initialization)
        except BaseException:
            if previous is not None:
                self.native.loadCheckpoint(previous[0])
                self.auxiliary.loadCheckpoint(previous[1])
                (self.positions, self.auxiliary_positions, self.native_energy,
                 self.metric, self.initialization) = previous[2:]
            raise

    def _set_positions(self, positions, *, project_initialization=False):
        positions = np.asarray(positions, dtype=float)
        if positions.shape != (len(self.bridge.masses), 3) or not np.isfinite(positions).all():
            raise ValueError("MD 初态必须是完整有限的全原子坐标")
        residual = self._constraint_error(positions)
        if residual > self.constraint_tolerance_nm and not project_initialization:
            raise ValueError("MD 数据入口违反约束；不能静默投影")
        canonical = _recenter(positions, self.bridge.masses)
        if project_initialization:
            self.native.setPositions(canonical*unit.nanometer)
            self.native.applyConstraints(1e-12)
            canonical = _recenter(_positions(self.native), self.bridge.masses)
        if not self._in_domain(canonical):
            raise ValueError("MD 初态不在全 L 型条件域")
        auxiliary = self.bridge.to_auxiliary(canonical)
        auxiliary = _recenter(auxiliary, np.full(len(auxiliary), self.bridge.auxiliary_mass))
        canonical = _recenter(self.bridge.to_native(auxiliary), self.bridge.masses)
        if max(self._constraint_error(canonical), self._constraint_error(auxiliary, auxiliary=True)) > self.constraint_tolerance_nm:
            raise ValueError("原生／辅助约束 bridge 未通过严格入口")
        self.native.setPositions(canonical*unit.nanometer)
        self.auxiliary.setPositions(auxiliary*unit.nanometer)
        self.native.setVelocities(np.zeros_like(canonical)*unit.nanometer/unit.picosecond)
        self.auxiliary.setVelocities(np.zeros_like(auxiliary)*unit.nanometer/unit.picosecond)
        self.positions, self.auxiliary_positions = canonical.copy(), auxiliary.copy()
        self.native_energy, self.metric = _energy(self.native), float(self.bridge.log_metric_ratio(canonical))
        self.initialization = {"constraint_projection": bool(project_initialization), "input_constraint_error_nm": residual,
                               "projected_initialization_is_equilibrium_data": False,
                               "native_com_gauge": "physical mass center zero", "auxiliary_com_gauge": "auxiliary mass center zero"}

    def step(self, check_budget=None):
        if check_budget is not None:
            check_budget()
        old_native, old_auxiliary = self.native.createCheckpoint(), self.auxiliary.createCheckpoint()
        old_fields = (self.positions.copy(), self.auxiliary_positions.copy(), self.native_energy, self.metric)
        accepted, numerical, domain, log_ratio = False, False, True, -math.inf
        delta_energy = delta_kinetic = None
        try:
            velocity = self.rng.standard_normal(self.auxiliary_positions.shape)*math.sqrt(self.kT/self.bridge.auxiliary_mass)
            velocity -= velocity.mean(0)
            self.auxiliary.setVelocities(velocity*unit.nanometer/unit.picosecond)
            self.auxiliary.applyVelocityConstraints(1e-12)
            old_velocity = self.auxiliary.getState(getVelocities=True).getVelocities(asNumpy=True).value_in_unit(unit.nanometer/unit.picosecond)
            old_kinetic = .5*self.bridge.auxiliary_mass*np.square(old_velocity).sum()
            self.integrator.step(self.steps_per_segment)
            state = self.auxiliary.getState(getPositions=True, getVelocities=True)
            auxiliary = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
            new_velocity = state.getVelocities(asNumpy=True).value_in_unit(unit.nanometer/unit.picosecond)
            auxiliary = _recenter(auxiliary, np.full(len(auxiliary), self.bridge.auxiliary_mass))
            new_velocity = new_velocity-new_velocity.mean(0)
            canonical = _recenter(self.bridge.to_native(auxiliary), self.bridge.masses)
            if (not np.isfinite(canonical).all() or not np.isfinite(new_velocity).all()
                    or max(self._constraint_error(canonical), self._constraint_error(auxiliary, auxiliary=True)) > self.constraint_tolerance_nm):
                raise FloatingPointError("MD 提议非有限或约束误差超过严格入口")
            domain = self._in_domain(canonical)
            if domain:
                kinetic = .5*self.bridge.auxiliary_mass*np.square(new_velocity).sum()
                self.native.setPositions(canonical*unit.nanometer)
                energy, metric = _energy(self.native), float(self.bridge.log_metric_ratio(canonical))
                delta_energy, delta_kinetic = energy-self.native_energy, float(kinetic-old_kinetic)
                log_ratio = -(delta_energy+delta_kinetic)/self.kT + metric-self.metric
                if not math.isfinite(log_ratio):
                    raise FloatingPointError("真实质量测度接受比非有限")
            log_uniform = math.log(max(float(self.rng.random()), np.finfo(float).tiny))
            accepted = domain and log_uniform < min(0., log_ratio)
            if accepted:
                self.positions, self.auxiliary_positions = canonical.copy(), auxiliary.copy()
                self.native_energy, self.metric = energy, metric
                self.auxiliary.setPositions(auxiliary*unit.nanometer)
                self.auxiliary.setVelocities(-new_velocity*unit.nanometer/unit.picosecond)
        except (FloatingPointError, mm.OpenMMException, np.linalg.LinAlgError):
            numerical, accepted = True, False

            self.rng.random()
        finally:
            if not accepted:
                self.native.loadCheckpoint(old_native)
                self.auxiliary.loadCheckpoint(old_auxiliary)
                self.positions, self.auxiliary_positions, self.native_energy, self.metric = old_fields
        self.attempted += 1
        self.accepted += int(accepted)
        self.domain_rejections += int(not domain and not numerical)
        self.numerical_rejections += int(numerical)
        if check_budget is not None:
            check_budget()
        return {"accepted": accepted, "domain_rejection": not domain and not numerical,
                "numerical_rejection": numerical, "log_acceptance_ratio": float(log_ratio) if math.isfinite(log_ratio) else None,
                "native_energy_kj_mol": self.native_energy, "log_metric_ratio": self.metric,
                "proposed_delta_energy_kj_mol": delta_energy, "proposed_delta_kinetic_kj_mol": delta_kinetic,
                "md_steps_attempted": self.steps_per_segment}

    def reversibility_check(self, *, seed=1024, position_tolerance_nm=1e-8,
                            velocity_tolerance_nm_ps=1e-6):
        native, auxiliary = self.native.createCheckpoint(), self.auxiliary.createCheckpoint()
        try:
            generator = np.random.default_rng(seed)
            velocity = generator.standard_normal(self.auxiliary_positions.shape)*math.sqrt(self.kT/self.bridge.auxiliary_mass)
            velocity -= velocity.mean(0)
            self.auxiliary.setVelocities(velocity*unit.nanometer/unit.picosecond)
            self.auxiliary.applyVelocityConstraints(1e-12)
            original_velocity = self.auxiliary.getState(getVelocities=True).getVelocities(asNumpy=True).value_in_unit(unit.nanometer/unit.picosecond)
            self.integrator.step(self.steps_per_segment)
            final_velocity = self.auxiliary.getState(getVelocities=True).getVelocities(asNumpy=True).value_in_unit(unit.nanometer/unit.picosecond)
            self.auxiliary.setVelocities(-final_velocity*unit.nanometer/unit.picosecond)
            self.integrator.step(self.steps_per_segment)
            reverse_positions = _positions(self.auxiliary)
            reverse_velocity = self.auxiliary.getState(getVelocities=True).getVelocities(asNumpy=True).value_in_unit(unit.nanometer/unit.picosecond)
            position_error = float(np.max(np.abs(reverse_positions-self.auxiliary_positions)))
            velocity_error = float(np.max(np.abs(reverse_velocity+original_velocity)))
            constraint_error = self._constraint_error(reverse_positions, auxiliary=True)
            return {"passed": position_error <= position_tolerance_nm and velocity_error <= velocity_tolerance_nm_ps
                              and constraint_error <= self.constraint_tolerance_nm,
                    "position_error_nm": position_error, "velocity_error_nm_ps": velocity_error,
                    "constraint_error_nm": constraint_error,
                    "position_tolerance_nm": position_tolerance_nm,
                    "velocity_tolerance_nm_ps": velocity_tolerance_nm_ps,
                    "md_steps_charged": 2*self.steps_per_segment}
        finally:
            self.native.loadCheckpoint(native)
            self.auxiliary.loadCheckpoint(auxiliary)

    def state_dict(self):
        return {"format_version": 1, "system": self.system.name, "endpoint": self.endpoint,
                "platform": self.platform_name, "steps_per_segment": self.steps_per_segment,
                "timestep_fs": self.timestep_fs, "constraint_tolerance_nm": self.constraint_tolerance_nm,
                "native_xml": self.native_xml, "auxiliary_xml": self.auxiliary_xml,
                "native_checkpoint": self.native.createCheckpoint(), "auxiliary_checkpoint": self.auxiliary.createCheckpoint(),
                "positions": self.positions.copy(), "auxiliary_positions": self.auxiliary_positions.copy(),
                "native_energy": self.native_energy, "metric": self.metric,
                "numpy_rng": copy.deepcopy(self.rng.bit_generator.state), "initialization": self.initialization,
                "attempted": self.attempted, "accepted": self.accepted,
                "domain_rejections": self.domain_rejections, "numerical_rejections": self.numerical_rejections,
                "openmm_version": mm.version.full_version}

    def load_state_dict(self, state):
        keys = ("system", "endpoint", "platform", "steps_per_segment", "timestep_fs", "constraint_tolerance_nm", "native_xml", "auxiliary_xml", "openmm_version")
        actual = self.state_dict()
        if state.get("format_version") != 1 or any(state.get(key) != actual[key] for key in keys):
            raise ValueError("MD 恢复的物理体系、平台、版本或积分配置发生变化")
        self.native.loadCheckpoint(state["native_checkpoint"])
        self.auxiliary.loadCheckpoint(state["auxiliary_checkpoint"])
        native, auxiliary = _positions(self.native), _positions(self.auxiliary)
        if not np.array_equal(native, state["positions"]) or not np.array_equal(auxiliary, state["auxiliary_positions"]):
            raise ValueError("双 context 检查点位置与保存的 canonical 状态不一致")
        if not self._in_domain(native) or max(self._constraint_error(native), self._constraint_error(auxiliary, auxiliary=True)) > self.constraint_tolerance_nm:
            raise ValueError("恢复状态不在严格约束的全 L 型域")
        bridge_native = _recenter(self.bridge.to_native(auxiliary), self.bridge.masses)
        if np.max(np.abs(bridge_native-native)) > 1e-12:
            raise ValueError("恢复的原生与辅助 context 不表示同一构型")
        energy, metric = _energy(self.native), float(self.bridge.log_metric_ratio(native))
        if abs(energy-state["native_energy"]) > 1e-9 or abs(metric-state["metric"]) > 1e-12:
            raise ValueError("恢复后的能量或质量测度不一致")
        self.positions, self.auxiliary_positions = native.copy(), auxiliary.copy()
        self.native_energy, self.metric = energy, metric
        self.rng.bit_generator.state = copy.deepcopy(state["numpy_rng"])
        self.initialization = state["initialization"]
        for key in ("attempted", "accepted", "domain_rejections", "numerical_rejections"):
            setattr(self, key, int(state[key]))

    def save(self, path):
        save_checkpoint(path, self.state_dict())

    @classmethod
    def load(cls, system, path):
        state = torch.load(path, weights_only=False)
        kernel = cls(system, state["endpoint"], 0, steps_per_segment=state["steps_per_segment"],
                     timestep_fs=state["timestep_fs"], platform=state["platform"],
                     constraint_tolerance_nm=state["constraint_tolerance_nm"], initial_positions=state["positions"])
        kernel.load_state_dict(state)
        return kernel

    def close(self):
        self.native = self.auxiliary = None
        self.native_integrator = self.integrator = None


def _jsonable(value):
    if isinstance(value, torch.Tensor):
        return _jsonable(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return _jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _md_settings(config):
    settings = dict(config.get("endpoint_md", {}))
    allocation = tuple(settings.get("allocation", [1, 1]))
    if allocation not in ((1, 3), (1, 1), (3, 1)):
        raise ValueError("D 端预算分配只接受预定的 1:3、1:1、3:1")
    if "endpoint_counts" in settings:
        counts = list(settings["endpoint_counts"])
    else:
        total = settings.get("total_saved_per_ladder", 2048)
        if type(total) is not int or total <= 0 or total % sum(allocation):
            raise ValueError("总保存量须为固定分配的正整数倍")
        counts = [total*value//sum(allocation) for value in allocation]
    if (len(counts) != 2 or any(type(value) is not int or value < 1 for value in counts)
            or counts[0]*allocation[1] != counts[1]*allocation[0]):
        raise ValueError("端态固定保存量必须与 D 冻结的预算分配一致")
    warmup = settings.get("warmup", 1000)
    warmup = [warmup, warmup] if type(warmup) is int else list(warmup)
    if len(warmup) != 2 or any(type(value) is not int or value < 0 for value in warmup):
        raise ValueError("端态预热量无效")
    settings.update(allocation=list(allocation), endpoint_counts=counts, warmup=warmup,
                    ladders=settings.get("ladders", 4), checkpoint_every=settings.get("checkpoint_every", 10),
                    steps_per_segment=settings.get("steps_per_segment", 100),
                    timestep_fs=settings.get("timestep_fs", 1.), platform=settings.get("platform", "CPU"),
                    constraint_tolerance_nm=settings.get("constraint_tolerance_nm", 1e-10))
    if type(settings["ladders"]) is not int or settings["ladders"] < 4:
        raise ValueError("每个端态至少需要四条独立 MD 链")
    if type(settings["checkpoint_every"]) is not int or settings["checkpoint_every"] < 1:
        raise ValueError("检查点间隔必须为正整数")
    if settings["platform"] not in ("CPU", "Reference", "CUDA"):
        raise ValueError("原生 MD 平台只能为 CPU、Reference 或 CUDA")
    if type(settings["steps_per_segment"]) is not int or settings["steps_per_segment"] < 1:
        raise ValueError("每段 MD 步数必须是正整数")
    if not math.isfinite(settings["timestep_fs"]) or settings["timestep_fs"] <= 0:
        raise ValueError("原生 MD 步长必须有限且为正")
    if not 0 < settings["constraint_tolerance_nm"] <= 1e-8:
        raise ValueError("严格约束入口容差必须在 (0,1e-8] nm")
    if type(settings.get("project_initialization", False)) is not bool:
        raise ValueError("初始化投影开关必须是显式布尔值")
    return settings


def _md_readout(system, records, floor):
    raw = getattr(system, "system", system)
    probabilities, counts, regions_by_endpoint = {}, {}, {}
    for endpoint in (0, 1):
        points = np.stack([row["positions"] for row in records[endpoint]])
        regions = raw.region_positions(torch.tensor(points, dtype=torch.float64))
        if not bool(regions["valid"].all()):
            raise FloatingPointError("MD 原生区域标签不确定，完整主批失败")
        regions_by_endpoint[endpoint] = regions
        for region in ("a", "b"):
            mask = regions[region]
            key = f"{endpoint}_{region}"
            probabilities[key], counts[key] = float(mask.to(torch.float64).mean()), int(mask.sum())
    native_defined = all(value > 0 for value in probabilities.values())
    def odds(floor_value):
        return math.log(max(probabilities["1_b"], floor_value))-math.log(max(probabilities["1_a"], floor_value))-math.log(max(probabilities["0_b"], floor_value))+math.log(max(probabilities["0_a"], floor_value))
    factor = -R_KJ*300 if system.name == "ALA4" else 1.
    return {"preference": factor*odds(floor), "native_preference": factor*odds(0.) if native_defined else None,
            "native_defined": native_defined, "endpoint_populations": probabilities,
            "raw_endpoint_counts": counts, "reporting_floor": floor,
            "floor_triggered": any(value < floor for value in probabilities.values()),
            "native_zero_hit": not native_defined,
            "units": "kJ/mol" if system.name == "ALA4" else "dimensionless"}, regions_by_endpoint


def _qualify_md(records, regions, settings, config):
    from .diagnostics import chain_diagnostics
    criteria = config.get("endpoint_md_qualification", config.get("qualification", {}))
    options = {"rhat_limit": float(criteria.get("rhat_limit", 1.01)),
               "minimum_ess": float(criteria.get("minimum_ess", 400.))}
    diagnostics, reasons = {}, []
    for endpoint in (0, 1):
        energy = np.asarray([row["energies"] for row in records[endpoint]])
        diagnostics[f"endpoint/{endpoint}/native_energy"] = chain_diagnostics(energy, **options)
        for region in ("a", "b"):
            diagnostics[f"endpoint/{endpoint}/indicator/{region}"] = chain_diagnostics(regions[endpoint][region], binary=True, **options)
        angles = regions[endpoint]["angles"].detach().cpu().numpy()
        for angle in range(angles.shape[-1]):
            radians = np.deg2rad(angles[..., angle])
            for name, transform in (("sin", np.sin), ("cos", np.cos)):
                diagnostics[f"endpoint/{endpoint}/angle/{angle}/{name}"] = chain_diagnostics(transform(radians), **options)
        if settings["warmup"][endpoint] == 0:
            reasons.append(f"端点 {endpoint} 未经过冻结预热")
        for row in records[endpoint]:
            if not row["validation"].get("passed", False):
                reasons.append("至少一条 MD 链未通过真实约束核正逆检查")
            if row["kernel_state"]["numerical_rejections"]:
                reasons.append("至少一条 MD 链有数值提议失败，须在 D 核对后再授予资格")
    if config.get("mode", "development") != "formal":
        reasons.append("开发／smoke 运行不取得正式端态资格")
    if criteria.get("warmup_selection") != "D_frozen":
        reasons.append("未登记 D 冻结的预热选择")
    failed = [key for key, row in diagnostics.items() if row["status"] != "passed"]
    eligible = not reasons and not failed
    return {"status": "passed_conditional_diagnostics" if eligible else "unresolved", "eligible_for_formal": eligible,
            "diagnostics": diagnostics, "reasons": reasons, "nonpassing_diagnostics": failed}


def run_endpoint_md(config, output, resume=False, preparation=None, check_budget=None):

    entered = time.monotonic()
    config = read_json(config) if not isinstance(config, dict) else copy.deepcopy(config)
    if config.get("molecule", "ALA4") != "ALA4":
        raise ValueError("直接端态 MD 仅支持 52 原子 ALA4 的构象偏好")
    if config.get("mode", "development") not in ("development", "formal"):
        raise ValueError("原生 MD 运行模式只能为 development 或 formal")
    if any(type(config.get(key, default)) is not int or config.get(key, default) < 0 for key, default in (("seed", 1024), ("replica", 0))):
        raise ValueError("主种子和独立重复编号必须为非负整数")
    settings = _md_settings(config)
    preparation = {} if preparation is None else preparation
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=True)
    contract = {"config": config, "endpoint_md": settings, "method": "Metropolized constrained endpoint MD"}
    with run_lock(directory):
        config_path = directory/"endpoint-md-config.json"
        if config_path.exists():
            if not resume or read_json(config_path) != contract:
                raise ValueError("MD 恢复配置不同，或未显式请求 resume")
        elif resume:
            raise FileNotFoundError("没有可恢复的 MD 运行")
        else:
            write_json(config_path, contract)
        result_path = directory/"endpoint-md-result.json"
        if resume and result_path.exists():
            return read_json(result_path)
        cost_path = directory/"endpoint-md-cost.json"
        previous = read_json(cost_path) if resume and cost_path.exists() else {"wall_seconds": float(preparation.get("preparation_seconds", 0.))}
        if preparation.get("budget") is None:
            previous["wall_seconds"] = float(previous.get("wall_seconds", 0.)) + time.monotonic()-entered
        budget = preparation.get("budget") or Budget(float(config.get("budget_seconds", 14400)), previous)
        def check():
            if check_budget is not None:
                check_budget()
            budget.check()
        records = {0: [], 1: []}
        fallback = preparation.get("fallback")
        fallback_path = directory/"paid-fallback.pt"
        if fallback_path.exists():
            saved = torch.load(fallback_path, weights_only=False)
            if fallback is not None and _jsonable(saved["readout"]) != _jsonable(fallback):
                raise ValueError("恢复时不能更换已付费的后备结果")
        else:
            saved = {"readout": fallback, "qualified": bool(preparation.get("fallback_qualified", False)),
                     "within_same_budget": bool(preparation.get("fallback_within_same_budget", preparation.get("budget") is not None))}
            save_checkpoint(fallback_path, saved)
        floor = 1e-4/float(config.get("budget_ratio", 1.))
        if not 0 < floor < 1:
            raise ValueError("端态 MD 的共同报告地板无效")
        kernel, current, checkpoint_path = None, None, None
        result = {"method": contract["method"],
                  "primary_complete": False, "eligible_for_formal": False, "fallback_triggered": False,
                  "delta_free_energy": None, "free_energy_status": "not_an_estimand_of_endpoint_MD",
                  "allocation": settings["allocation"], "fixed_endpoint_counts_per_ladder": settings["endpoint_counts"],
                  "ladders_per_endpoint": settings["ladders"], "mode": config.get("mode", "development")}
        try:
            check()
            system = preparation.get("system") or MolecularSystem("ALA4")
            raw = getattr(system, "system", system)
            for endpoint in (0, 1):
                for ladder in range(settings["ladders"]):
                    checkpoint_path = directory/f"endpoint-{endpoint}-ladder-{ladder}.pt"
                    if resume and checkpoint_path.exists():
                        current = torch.load(checkpoint_path, weights_only=False)
                    else:
                        current = {"completed_warmup": 0, "positions": [], "energies": [], "segments": [], "validation": {}}
                    expected_unit = f"ALA4:endpoint-MD:{config.get('seed', 1024)}:{config.get('replica', 0)}:{endpoint}:{ladder}"
                    if "unit_id" in current and current["unit_id"] != expected_unit:
                        raise ValueError("恢复的 MD 独立链身份发生变化")
                    current["unit_id"] = expected_unit
                    if len(current["positions"]) == settings["endpoint_counts"][endpoint] and current["completed_warmup"] == settings["warmup"][endpoint]:
                        records[endpoint].append(current)
                        continue
                    seed = int(np.random.SeedSequence([int(config.get("seed", 1024)), 8753, int(config.get("replica", 0)), endpoint, ladder]).generate_state(1)[0])
                    kernel = MetropolizedEndpointMD(system, endpoint, seed, steps_per_segment=settings["steps_per_segment"],
                                timestep_fs=settings["timestep_fs"], platform=settings["platform"],
                                constraint_tolerance_nm=settings["constraint_tolerance_nm"],
                                initial_positions=raw.initial_positions[ladder % len(raw.initial_positions)].detach().cpu().numpy(),
                                project_initialization=bool(settings.get("project_initialization", False)))
                    if "kernel_state" in current:
                        kernel.load_state_dict(current["kernel_state"])
                    else:
                        current["validation"] = kernel.reversibility_check(seed=seed)
                        budget.count("endpoint_md_validation_steps", current["validation"]["md_steps_charged"])
                        if not current["validation"]["passed"]:
                            raise FloatingPointError("真实 MD 段正逆核检查未通过")
                    while current["completed_warmup"] < settings["warmup"][endpoint] or len(current["positions"]) < settings["endpoint_counts"][endpoint]:
                        check()
                        segment = kernel.step()
                        budget.count("endpoint_md_steps", settings["steps_per_segment"])
                        current["segments"].append(segment)
                        if current["completed_warmup"] < settings["warmup"][endpoint]:
                            current["completed_warmup"] += 1
                        else:
                            current["positions"].append(kernel.positions.copy())
                            current["energies"].append(kernel.native_energy)
                        if kernel.attempted % settings["checkpoint_every"] == 0:
                            current["kernel_state"] = kernel.state_dict()
                            save_checkpoint(checkpoint_path, current)
                            write_json(cost_path, _jsonable(budget.record()))
                        check()
                    current["kernel_state"] = kernel.state_dict()
                    save_checkpoint(checkpoint_path, current)
                    records[endpoint].append(current)
                    kernel.close()
                    kernel = None
            check()
            readout, regions = _md_readout(system, records, floor)
            qualification = _qualify_md(records, regions, settings, config)
            result.update(primary_complete=True, native=readout, output=readout,
                          qualification=qualification, eligible_for_formal=qualification["eligible_for_formal"],
                          status="completed" if qualification["eligible_for_formal"] else "completed_development",
                          source_unit_ids=[row["unit_id"] for endpoint in (0, 1) for row in records[endpoint]])
            if config.get("mode") == "formal" and not qualification["eligible_for_formal"]:
                raise ValueError("完整 MD 主批未取得正式平衡／混合资格")
        except (TimeoutError, ValueError, FloatingPointError, mm.OpenMMException) as exc:
            if kernel is not None and current is not None and checkpoint_path is not None:
                current["kernel_state"] = kernel.state_dict()
                save_checkpoint(checkpoint_path, current)
            result.update(status="no_legal_output", reason=str(exc), output=None,
                          primary_batch_discarded=True, fallback_triggered=True)
            fallback_value = saved["readout"]
            valid_fallback = (saved["within_same_budget"] and fallback_value is not None
                              and math.isfinite(float(fallback_value.get("preference", math.nan)))
                              and (config.get("mode") != "formal" or saved["qualified"])
                              and math.isclose(float(fallback_value.get("reporting_floor", floor)), floor))
            if valid_fallback:
                result.update(status="paid_identity_fallback", output=fallback_value,
                              eligible_for_formal=bool(saved["qualified"]), fallback_source="paid_same_budget_identity_FEP")
        except BaseException:
            if kernel is not None and current is not None and checkpoint_path is not None:
                current["kernel_state"] = kernel.state_dict()
                save_checkpoint(checkpoint_path, current)
            write_json(cost_path, _jsonable(budget.record()))
            raise
        finally:
            if kernel is not None:
                kernel.close()
        result["cost"] = budget.record()
        result["estimator_scope"] = "endpoint_basin_counts"
        write_json(cost_path, _jsonable(result["cost"]))
        write_json(result_path, _jsonable(result))
        return _jsonable(result)
