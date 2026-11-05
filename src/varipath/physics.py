from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import openmm as mm
from openmm import unit
import torch
from torch import nn
from .coordinates import BodyCoordinates, dihedral_radians

COULOMB = 138.93545764438198
GB_COULOMB = 138.935485
R_KJ = 0.00831446261815324
ASSETS = Path(__file__).with_name("assets")

class MolecularEnergy(nn.Module):
    def __init__(self, system_xml, gb_kind):
        super().__init__()
        self.gb_kind = gb_kind
        system = mm.XmlSerializer.deserialize(Path(system_xml).read_text())
        self.n_atoms = system.getNumParticles()
        forces = {type(f).__name__: f for f in system.getForces()}
        if set(forces) != {"HarmonicBondForce", "HarmonicAngleForce", "PeriodicTorsionForce", "NonbondedForce", "CustomGBForce"}:
            raise ValueError("System 包含不支持的力项")
        bonds, angles, torsions = [], [], []
        f = forces["HarmonicBondForce"]
        for i in range(f.getNumBonds()):
            a, b, length, k = f.getBondParameters(i)
            bonds.append((a, b, length.value_in_unit(unit.nanometer), k.value_in_unit(unit.kilojoule_per_mole/unit.nanometer**2)))
        f = forces["HarmonicAngleForce"]
        for i in range(f.getNumAngles()):
            a, b, c, angle, k = f.getAngleParameters(i)
            angles.append((a, b, c, angle.value_in_unit(unit.radian), k.value_in_unit(unit.kilojoule_per_mole/unit.radian**2)))
        f = forces["PeriodicTorsionForce"]
        for i in range(f.getNumTorsions()):
            a, b, c, d, n, phase, k = f.getTorsionParameters(i)
            torsions.append((a, b, c, d, n, phase.value_in_unit(unit.radian), k.value_in_unit(unit.kilojoule_per_mole)))
        for name, rows, nindex in [("bond", bonds, 2), ("angle", angles, 3), ("torsion", torsions, 4)]:
            a = np.asarray(rows)
            self.register_buffer(name+"_index", torch.tensor(a[:, :nindex], dtype=torch.long))
            self.register_buffer(name+"_parameters", torch.tensor(a[:, nindex:], dtype=torch.float64))
        f = forces["NonbondedForce"]
        if f.getNonbondedMethod() != mm.NonbondedForce.NoCutoff or f.getNumParticleParameterOffsets() or f.getNumExceptionParameterOffsets():
            raise ValueError("冻结分子势必须是无偏移的NoCutoff定义。")
        particles = []
        for i in range(self.n_atoms):
            q, sigma, eps = f.getParticleParameters(i)
            particles.append((q.value_in_unit(unit.elementary_charge), sigma.value_in_unit(unit.nanometer), eps.value_in_unit(unit.kilojoule_per_mole)))
        exceptions = {}
        for k in range(f.getNumExceptions()):
            i, j, qp, sig, eps = f.getExceptionParameters(k)
            exceptions[tuple(sorted((i, j)))] = (qp.value_in_unit(unit.elementary_charge**2), sig.value_in_unit(unit.nanometer), eps.value_in_unit(unit.kilojoule_per_mole))
        pairs, parameters = [], []
        for i in range(self.n_atoms):
            for j in range(i+1, self.n_atoms):
                if (i, j) in exceptions:
                    values = exceptions[i, j]
                else:
                    qi, si, ei = particles[i]
                    qj, sj, ej = particles[j]
                    values = qi*qj, (si+sj)/2, np.sqrt(ei*ej)
                if values[0] != 0 or values[2] != 0:
                    pairs.append((i, j)); parameters.append(values)
        self.register_buffer("nonbonded_index", torch.tensor(pairs, dtype=torch.long))
        self.register_buffer("nonbonded_parameters", torch.tensor(parameters, dtype=torch.float64))
        gb = forces["CustomGBForce"]
        gb_names = [gb.getPerParticleParameterName(i) for i in range(gb.getNumPerParticleParameters())]
        gb_values = np.asarray([gb.getParticleParameters(i) for i in range(self.n_atoms)])
        for i, name in enumerate(gb_names):
            self.register_buffer("gb_"+name, torch.tensor(gb_values[:, i], dtype=torch.float64))
        if gb_kind not in ("GBn2", "OBC1", "OBC2"):
            raise ValueError("不支持的 GB 模型")
        if ("radindex" in gb_names) != (gb_kind == "GBn2"):
            raise ValueError("GB 参数与模型不一致")
        self.offset = 0.0195141 if self.gb_kind == "GBn2" else .009
        if self.gb_kind == "GBn2":
            indices = self.gb_radindex.to(torch.long)
            for k in range(gb.getNumTabulatedFunctions()):
                fn = gb.getTabulatedFunction(k)
                if not isinstance(fn, mm.Discrete2DFunction):
                    raise ValueError("GBn2表类型改变，需独立核对。")
                xs, ys, values = fn.getFunctionParameters()
                table = torch.tensor(values, dtype=torch.float64).reshape(ys, xs)
                self.register_buffer("gb_"+gb.getTabulatedFunctionName(k), table[indices[None, :], indices[:, None]])
        self.register_buffer("off_diagonal", ~torch.eye(self.n_atoms, dtype=torch.bool))
        self.register_buffer("diagonal", torch.eye(self.n_atoms, dtype=torch.float64))

    def base_energy(self, x):
        bi, bp = self.bond_index, self.bond_parameters
        r = torch.linalg.vector_norm(x[bi[:, 0]]-x[bi[:, 1]], dim=-1)
        bond = (.5*bp[:, 1]*(r-bp[:, 0]).square()).sum()
        ai, ap = self.angle_index, self.angle_parameters
        u, v = x[ai[:, 0]]-x[ai[:, 1]], x[ai[:, 2]]-x[ai[:, 1]]
        theta = torch.atan2(torch.linalg.vector_norm(torch.linalg.cross(u, v), dim=-1), (u*v).sum(-1))
        angle = (.5*ap[:, 1]*(theta-ap[:, 0]).square()).sum()
        ti, tp = self.torsion_index, self.torsion_parameters
        torsion = (tp[:, 2]*(1+torch.cos(tp[:, 0]*dihedral_radians(x[ti])-tp[:, 1]))).sum()
        ni, np_ = self.nonbonded_index, self.nonbonded_parameters
        distance = torch.linalg.vector_norm(x[ni[:, 0]]-x[ni[:, 1]], dim=-1)
        sr6 = (np_[:, 1]/distance).pow(6)
        nonbonded = (COULOMB*np_[:, 0]/distance+4*np_[:, 2]*(sr6.square()-sr6)).sum()
        return bond+angle+torsion+nonbonded

    def gb_components(self, x):
        delta = x[:, None, :]-x[None, :, :]

        r = torch.sqrt(delta.square().sum(-1)+self.diagonal)
        or1, sr2 = self.gb_or[:, None], self.gb_sr[None, :]
        lower = torch.maximum(or1, (r-sr2).abs())
        upper = r+sr2
        term = .5*(1/lower-1/upper+.25*(r-sr2.square()/r)*(1/upper.square()-1/lower.square())+.5*torch.log(lower/upper)/r)
        ivdw = torch.where(r+sr2-or1 >= 0, term, torch.zeros_like(term))
        radius = self.gb_or+self.offset
        if self.gb_kind == "GBn2":
            dr = r-self.gb_getd0
            neck = self.gb_getm0/(1+100*dr.square()+300000*dr.pow(6))
            neck = torch.where(radius[:, None]+radius[None, :]+.68-r >= 0, neck, torch.zeros_like(neck))
            integral = ((ivdw+.826836*neck)*self.off_diagonal).sum(-1)
            psi = integral*self.gb_or
            argument = self.gb_alpha*psi-self.gb_beta*psi.square()+self.gb_gamma*psi.pow(3)
        else:
            integral = (ivdw*self.off_diagonal).sum(-1)
            psi = integral*self.gb_or
            argument = .8*psi+2.909125*psi.pow(3) if self.gb_kind == "OBC1" else psi-.8*psi.square()+4.85*psi.pow(3)
        born = 1/(1/self.gb_or-torch.tanh(argument)/radius)
        charge = self.gb_charge
        coefficient = GB_COULOMB*(1-1/78.5)
        self_energy = (-.5*coefficient*charge.square()/born).sum()
        product = born[:, None]*born[None, :]
        effective_distance = torch.sqrt(r.square()+product*torch.exp(-r.square()/(4*product)))
        pair = -.5*coefficient*((charge[:, None]*charge[None, :]/effective_distance)*self.off_diagonal).sum()
        ace = (28.3919551*(radius+.14).square()*(radius/born).pow(6)).sum()
        return ace, self_energy+pair

    def forward(self, x, s=1.0):
        ace, polar = self.gb_components(x)
        factor = s
        return self.base_energy(x)+ace+factor*polar


class MolecularSystem(nn.Module):
    def __init__(self, name):
        super().__init__()
        if name not in ("ALDP", "ALA4"):
            raise ValueError("分子体系必须是 ALDP 或 ALA4")
        self.name = name
        self.asset_path = ASSETS/name
        self.metadata = json.loads((self.asset_path/"metadata.json").read_text())
        self.chart = BodyCoordinates(self.metadata)
        self.dim = self.chart.dim
        self.energy = MolecularEnergy(self.asset_path/"system.xml",self.metadata["gb_kind"])
        self.triad = self.chart.triad
        self.bonds = [tuple(pair) for pair in self.metadata["bonds"]]
        self.torsion_quads = self.metadata["dihedral_quads"]
        self.register_buffer("quads",torch.tensor(self.torsion_quads,dtype=torch.long))
        self.register_buffer("chiral_quads",torch.tensor(self.metadata["chiral_quads"],dtype=torch.long))
        self.register_buffer("chiral_signs",torch.tensor(self.metadata["chiral_signs"],dtype=torch.float64))
        self.register_buffer("initial_positions",torch.tensor(self.metadata["initial_positions_nm"],dtype=torch.float64))

    def positions(self,z):
        return self.chart(z)

    def encode(self,x):
        return self.chart.encode(x)

    def log_volume(self,z):
        return self.chart.log_volume(z)

    def initial_coordinates(self,index=0):
        return self.encode(self.initial_positions[index]).clone()

    def in_domain_positions(self,x):
        q = x[...,self.chiral_quads,:]
        vectors = q[...,:3,:]-q[...,3,None,:]
        volume = (torch.linalg.cross(vectors[...,0,:],vectors[...,1,:])*vectors[...,2,:]).sum(-1)
        return ((volume*self.chiral_signs)>0).all(-1)&torch.isfinite(x).all((-1,-2))

    def in_domain(self,z):
        return self.in_domain_positions(self.positions(z))

    def unrestricted_u_state(self,z,state):
        temperature = state if self.name=="ALDP" else 300.
        s = 1. if self.name=="ALDP" else state
        if z.ndim > 1:
            return torch.vmap(lambda q:self.unrestricted_u_state(q,state))(z)
        return self.energy(self.positions(z),s)/(R_KJ*temperature)-self.log_volume(z)

    def effective_u_state(self,z,state):
        value = self.unrestricted_u_state(z,state)
        return torch.where(self.in_domain(z),value,torch.full_like(value,float("inf")))

    def state(self,endpoint):
        if endpoint not in (0,1):
            raise ValueError("端点必须是 0 或 1")
        return (550. if endpoint==0 else 300.) if self.name=="ALDP" else float(endpoint)

    def effective_u(self,z,endpoint):
        return self.effective_u_state(z,self.state(endpoint))

    def region_positions(self,x):
        angles = torch.rad2deg(dihedral_radians(x[...,self.quads,:]))
        pairs = angles.reshape(*angles.shape[:-1],-1,2)
        centers = pairs.new_tensor([[-70.,-40.],[-150.,160.],[-70.,150.],[60.,50.]])
        distance = (pairs[...,None,:]-centers+180.)%360.-180.
        membership = distance.square().sum(-1)<1600.
        labels = torch.where(membership.any(-1),membership.to(torch.long).argmax(-1),4)
        valid = torch.isfinite(angles).all(-1)
        a = (labels==0).all(-1)&valid
        b = ((labels==1)|(labels==2)).all(-1)&valid
        fine = labels[...,0] if self.name=="ALDP" else 5*labels[...,0]+labels[...,1]
        fine = torch.where(valid,fine,torch.full_like(fine,-1))
        return dict(a=a,b=b,remainder=~(a|b)&valid,fine=fine,angles=angles,valid=valid)

    def regions(self,z):
        return self.region_positions(self.positions(z))

    def oracle(self):
        return OpenMMOracle(self)


class AffineSystem(nn.Module):

    def __init__(self,system,location,scale):
        super().__init__()
        self.system = system
        self.dim,self.name,self.metadata = system.dim,system.name,system.metadata
        self.chart,self.triad = system.chart,system.triad
        self.bonds,self.torsion_quads = system.bonds,system.torsion_quads
        self.register_buffer("location",torch.as_tensor(location,dtype=torch.float64).clone())
        self.register_buffer("scale",torch.as_tensor(scale,dtype=torch.float64).clone())
        if self.location.shape!=(self.dim,) or self.scale.shape!=(self.dim,) or not torch.isfinite(self.location).all() or not torch.isfinite(self.scale).all() or (self.scale<=0).any():
            raise ValueError("仿射图需要有限中心和严格正的同维尺度")

    def raw(self,z):
        return self.location+self.scale*z

    def positions(self,z):
        return self.system.positions(self.raw(z))

    def encode(self,x):
        return (self.system.encode(x)-self.location)/self.scale

    def log_volume(self,z):
        return self.system.log_volume(self.raw(z))+self.scale.log().sum()

    def initial_coordinates(self,index=0):
        return (self.system.initial_coordinates(index)-self.location)/self.scale

    def in_domain(self,z):
        return self.system.in_domain(self.raw(z))

    def effective_u(self,z,endpoint):
        return self.system.effective_u(self.raw(z),endpoint)-self.scale.log().sum()

    def effective_u_state(self,z,state):
        return self.system.effective_u_state(self.raw(z),state)-self.scale.log().sum()

    def unrestricted_u_state(self,z,state):
        return self.system.unrestricted_u_state(self.raw(z),state)-self.scale.log().sum()

    def state(self,endpoint):
        return self.system.state(endpoint)

    def regions(self,z):
        return self.system.regions(self.raw(z))


class OpenMMOracle:

    def __init__(self,system):
        self.name = system.name
        self.system = mm.XmlSerializer.deserialize((system.asset_path/"system.xml").read_text())
        self.gb = next(f for f in self.system.getForces() if isinstance(f,mm.CustomGBForce))
        self.original = [list(self.gb.getParticleParameters(i)) for i in range(self.gb.getNumParticles())]
        self.integrator = mm.VerletIntegrator(.001*unit.picosecond)
        self.context = mm.Context(self.system,self.integrator,mm.Platform.getPlatformByName("Reference"))

    def evaluate(self,positions_nm,s=1.):
        if not 0<=s<=1:
            raise ValueError("极性耦合 s 必须位于 [0,1]")
        factor = float(np.sqrt(s)) if self.name=="ALA4" else 1.
        for i,original in enumerate(self.original):
            values = original.copy()
            values[0] *= factor
            self.gb.setParticleParameters(i,values)
        self.gb.updateParametersInContext(self.context)
        self.context.setPositions(np.asarray(positions_nm)*unit.nanometer)
        state = self.context.getState(getEnergy=True,getForces=True)
        return (state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole),state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole/unit.nanometer))

    def close(self):
        self.context = None
        self.integrator = None
