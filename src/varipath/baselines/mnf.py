from dataclasses import asdict, dataclass
import math
from pathlib import Path

import numpy as np
from scipy.interpolate import BSpline
import torch
from torch.nn import functional as F

from ..coordinates import dihedral_radians
from .bridge import _classify, bar, bridge_readout
from .feat import _atomic_save, _pool


@dataclass(frozen=True)
class MNFConfig:
    iterations: int = 500
    batch_size: int = 128
    learning_rate: float = 0.001
    seed: int = 1327200
    checkpoint_every: int = 10


def distance_loss(kind, forward, reverse, free_energy):
    if kind == "bd":
        a, b = .5*(free_energy-forward), .5*(-reverse-free_energy)
    elif kind == "hd":
        a = math.log(2)-F.softplus(forward-free_energy)
        b = math.log(2)-F.softplus(reverse+free_energy)
    else:
        raise ValueError("MNF仅使用已核对的BD和HD原生损失")
    return -.5*(torch.logsumexp(a, 0)-math.log(len(a))+torch.logsumexp(b, 0)-math.log(len(b)))


class PeriodicMap(torch.nn.Module):

    def __init__(self, dimension):
        super().__init__()
        if dimension < 1:
            raise ValueError("周期映射维数必须为正")
        self.register_buffer("basis", torch.tensor(BSpline.design_matrix((np.arange(8)+.5)/8, [0,0,0,0,.5,1,1,1,1], 3).toarray(), dtype=torch.float64))
        self.coeff = torch.nn.Parameter(torch.zeros(5, dimension, 2, 1+2*(dimension-1), dtype=torch.float64))

    def circle(self, theta, inverse=False):
        logj = torch.zeros_like(theta[..., 0])
        cols = list(theta.unbind(-1))
        coeff = torch.einsum("tk,kdif->tdif", self.basis, self.coeff)
        order = [(t, i) for t in range(len(self.basis)) for i in range(len(cols))]
        for t, i in reversed(order) if inverse else order:
            other = torch.stack(cols[:i]+cols[i+1:], -1) if len(cols)>1 else theta[..., :0]
            feat = torch.cat([torch.ones_like(theta[..., :1]), other.cos(), other.sin()], -1)
            scale, shift = (torch.tanh(feat@coeff[t, i].T)/len(self.basis)).unbind(-1)
            if inverse:
                scale = -scale
            angle = (cols[i]-shift if inverse else cols[i])/2
            numer, denom = scale.exp()*angle.sin(), angle.cos()
            cols[i] = 2*torch.atan2(numer, denom)+(0 if inverse else shift)
            logj = logj+scale-(numer.square()+denom.square()).log()
        return torch.stack(cols, -1), logj

    def map(self, points, inverse=False):
        return self.circle(points, inverse)


class MNFIC(PeriodicMap):
    def __init__(self, system, config=None):
        self.config = config or MNFConfig()
        c = self.config
        if c.iterations < 2 or c.iterations % 2 or min(c.batch_size, c.checkpoint_every) < 1 or c.learning_rate <= 0:
            raise ValueError("MNF需要正批量和等长BD/HD两阶段")
        super().__init__(len(system.torsion_quads))
        self.system = system
        self.register_buffer("quads", torch.tensor(system.torsion_quads, dtype=torch.long))
        adjacency = [set() for _ in system.metadata["atoms"]]
        for a, b in system.bonds:
            adjacency[a].add(b); adjacency[b].add(a)
        self.rotors = []
        for q in system.torsion_quads:
            left, right = q[1:3]
            selected, pending = set(), [right]
            while pending:
                at = pending.pop()
                if at in selected:
                    continue
                selected.add(at)
                pending.extend(j for j in adjacency[at] if {at, j}!={left, right})
            if left in selected or q[3] not in selected or q[0] in selected:
                raise ValueError("指定主链扭转不是可分离的树形内坐标")
            self.rotors.append(sorted(selected))

    def _encode_unchecked(self, x):

        chart = self.system.chart
        origin, first, second = chart.triad
        delta = x-x[origin]
        length = torch.linalg.vector_norm(delta[first])
        ex = delta[first]/length
        perp = delta[second]-(delta[second]*ex).sum()*ex
        height = torch.linalg.vector_norm(perp)
        ey = perp/height
        axes = torch.stack([ex, ey, torch.linalg.cross(ex, ey)])
        body = delta@axes.T
        directions = (body[chart.hydrogens]-body[chart.parents])/chart.lengths[:, None]
        transverse2 = directions[:, :2].square().sum(-1)
        north = (transverse2==0)&(directions[:, 2]>=0)
        v = torch.where(directions[:, 2]<0, transverse2/(2*(1-directions[:, 2])), (1+directions[:, 2])/2)
        radius = torch.sqrt(-2*torch.log(v.clamp(max=1)))
        denominator = torch.sqrt(torch.where(north, torch.ones_like(transverse2), transverse2))
        u = torch.where(north[:, None], torch.zeros_like(directions[:, :2]), radius[:, None]*directions[:, :2]/denominator[:, None])
        root = torch.stack([length.log(), body[second, 0], height.log()])
        result = torch.cat([root, body[chart.heavy].flatten(), u.flatten()])
        if hasattr(self.system, "raw"):
            result = (result-self.system.location)/self.system.scale
        return result

    def _one_map(self, z, inverse=False):
        x = self.system.positions(z)
        old = dihedral_radians(x[self.quads])
        new, _ = self.circle(old, inverse)
        for j, (quad, ids) in enumerate(zip(self.quads, self.rotors)):
            angle = new[j]-old[j]
            axis = x[quad[2]]-x[quad[1]]
            axis = axis/torch.linalg.vector_norm(axis)
            v = x[ids]-x[quad[1]]
            arm = v-(v*axis).sum(-1, keepdim=True)*axis
            move = (angle.cos()-1)*arm+angle.sin()*torch.linalg.cross(axis.expand_as(arm), arm)
            x = x.index_add(0, torch.as_tensor(ids, dtype=torch.long, device=x.device), move)
        return self._encode_unchecked(x)

    def map(self, points, inverse=False, check_budget=None):
        if check_budget is not None:
            check_budget()
        if points.ndim == 1:
            y, ld = self.map(points[None], inverse, check_budget=check_budget)
            return y[0], ld[0]
        _pool(points, self.system.dim)
        if not self.system.in_domain(points).all():
            raise ValueError("MNF输入不属于全L物理域")
        function = lambda z: self._one_map(z, inverse)
        moved = torch.vmap(function)(points)

        matrices = torch.vmap(torch.func.jacrev(function))(points)
        sign, logdet = torch.linalg.slogdet(matrices)
        if not torch.isfinite(moved).all() or not torch.isfinite(logdet).all() or not (sign>0).all():
            raise FloatingPointError("MNF实际分子图不正则或未保持方向")
        if not self.system.in_domain(moved).all():
            raise ValueError("MNF扭转映射离开全L物理域")
        if check_budget is not None:
            check_budget()
        return moved, logdet

    def works(self, source, target, check_budget=None):
        forward, ld = self.map(source, check_budget=check_budget)
        reverse, ild = self.map(target, True, check_budget=check_budget)
        u = lambda x, endpoint: torch.vmap(lambda z: self.system.effective_u(z, endpoint))(x)
        wf = u(forward, 1)-u(source, 0)-ld
        wr = u(reverse, 0)-u(target, 1)-ild
        return wf, wr, forward, reverse

    def batched_works(self, source, target, batch_size=32, check_budget=None):
        if type(batch_size) is not int or batch_size<1:
            raise ValueError("实际计算批量必须为正整数")
        _pool(source,self.system.dim); _pool(target,self.system.dim)
        def one_direction(points,inverse):
            values=[];works=[]
            for start in range(0,len(points),batch_size):
                batch=points[start:start+batch_size]
                moved,logdet=self.map(batch,inverse,check_budget=check_budget)
                initial_endpoint=1 if inverse else 0
                final_endpoint=0 if inverse else 1
                initial_u=torch.vmap(lambda z:self.system.effective_u(z,initial_endpoint))(batch)
                final_u=torch.vmap(lambda z:self.system.effective_u(z,final_endpoint))(moved)
                works.append(final_u-initial_u-logdet);values.append(moved)
            return torch.cat(works),torch.cat(values)
        wf,forward=one_direction(source,False)
        wr,reverse=one_direction(target,True)
        return wf,wr,forward,reverse

    def fit(self, source, target, checkpoint=None, resume=False, stop_after=None, check_budget=None):
        _pool(source, self.system.dim); _pool(target, self.system.dim)
        c = self.config
        first, history, free_energy, initial = 0, [], 0., None
        optimizer = torch.optim.Adam([self.coeff], lr=c.learning_rate)
        generator = torch.Generator().manual_seed(c.seed)
        if resume:
            if checkpoint is None or not Path(checkpoint).exists():
                raise FileNotFoundError("MNF恢复需要现有检查点")
            initial = torch.load(checkpoint, weights_only=False)
            if initial["config"]!=asdict(c) or not torch.equal(initial["source"],source) or not torch.equal(initial["target"],target):
                raise ValueError("MNF恢复的完整训练池或配置改变")
            self.load_state_dict(initial["model"])
            optimizer.load_state_dict(initial["optimizer"])
            generator.set_state(initial["rng"])
            first, history, free_energy = initial["iteration"], initial["history"], initial["free_energy"]
        elif checkpoint is not None and Path(checkpoint).exists():
            raise FileExistsError("MNF检查点已存在，请显式恢复")
        if stop_after is not None and stop_after < first:
            raise ValueError("停止迭代早于现有进度")
        last = c.iterations if stop_after is None else min(c.iterations, stop_after)
        for iteration in range(first, last):
            if check_budget is not None:
                check_budget()
            kind = "bd" if iteration<c.iterations//2 else "hd"
            i = torch.randint(len(source), (c.batch_size,), generator=generator)
            j = torch.randint(len(target), (c.batch_size,), generator=generator)
            wf, wr, _, _ = self.works(source[i], target[j], check_budget=check_budget)
            loss = distance_loss(kind, wf, wr, free_energy)
            if not torch.isfinite(loss):
                raise FloatingPointError("MNF原生训练损失非有限")
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_([self.coeff], 1, error_if_nonfinite=True)
            optimizer.step()
            history.append({"iteration":iteration+1,"phase":kind,"loss":float(loss.detach())})
            if iteration+1==c.iterations//2:
                with torch.no_grad():
                    wf, wr, _, _ = self.batched_works(source, target, check_budget=check_budget)
                calibrated = bar(wf.cpu().numpy(), wr.cpu().numpy())
                free_energy = calibrated["free_energy"]

                optimizer = torch.optim.Adam([self.coeff], lr=c.learning_rate)
            if checkpoint is not None and ((iteration+1)%c.checkpoint_every==0 or iteration+1==last):
                if check_budget is not None:
                    check_budget()
                _atomic_save(checkpoint, {"config":asdict(c),"iteration":iteration+1,"model":self.state_dict(),"optimizer":optimizer.state_dict(),"rng":generator.get_state(),"history":history,"free_energy":free_energy,"source":source,"target":target})
        return {"iterations":last,"complete":last==c.iterations,"after_bd_free_energy_for_hd":free_energy,"history":history}

    @torch.no_grad()
    def evaluate(self, source, target, eta=1e-4, min_overlap=1e-8, check_budget=None, batch_size=32):
        wf,wr,forward,reverse=self.batched_works(source,target,batch_size,check_budget)
        if check_budget is not None:
            check_budget()

        result = bar(wf.cpu().numpy(), wr.cpu().numpy(), min_overlap)
        fp, rp = torch.stack([source, forward], 1), torch.stack([target, reverse], 1)
        readout = bridge_readout(fp.cpu().numpy(), rp.cpu().numpy(), -wf.cpu().numpy(), -wr.cpu().numpy(), result["log_partition_ratio"], lambda points: _classify(self.system, points), eta)
        if getattr(self.system,"name",None)=="ALDP":
            readout["task_log_odds_difference"]=-readout["reported_dimensionless_preference"]
        return {"native_bar":result,"readout":readout,"forward_work":wf,"reverse_work":wr,"forward":forward,"reverse":reverse}


class PeriodicMNF(MNFIC):

    def __init__(self, system, config=None):
        PeriodicMap.__init__(self, system.dim)
        self.system, self.config = system, config or MNFConfig()
        c = self.config
        if c.iterations<2 or c.iterations%2 or min(c.batch_size,c.checkpoint_every)<1 or c.learning_rate<=0:
            raise ValueError("MNF需要正批量和等长BD/HD两阶段")

    def map(self, points, inverse=False, check_budget=None):
        if check_budget is not None:
            check_budget()
        _pool(points, self.system.dim)
        return self.circle(points, inverse)
