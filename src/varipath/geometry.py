from __future__ import annotations

import math
import torch
from torch import nn


def smoothstep(t):
    t = t.clamp(0, 1)
    return t**3 * (10 + t * (6*t - 15))


class MolecularProtection(nn.Module):


    compact_support = True

    def __init__(self, system, lower, upper, width, temperature,
                 collision_nm=.045, sine_min=.04, chiral_margin=1e-8,
                 neck_exclusion_nm=.010):
        super().__init__()
        self.system = system
        self.dim = system.dim
        lower = torch.as_tensor(lower, dtype=torch.float64)
        upper = torch.as_tensor(upper, dtype=torch.float64)
        if lower.shape != (self.dim,) or upper.shape != lower.shape or not torch.isfinite(lower).all() or not torch.isfinite(upper).all() or (upper <= lower).any():
            raise ValueError("Protection requires a finite nonempty coordinate box")
        values = (width, temperature, collision_nm, sine_min, chiral_margin, neck_exclusion_nm)
        if not all(math.isfinite(v) and v > 0 for v in values) or sine_min >= 1:
            raise ValueError("Safe support margins and smooth-min scales must be positive")
        self.register_buffer("lower", lower)
        self.register_buffer("upper", upper)
        self.width, self.temperature = float(width), float(temperature)
        self.lipschitz = 1.875/self.width
        self.collision_nm, self.sine_min = collision_nm, sine_min
        self.chiral_margin, self.neck_exclusion_nm = chiral_margin, neck_exclusion_nm
        raw = getattr(system, "system", system)
        location = getattr(system, "location", torch.zeros(self.dim, dtype=torch.float64))
        scale = getattr(system, "scale", torch.ones(self.dim, dtype=torch.float64))
        self.register_buffer("location", location.detach().clone())
        self.register_buffer("scale", scale.detach().clone())
        lo, hi = self.location + self.scale*lower, self.location + self.scale*upper
        chart = raw.chart
        n = chart.n_atoms


        B, L = [0.]*n, [0.]*n
        origin, first, second = chart.triad
        B[first], L[first] = math.exp(float(hi[0])), math.exp(float(hi[0]))*float(scale[0])
        B[second] = math.hypot(max(abs(float(lo[1])), abs(float(hi[1]))), math.exp(float(hi[2])))
        L[second] = max(float(scale[1]), math.exp(float(hi[2]))*float(scale[2]))
        for k, atom in enumerate(chart.heavy):
            indices = slice(3+3*k, 6+3*k)
            B[atom] = float(torch.maximum(lo[indices].abs(), hi[indices].abs()).norm())
            L[atom] = float(scale[indices].max())
        for k, (h, p) in enumerate(zip(chart.hydrogens, chart.parents)):
            length = float(chart.lengths[k])
            B[h] = B[p] + length

            L[h] = L[p] + math.sqrt(2)*length*float(scale[chart.split+2*k:chart.split+2*k+2].max())
        self.register_buffer("position_bound", torch.tensor(B, dtype=torch.float64))
        self.register_buffer("position_lipschitz", torch.tensor(L, dtype=torch.float64))
        pairs = torch.triu_indices(n, n, 1).T
        self.register_buffer("pairs", pairs)
        pairL = self.position_lipschitz[pairs].sum(-1)
        self.register_buffer("pair_lipschitz", pairL)
        angles = set()
        for row in raw.energy.angle_index.tolist():
            angles.add(tuple(row))
        for row in raw.energy.torsion_index.tolist():
            angles.update((tuple(row[:3]), tuple(row[1:])))
        self.register_buffer("angle_triples", torch.tensor(sorted(angles), dtype=torch.long))
        a, b, c = self.angle_triples.T
        Lu, Lv = self.position_lipschitz[a]+self.position_lipschitz[b], self.position_lipschitz[c]+self.position_lipschitz[b]
        Bu, Bv = self.position_bound[a]+self.position_bound[b], self.position_bound[c]+self.position_bound[b]
        self.register_buffer("angle_lipschitz", (1+sine_min)*(Lu*Bv+Lv*Bu))
        q = raw.chiral_quads
        self.register_buffer("chiral_quads", q.clone())
        self.register_buffer("chiral_signs", raw.chiral_signs.clone())
        cqL = self.position_lipschitz[q[:, :3]]+self.position_lipschitz[q[:, 3, None]]
        cqB = self.position_bound[q[:, :3]]+self.position_bound[q[:, 3, None]]
        chiralL = sum(cqL[:, k]*cqB[:, (k+1)%3]*cqB[:, (k+2)%3] for k in range(3))
        self.register_buffer("chiral_lipschitz", chiralL)
        self.has_neck = raw.energy.gb_kind == "GBn2"
        if self.has_neck:
            radii = raw.energy.gb_or + raw.energy.offset
            neck = radii[pairs[:, 0]]+radii[pairs[:, 1]]+.68
            self.register_buffer("neck", neck)

    def normalized_margins(self, z):


        clipped = z.clamp(self.lower, self.upper)
        x = self.system.positions(clipped)
        p = self.pairs
        distances = (x[p[:, 0]]-x[p[:, 1]]).square().sum(-1).clamp_min(1e-30).sqrt()
        margins = [z-self.lower, self.upper-z,
                   (distances-self.collision_nm)/self.pair_lipschitz]
        a, b, c = self.angle_triples.T
        u, v = x[a]-x[b], x[c]-x[b]
        norm = lambda t: t.square().sum(-1).clamp_min(1e-30).sqrt()
        angle = norm(torch.linalg.cross(u, v))-self.sine_min*norm(u)*norm(v)
        margins.append(angle/self.angle_lipschitz)
        q = x[self.chiral_quads]
        vectors = q[:, :3]-q[:, 3, None]
        signed = (torch.linalg.cross(vectors[:, 0], vectors[:, 1])*vectors[:, 2]).sum(-1)*self.chiral_signs
        margins.append((signed-self.chiral_margin)/self.chiral_lipschitz)
        if self.has_neck:
            margins.append(((distances-self.neck).abs()-self.neck_exclusion_nm)/self.pair_lipschitz)
        return torch.cat(margins)

    def clearance(self, z):
        return -self.temperature*torch.logsumexp(-self.normalized_margins(z)/self.temperature, 0)

    def forward(self, z):
        if z.ndim > 1:
            return torch.vmap(self.forward)(z)
        return smoothstep(self.clearance(z)/self.width)

    def specification(self):
        return dict(lower=self.lower.tolist(), upper=self.upper.tolist(), width=self.width,
                    temperature=self.temperature, collision_nm=self.collision_nm,
                    sine_min=self.sine_min, chiral_margin=self.chiral_margin,
                    neck_exclusion_nm=self.neck_exclusion_nm, lipschitz=self.lipschitz)


def fit_protection(system, development, padding=1.0, width_fraction=.25):
    if development.ndim != 2 or len(development) < 2 or not torch.isfinite(development).all():
        raise ValueError("Protection fitting requires complete ordinary development geometry")
    spread = development.std(0).clamp_min(.1)
    lower, upper = development.amin(0)-padding*spread, development.amax(0)+padding*spread
    temporary = MolecularProtection(system, lower, upper, 1., 1e-20)
    minimum = torch.stack([temporary.normalized_margins(z).min() for z in development]).detach()
    positive = minimum[minimum > 0]
    if len(positive) == 0:
        raise ValueError("No D geometry lies in the chosen safe molecular support")
    available = float(torch.quantile(positive, .1))
    count = len(temporary.normalized_margins(development[0]))
    tau = .01*available/max(1., math.log(count))
    width = width_fraction*available
    return MolecularProtection(system, lower, upper, width, tau)
