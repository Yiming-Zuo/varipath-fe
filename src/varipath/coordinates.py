from __future__ import annotations

import math
import torch
from torch import nn


def sphere_direction(u):
    s = (u * u).sum(-1)
    small = s < 1e-5
    safe_s = torch.where(small, torch.ones_like(s), s)
    v = torch.exp(-s / 2)
    exact = 2 * torch.sqrt(torch.exp(-safe_s / 2) * (-torch.expm1(-safe_s / 2)) / safe_s)
    series = math.sqrt(2) * (1 - 3*s/8 + 29*s*s/384 - 11*s*s*s/1024)
    scale = torch.where(small, series, exact)
    return torch.cat((scale.unsqueeze(-1)*u, (2*v-1).unsqueeze(-1)), -1)


def sphere_inverse(n):
    transverse2 = n[..., :2].square().sum(-1)
    north = (transverse2 == 0) & (n[..., 2] >= 0)
    v = torch.where(n[..., 2] < 0, transverse2/(2*(1-n[..., 2])), (1+n[..., 2])/2)
    if torch.any(v <= 0) or torch.any(v > 1+1e-12):
        raise FloatingPointError("球面图极点或无效方向没有有限逆坐标")
    rho = torch.sqrt(-2*torch.log(v.clamp(max=1)))
    denominator = torch.sqrt(torch.where(north, torch.ones_like(transverse2), transverse2))
    return torch.where(north.unsqueeze(-1), torch.zeros_like(n[..., :2]), rho.unsqueeze(-1)*n[..., :2]/denominator.unsqueeze(-1))


def dihedral_radians(points):
    b0 = points[..., 0, :]-points[..., 1, :]
    b1 = points[..., 2, :]-points[..., 1, :]
    b2 = points[..., 3, :]-points[..., 2, :]
    length = torch.linalg.vector_norm(b1, dim=-1, keepdim=True)
    axis = b1/length
    v = b0-(b0*axis).sum(-1, keepdim=True)*axis
    w = b2-(b2*axis).sum(-1, keepdim=True)*axis
    angle = torch.atan2((torch.linalg.cross(axis, v)*w).sum(-1), (v*w).sum(-1))
    valid = (length.squeeze(-1)>1e-12) & (torch.linalg.vector_norm(v,dim=-1)>1e-12) & (torch.linalg.vector_norm(w,dim=-1)>1e-12)
    return torch.where(valid, angle, torch.full_like(angle, float("nan")))


class BodyCoordinates(nn.Module):
    def __init__(self, metadata):
        super().__init__()
        self.triad = tuple(metadata["body_triad"])
        self.n_atoms = len(metadata["atoms"])
        self.heavy = [r["index"] for r in metadata["atoms"] if r["element"] != "H" and r["index"] not in self.triad]
        self.parents = [r["parent_index"] for r in metadata["constraints"]]
        self.hydrogens = [r["hydrogen_index"] for r in metadata["constraints"]]
        self.register_buffer("lengths", torch.tensor([r["length_nm"] for r in metadata["constraints"]], dtype=torch.float64))
        self.register_buffer("masses", torch.tensor([r["mass_dalton"] for r in metadata["atoms"]], dtype=torch.float64))
        self.split = 3+3*len(self.heavy)
        self.dim = self.split+2*len(self.hydrogens)
        self.groups = [(p,[i for i, parent in enumerate(self.parents) if parent==p]) for p in sorted(set(self.parents))]

    def constraint_error(self, x):
        return (torch.linalg.vector_norm(x[..., self.hydrogens,:]-x[..., self.parents,:],dim=-1)-self.lengths).abs().amax()

    def encode_with_rotation(self, x):
        if x.dtype!=torch.float64 or x.shape[-2:] != (self.n_atoms,3) or not torch.isfinite(x).all():
            raise ValueError("输入必须是完整有限的原子坐标")
        if self.constraint_error(x) > 1e-10:
            raise ValueError("输入违反 1e-10 nm 键长容差；编码不能投影数据")
        origin, first, second = self.triad
        delta = x-x[..., origin:origin+1,:]
        ex = delta[..., first,:]
        length = torch.linalg.vector_norm(ex,dim=-1,keepdim=True)
        ex = ex/length
        perp = delta[..., second,:]-(delta[..., second,:]*ex).sum(-1,keepdim=True)*ex
        height = torch.linalg.vector_norm(perp,dim=-1,keepdim=True)
        if torch.any(length<=1e-12) or torch.any(height<=1e-12):
            raise ValueError("体固参考框退化")
        ey = perp/height
        axes = torch.stack((ex,ey,torch.linalg.cross(ex,ey)),-2)
        body = delta@axes.transpose(-1,-2)
        direction = (body[..., self.hydrogens,:]-body[..., self.parents,:])/self.lengths[...,None]
        u = sphere_inverse(direction)
        root = torch.stack((length.squeeze(-1).log(),body[...,second,0],height.squeeze(-1).log()),-1)
        z = torch.cat((root,body[...,self.heavy,:].flatten(-2),u.flatten(-2)),-1)
        return z, axes

    def encode(self,x):
        return self.encode_with_rotation(x)[0]

    def forward(self,z):
        if z.dtype!=torch.float64 or z.shape[-1]!=self.dim:
            raise ValueError("形状坐标必须为完整维数的 float64")
        zero = torch.zeros_like(z[..., :3])
        xaxis = z.new_tensor([1.,0.,0.])
        yaxis = z.new_tensor([0.,1.,0.])
        atom = [None]*self.n_atoms
        origin, first, second = self.triad
        atom[origin] = zero
        atom[first] = z[...,0,None].exp()*xaxis
        atom[second] = z[...,1,None]*xaxis+z[...,2,None].exp()*yaxis
        for i,index in enumerate(self.heavy):
            atom[index] = z[...,3+3*i:6+3*i]
        directions = sphere_direction(z[...,self.split:].reshape(*z.shape[:-1],-1,2))
        for i,(hydrogen,parent) in enumerate(zip(self.hydrogens,self.parents)):
            atom[hydrogen] = atom[parent]+self.lengths[i]*directions[...,i,:]
        positions = torch.stack(atom,-2)
        return positions-(positions*self.masses[...,None]).sum(-2,keepdim=True)/self.masses.sum()

    def log_volume(self,z):
        u = z[...,self.split:].reshape(*z.shape[:-1],-1,2)
        directions = sphere_direction(u)
        result = 3*z[...,0]+2*z[...,2]-.5*u.square().sum((-1,-2))
        for parent,group in self.groups:
            n = directions[...,group,:]
            gram = torch.diag(1/self.masses[[self.hydrogens[i] for i in group]])+(n@n.transpose(-1,-2))/self.masses[parent]
            result = result+torch.linalg.cholesky(gram).diagonal(dim1=-2,dim2=-1).log().sum(-1)
        return result

    def dense_log_volume(self,z):

        x = self(z)
        derivative = torch.func.jacfwd(self.forward)(z).reshape(3*self.n_atoms,self.dim)
        eye = torch.eye(3,dtype=z.dtype,device=z.device)
        rotation = torch.linalg.cross(eye[:,None,:],x[None,:,:]).reshape(3,-1).T
        basis = torch.cat((derivative,rotation),1)
        gram = basis.T@(self.masses.repeat_interleave(3)[:,None]*basis)
        return torch.linalg.cholesky(gram).diagonal().log().sum()
