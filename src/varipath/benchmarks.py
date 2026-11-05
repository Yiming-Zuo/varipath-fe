import math
import numpy as np
from scipy.integrate import quad
from scipy.special import i0
import torch


class HarmonicSystem:
    name, dim = "harmonic", 2

    def effective_u(self, z, endpoint):
        if endpoint == 0:
            return .5*z.square().sum(-1)
        return .5*((z-z.new_tensor([.35, -.2]))*z.new_tensor([1.1, .9])).square().sum(-1)+.3

    def sample(self, count, endpoint, seed):
        rng = np.random.default_rng(seed)
        x = rng.normal(size=(count, self.dim))
        if endpoint:
            x = x/np.array([1.1, .9])+np.array([.35, -.2])
        return torch.tensor(x, dtype=torch.float64)

    def regions(self, z):
        a, b = z[..., 0] < -.25, z[..., 0] > .25
        return dict(a=a, b=b, remainder=~(a|b))

    def exact_delta_f(self):
        return .3+math.log(1.1*.9)


class PeriodicSystem:
    name, dim = "periodic", 2

    def effective_u(self, z, endpoint):
        if endpoint == 0:
            return -.4*z.cos().sum(-1)
        return -1.3*(z-z.new_tensor([.7, -.4])).cos().sum(-1)+.3

    def sample(self, count, endpoint, seed):
        rng = np.random.default_rng(seed)
        x = rng.vonmises([.7, -.4] if endpoint else [0., 0.],
                         1.3 if endpoint else .4, size=(count, 2))
        return torch.tensor(x, dtype=torch.float64)

    def regions(self, z):
        a, b = z[..., 0].cos() > .8, z[..., 0].cos() < -.8
        return dict(a=a, b=b, remainder=~(a|b))

    def exact_delta_f(self):
        return .3-2*math.log(i0(1.3)/i0(.4))


class DoubleWellSystem:
    name, dim = "double-well", 2

    @staticmethod
    def tilt(x):
        x = x.clamp(-.5, .5)
        primitive = x**3/48-x**5/10+x**7/7
        return primitive/(.5**3/48-.5**5/10+.5**7/7)

    def effective_u(self, z, endpoint):
        return (z[..., 0].square()-1).square()+.5*z[..., 1].square()-.05*endpoint*self.tilt(z[..., 0])

    def sample(self, count, endpoint, seed):
        rng = np.random.default_rng(seed)
        points = []


        while sum(len(p) for p in points) < count:
            x = rng.normal(size=max(64, 2*count))
            value = torch.tensor(x, dtype=torch.float64)
            ratio = -(x*x-1)**2+.5*x*x-.5625
            if endpoint:
                ratio += .05*self.tilt(value).numpy()-.05
            points.append(x[np.log(rng.random(len(x))) < ratio])
        x = np.concatenate(points)[:count]
        return torch.tensor(np.column_stack((x, rng.normal(size=count))), dtype=torch.float64)

    def regions(self, z):
        return dict(a=z[..., 0] > 0, b=z[..., 0] <= 0,
                    remainder=torch.zeros(z.shape[:-1], dtype=torch.bool, device=z.device))

    def exact_delta_f(self):
        def integrand(x, endpoint):
            z = torch.tensor([x, 0.], dtype=torch.float64)
            return math.exp(-float(self.effective_u(z, endpoint)))
        return -math.log(quad(integrand, -5, 5, args=(1,), epsabs=1e-12)[0]
                         /quad(integrand, -5, 5, args=(0,), epsabs=1e-12)[0])
