"""Modules describing a parameterized function. Used to finish off the solution."""

from torch.nn import Module
from pyro import distributions as dists
import torch

def dist_as_target(dist_name):
    if isinstance(dist_name, str):
        dist = getattr(dists, dist_name)
    else:
        dist = dist_name

    class DistCurve(Module):
        def __init__(self, theta):
            super().__init__()
            self.dist = dist
            self.theta = theta
            self.frozen = self.dist(*[self.theta[..., i] for i in range(self.theta.shape[-1])])

        def forward(self, x: torch.Tensor):
            for _ in range(len(self.frozen.batch_shape)):
                x = x.unsqueeze(-1)

            y = self.frozen.expand_by(x.shape[:1]).log_prob(x)
            x = x.expand_as(y)

            x, y = torch.moveaxis(x, 0, -1), torch.moveaxis(y, 0, -1)
            return (x, y)

    return DistCurve