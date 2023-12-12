import pyro
import pyro.distributions as dists
import pyro.distributions.constraints as constrs
import pyro.distributions.transforms as transforms
import torch.distributions.transforms as torch_transforms
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import rho_plus as rp
from scipy.stats import beta

def to_np(tt: torch.Tensor) -> np.ndarray:
    return tt.detach().cpu().numpy()


def prior_dist(constr, scale=1):
    if isinstance(constr, type(constrs.real)):
        return dists.Normal(0, 2 * scale)
    elif isinstance(constr, constrs.interval):
        lb, ub = constr.lower_bound, constr.upper_bound
        loc = lb
        scale = ub - lb
        return dists.TransformedDistribution(
            dists.Normal(0, 3),
            [
                torch_transforms.CumulativeDistributionTransform(dists.Logistic(0, 1)),
                transforms.AffineTransform(loc, scale)
            ])
    elif isinstance(constr, (constrs.greater_than, constrs.greater_than_eq)):
        lb = constr.lower_bound
        return dists.TransformedDistribution(
            dists.Exponential(1 / scale),
            [transforms.AffineTransform(lb, 1)])
    elif isinstance(constr, (constrs.less_than,)):
        ub = constr.upper_bound
        return dists.TransformedDistribution(
            dists.Exponential(1 / scale),
            [transforms.AffineTransform(ub, -1)])
    else:
        print(f'No custom value for {constr}')
        return dists.TransformedDistribution(
            prior_dist(constrs.real, scale),
            [transforms.transform_to(constr)]
        )


class ParamSpace:
    """Parameter space for a nonlinear regression problem."""
    def __init__(
        self,
        theta_constraints,
        theta_dists={},
        n_groups = 7,
        n_samples = 20,
        scale=1):
        self.theta_transforms = {
            param: transforms.transform_to(constr)
            for param, constr in theta_constraints.items()
        }
        self.param_names = list(self.theta_transforms)
        self.n_groups = n_groups
        self.n_samples = n_samples
        self.n_params = len(theta_constraints)
        self.param_dists = [prior_dist(constr, scale) for constr in theta_constraints.values()]
        for i, param in enumerate(self.param_names):
            if param in theta_dists and theta_dists[param] is not None:
                self.param_dists[i] = theta_dists[param]
        self.fixed_values = [None for _ in range(self.n_params)]
        self._initialize_space()

    def _initialize_space(self):
        self.group_levels = []
        samples_shape = (self.n_params, self.n_groups, self.n_samples, self.n_params)
        raw_samples = torch.stack([
            param_dist.sample([self.n_samples]) for param_dist in self.param_dists
        ], dim=-1)
        self.samples = raw_samples.unsqueeze(0).expand(samples_shape).clone()
        for i, param_dist in enumerate(self.param_dists):
            levels = param_dist.icdf(torch.linspace(0, 1, self.n_groups + 2)[1:-1])
            self.group_levels.append(levels)
            self.samples[i, :, :, i] = levels.unsqueeze(0).expand((self.n_samples, self.n_groups)).T.clone()


    def fix(self, param, value):
        if isinstance(param, str):
            param = self.param_names.index(param)
        else:
            param = int(param)

        self.fixed_values[param] = value

    def unfix(self, param):
        if isinstance(param, str):
            param = self.param_names.index(param)
        else:
            param = int(param)

        self.fixed_values[param] = None

    def free_sample_space(self):
        samples = self.samples.clone()
        free_vars = []
        for i, value in enumerate(self.fixed_values):
            if value is None:
                free_vars.append(i)
            else:
                samples[..., i] = value
        return samples[free_vars, ...], [self.param_names[i] for i in free_vars], free_vars

def pop_values(space: ParamSpace, f, x, pop):
    samples, param_names, param_is = space.free_sample_space()
    x, y = f(samples)(x)
    # print(samples.shape, x.shape, y.shape)
    pop_vals = {}
    for name, individual in pop.items():
        pop_vals[name] = to_np(individual(x, y)).flatten()

    return pd.DataFrame(
        pop_vals,
        index=pd.MultiIndex.from_product([
            param_names,
            [f'G{i}' for i in np.arange(space.n_groups)],
            [f'S{i}' for i in np.arange(space.n_samples)]
        ], names=['param', 'group', 'sample'])
    )


def exp_kernel(n, bw=0.3):
    kernel = bw ** np.abs(np.arange(n) - np.arange(n).reshape(-1, 1))
    kernel /= kernel.sum(axis=1, keepdims=True)
    return kernel

def single_f_stat(space, subs, bw=0.3, aggfunc=np.median):
    dfn = (space.n_groups - 1)
    dfd = (space.n_samples * space.n_groups - space.n_groups)
    v = subs[['group', 'value']].groupby('group').median()
    kv = exp_kernel(space.n_groups, bw) @ v
    kv.index = v.index
    avg = kv.loc[subs['group']].values
    err = subs['value'].values - avg.reshape(-1)

    ss_treat = (((kv['value'] - kv['value'].mean()) ** 2) * subs['group'].value_counts()).sum()
    ms_treat = ss_treat / dfn

    ss_err = np.sum(err ** 2)
    ms_err = ss_err / dfd

    F = ms_treat / ms_err

    return F


def f_stat(space, pop={}):
    sdf = pop_values(space, pop)
    sdf = sdf.melt(ignore_index=False, var_name='stat')
    f_stats = sdf.reset_index().groupby(['param', 'stat']).apply(single_f_stat)
    f_stats.name = 'F'
    return f_stats.sort_values(ascending=False)


def gauss_kde(loc: torch.Tensor, bw='scott'):
    if bw == 'scott':
        bw = 0.5 * loc.shape[-1] ** (-1/5)
    return dists.MixtureSameFamily(
        dists.Categorical(torch.ones(loc.shape[-1])),
        dists.Normal(loc, torch.ones_like(loc) * bw)
    )
