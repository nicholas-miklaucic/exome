"""Main class that ties together the idea of fitting a parameterized function to data."""

from typing import ItemsView
from funcs import ParamSpace, prior_dist, gauss_kde, to_np, exp_kernel
from summary_stats import dist_pop
import torch
from torch.nn import Module
import torch.nn as nn
import pyro.distributions as dists
import pyro.distributions.constraints as constrs
from scipy.interpolate import InterpolatedUnivariateSpline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import rho_plus as rp
import logging

logging.basicConfig(level=logging.INFO)

class ParameterizedCurve(Module):
    @classmethod
    def arg_constraints(cls) -> dict:
        """Constraints on the input arguments."""
        return NotImplementedError

    @classmethod
    def arg_dists(cls) -> dict:
        """Prior distributions on the input arguments.
        Not needed (defaults are chosen), but if you have
        special knowledge about the parameter use this
        to guide."""
        return {}

    @classmethod
    def support(cls) -> constrs.Constraint:
        """Constraint on the support of the function."""
        return constrs.real


    @classmethod
    def _target_inner(cls, samples):
        raise NotImplementedError()

    @classmethod
    def target(cls, samples):
        if isinstance(samples, dict):
            samples = {
                k: torch.as_tensor(samples[k])
                if not isinstance(samples[k], torch.Tensor)
                else samples[k] for k in cls.arg_constraints()}
            samples = torch.stack(list(samples.values()), dim=-1)

        batch_shape = samples.shape[:-1]
        def f(x: torch.Tensor):
            new_shape = list(x.shape) + list(batch_shape)
            for _ in range(len(batch_shape)):
                x = x.unsqueeze(-1)
            x = x.expand(new_shape)
            y = cls._target_inner(samples)(x)
            x = x.moveaxis(-len(batch_shape)-1, -1)
            y = y.moveaxis(-len(batch_shape)-1, -1)
            return (x, y)
        return f


    def __init__(
        self,
        theta_dists={},
        n_groups = 7,
        n_samples = 20,
        n_test = 40,
        scale=1,
        loss_fn=nn.SmoothL1Loss(),
        param_kernel=gauss_kde,
        param_bw=0.3,
        param_agg=np.median,
        n_xgrid=100,
        n_finish_epochs=100,
        pop={}
    ):
        super().__init__()
        self.theta_dist_total = self.arg_dists()
        self.theta_dist_total.update(theta_dists)
        self.space = ParamSpace(self.arg_constraints(), self.theta_dist_total, n_groups, n_samples, scale)
        self.loss_fn = loss_fn
        self.param_kernel = param_kernel
        self.n_xgrid = n_xgrid
        self.pop = pop
        self.dfn = (self.space.n_groups - 1)
        self.dfd = (self.space.n_samples * self.space.n_groups - self.space.n_groups)
        self.param_bw = param_bw
        self.param_agg = param_agg
        self.n_test = n_test
        self.n_finish_epochs = n_finish_epochs
        self.xx = prior_dist(self.support()).icdf(torch.linspace(0, 1, self.n_test+2)[1:-1])


    def single_f_stat(self, subs):
        v = subs[['group', 'value']].groupby('group').agg(self.param_agg)
        kv = exp_kernel(self.space.n_groups, self.param_bw) @ v
        kv.index = v.index
        avg = kv.loc[subs['group']].values
        err = subs['value'].values - avg.reshape(-1)

        ss_treat = (((kv['value'] - kv['value'].mean()) ** 2) * subs['group'].value_counts()).sum()
        ms_treat = ss_treat / self.dfn

        ss_err = np.sum(err ** 2)
        ms_err = ss_err / self.dfd

        F = ms_treat / ms_err

        return F



    def fit_initial(self, x, y):
        self._x = x
        self._y = y

        sdfs = []
        all_roots = []
        fs = []
        theta_star = {}
        for trial_i in range(self.space.n_params):
            f1 = self._anova()
            fs.append(f1)
            i = 0
            sdf, true_stat, trans, z, roots = self._fit_param(*f1.index[i])
            num_tries = min(len(f1.index), 5)
            while not len(roots) and i < num_tries - 1:
                logging.warning('{} failed (trial {})'.format(f1.index[i], i))
                i += 1
                sdf, true_stat, trans, z, roots = self._fit_param(*f1.index[i])


            logging.debug('{} {}'.format(*f1.index[i]))
            roots.sort()
            sdf[['param', 'stat']] = f1.index[i]
            sdf['true_stat'] = true_stat
            sdf['trial'] = trial_i
            sdfs.append(sdf)
            all_roots.append(roots)
            if len(roots) == 0:
                logging.warn('No root found')
                root = trans(torch.as_tensor(z.abs().sort_values('value').index[0])).item()
            else:
                root = roots[(len(roots) - 1) // 2]
            logging.debug('Actual value of {} {:.3f}, finding {}, found value {:.3f}'.format(
                f1.index[i][1], true_stat, f1.index[i][0], root))
            theta_star[f1.index[i][0]] = root
            self.space.fix(f1.index[i][0], root)

        self._sdfs = sdfs
        self._roots = all_roots
        self._fs = fs
        self._theta = theta_star


    def pop_values(self):
        samples, param_names, param_is = self.space.free_sample_space()
        x, y = self.target(samples)(self.xx)
        # print(samples.shape, x.shape, y.shape)
        pop_vals = {}
        for name, individual in self.pop.items():
            pop_vals[name] = to_np(individual(x, y)).flatten()

        return pd.DataFrame(
            pop_vals,
            index=pd.MultiIndex.from_product([
                param_names,
                [f'G{i}' for i in np.arange(self.space.n_groups)],
                [f'S{i}' for i in np.arange(self.space.n_samples)]
            ], names=['param', 'group', 'sample'])
        )

    def _anova(self):
        sdf = self.pop_values()
        sdf = sdf.melt(ignore_index=False, var_name='stat')
        f_stats = sdf.reset_index().groupby(['param', 'stat']).apply(self.single_f_stat)
        f_stats.name = 'F'
        return f_stats.sort_values(ascending=False)


    def _fit_param(self, param, stat):
        self.space.unfix(param)

        samples, free_names, free_is = self.space.free_sample_space()
        param_i = self.space.param_names.index(param)

        group_i = free_names.index(param)
        samples = samples[group_i, ...]

        supp_trans = dists.transforms.transform_to(self.support())
        xx = dists.Normal(
            supp_trans.inv(self._x.mean()),
            1.5 * supp_trans.inv(self._x).std()).icdf(
                torch.linspace(0, 1, self.n_xgrid+2)[1:-1])
        xx = supp_trans(xx)
        true_stat = self.pop[stat](self._x, self._y).item()
        sdf = self.pop_values()
        sdf['value'] = sdf[stat]
        grps = sdf[['value']].query('param == @param').groupby('group')
        iqr = grps.agg(lambda x: np.quantile(x, 0.75) - np.quantile(x, 0.25))
        med = grps.median()
        z_ish_score = (true_stat - med) / (iqr + 1)

        trans = self.space.theta_transforms[param]
        z_ish_score.index = to_np(trans.inv(self.space.group_levels[param_i]))
        z_ish_score = z_ish_score.sort_index()

        spl = InterpolatedUnivariateSpline(
            z_ish_score.index,
            z_ish_score.values,
            k=3
        )

        return sdf, true_stat, trans, z_ish_score, to_np(trans(torch.tensor(spl.roots())))


    def plot_diagnostics(self):
        n_plots = len(self._sdfs)
        n_cols = int(np.ceil(np.sqrt(n_plots)))
        n_rows = int(np.ceil(n_plots / n_cols))

        fig, axs = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 5 * n_rows))
        for sdf, roots, ax in zip(self._sdfs, self._roots, axs.flatten()):
            levels = to_np(self.space.group_levels[self.space.param_names.index(sdf['param'][0])])
            sdf['group_val'] = [levels[int(s[1:])] for s in sdf.index.get_level_values('group')]
            sns.histplot(sdf, x='group_val', y='value', ax=ax, cmap='rho_heatmap', bins=20)
            ax.axhline(sdf['true_stat'].iloc[0], color=plt.rcParams['text.color'])
            ax.set_xlabel(sdf['param'][0])
            ax.set_ylabel(sdf['stat'][0])
            for i, r in enumerate(roots):
                if i == (len(roots) - 1) // 2:
                    c = sns.color_palette()[0]
                else:
                    c = plt.rcParams['text.color']

                ax.axvline(r, c=c)

    def plot_target(self, transform=None, plot_real=True, margin=1,
                    model_label='Model Fit', theta={}, **kwargs):
        if transform is None:
            transform = dists.transforms.identity_transform

        xx = torch.linspace(
            float(self._x.min() - margin * self._x.std()),
            float(self._x.max() + margin * self._x.std()),
            100
        )

        xx = xx[self.support().check(xx)]

        # print(theta)
        txx, tyy = self.target(theta)(xx)


        plt.plot(
            to_np(txx.squeeze()),
            to_np(transform(tyy.squeeze())),
            ls='--',
            label=model_label
        )
        if plot_real:
            plt.plot(
                to_np(self._x),
                to_np(transform(self._y)),
                label='True Values'
            )
        plt.legend()
        # plt.ylim(tyy.min() - tyy.std() * 0.5, tyy.max() + tyy.std() * 0.5)


    def interact_params(self):
        space = self.space
        theta_star = self._theta
        from ipywidgets import widgets
        widgetdict = {}
        for p, dist in zip(space.param_names, space.param_dists):
            extent = torch.abs(dist.cdf(torch.as_tensor(theta_star[p])) - 0.5)
            extent = torch.maximum(extent, torch.tensor(0.45))
            widgetdict[p] = widgets.FloatSlider(
                theta_star[p],
                min=dist.icdf(0.5 - extent).item(),
                max=dist.icdf(0.5 + extent).item(),
                step=0.01,
            )
        return widgetdict


    def fit_finish(self, n_epochs=None):
        """Uses gradient descent to fine-tune fitting."""
        if n_epochs is None:
            n_epochs = self.n_finish_epochs
        self._theta_init = self._theta
        to_param = self.space.theta_transforms
        self._theta_raw = {k: to_param[k].inv(torch.as_tensor(v)).detach().clone().requires_grad_(True) for k, v in self._theta.items()}

        opt = torch.optim.Adam(list(self._theta_raw.values()), lr=3e-2)
        sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.99)
        self._losses = []
        for _ in range(n_epochs):
            opt.zero_grad()
            self._theta = {k: to_param[k](torch.as_tensor(v)) for k, v in self._theta_raw.items()}
            xhat, yhat = self.target(self._theta)(self._x)
            loss = self.loss_fn(yhat, self._y)
            self._losses.append(loss)
            loss.backward()
            opt.step()
            sched.step()

        self._losses = torch.stack(self._losses, dim=0)
        self._theta = {k: to_param[k](v) for k, v in self._theta_raw.items()}


def dist_as_curve(dist):
    class ParameterizedDistribution(ParameterizedCurve):
        @classmethod
        def arg_constraints(cls) -> dict:
            """Constraints on the input arguments."""
            return dist.arg_constraints

        @classmethod
        def support(cls) -> constrs.Constraint:
            """Constraint on the support of the function."""
            return dist.support

        @classmethod
        def _target_inner(cls, samples):
            return dist(*[samples[..., i] for i in range(samples.size(-1))]).log_prob

    return ParameterizedDistribution


if __name__ == '__main__':
    from scipy import stats
    df = sns.load_dataset('penguins').sort_values('body_mass_g')
    df['body_mass_kg'] = df['body_mass_g'] / 1000
    df_xx = df['body_mass_kg'].dropna()
    df_yy = np.log(stats.gaussian_kde(df_xx)(df_xx))
    torch_xx = torch.tensor(df_xx.values.reshape(-1))
    torch_yy = torch.tensor(df_yy.reshape(-1))

    dist = dists.LogNormal
    curve = dist_as_curve(dist)(n_samples=20, scale=2, pop=dist_pop)
    curve.fit_initial(torch_xx, torch_yy)
    print(curve.loss_fn(curve.target(curve._theta)(torch_xx)[1], torch_yy))
    curve.fit_finish()
    print(curve.loss_fn(curve.target(curve._theta)(torch_xx)[1], torch_yy))