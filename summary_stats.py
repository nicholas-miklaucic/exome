"""Defines summary statistics for functions."""

import torch

def expected_value(func):
    def e_f(x, log_y):
        # return (func(x) * torch.softmax(log_y, dim=-1).nan_to_num(0)).sum(dim=-1, keepdim=True)
        ii = torch.argsort(x, dim=-1)
        x = x.gather(-1, ii)
        log_y = log_y.gather(-1, ii)
        area = torch.log(torch.trapezoid(torch.exp(log_y), x)).unsqueeze(-1)
        avg = (func(x) * torch.softmax(torch.clip(log_y, -30, 30), -1)).sum(dim=-1)
        log_y = log_y - area
        trap = torch.trapezoid(func(x) * torch.exp(log_y), x)
        trap[trap.isnan()] = avg[trap.isnan()]
        return trap
    return e_f

mean = expected_value(lambda x: x)

def std(x, log_y):
    mu = mean(x, log_y)
    x_diff = x - mu.unsqueeze(-1)
    return torch.sqrt(expected_value(torch.square)(x_diff, log_y))

def mode(x, log_y):
    return x.gather(-1, torch.argmax(log_y, dim=-1, keepdim=True))[..., 0]


dist_pop = {
    'mode': mode,
    'mean': mean,
    'std': std,
}


