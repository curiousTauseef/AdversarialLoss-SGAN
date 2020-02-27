import os
import time
import torch
import numpy as np
import inspect
from contextlib import contextmanager
import subprocess


def int_tuple(s):
    return tuple(int(i) for i in s.split(','))


def find_nan(variable, var_name):
    variable_n = variable.data.cpu().numpy()
    if np.isnan(variable_n).any():
        exit('%s has nan' % var_name)


def bool_flag(s):
    if s == '1':
        return True
    elif s == '0':
        return False
    msg = 'Invalid value "%s" for bool flag (should be 0 or 1)'
    raise ValueError(msg % s)


def lineno():
    return str(inspect.currentframe().f_back.f_lineno)


def get_total_norm(parameters, norm_type=2):
    if norm_type == float('inf'):
        total_norm = max(p.grad.data.abs().max() for p in parameters)
    else:
        total_norm = 0
        for p in parameters:
            try:
                param_norm = p.grad.data.norm(norm_type)
                total_norm += param_norm**norm_type
                total_norm = total_norm**(1. / norm_type)
            except:
                continue
    return total_norm


@contextmanager
def timeit(msg, should_time=True):
    if should_time:
        torch.cuda.synchronize()
        t0 = time.time()
    yield
    if should_time:
        torch.cuda.synchronize()
        t1 = time.time()
        duration = (t1 - t0) * 1000.0
        print('%s: %.2f ms' % (msg, duration))


def get_gpu_memory():
    torch.cuda.synchronize()
    opts = [
        'nvidia-smi', '-q', '--gpu=' + str(1), '|', 'grep', '"Used GPU Memory"'
    ]
    cmd = str.join(' ', opts)
    ps = subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output = ps.communicate()[0].decode('utf-8')
    output = output.split("\n")[0].split(":")
    consumed_mem = int(output[1].strip().split(" ")[0])
    return consumed_mem


def get_dset_path(dset_name, dset_type):
    _dir = os.path.dirname(__file__)
    _dir = _dir.split("/")[:-1]
    _dir = "/".join(_dir)
    return os.path.join(_dir, 'datasets', dset_name, dset_type)


def relative_to_abs(rel_traj, start_pos):
    """
    Inputs:
    - rel_traj: pytorch tensor of shape (seq_len, batch, 2)
    - start_pos: pytorch tensor of shape (batch, 2)
    Outputs:
    - abs_traj: pytorch tensor of shape (seq_len, batch, 2)
    """
    # batch, seq_len, 2
    rel_traj = rel_traj.permute(1, 0, 2)
    displacement = torch.cumsum(rel_traj, dim=1)
    start_pos = torch.unsqueeze(start_pos, dim=1)
    abs_traj = displacement + start_pos
    return abs_traj.permute(1, 0, 2)

## Important for controlled Experiments to plot trajectory distribution
def view_traj(ax, fake_pred, real_pred, obs, args, all_three=False):
    fake_pred = fake_pred.cpu().numpy()
    real_pred = real_pred.cpu().numpy()
    obs = obs.cpu().numpy()

    fake_traj = np.concatenate((obs, fake_pred), axis=0)
    real_traj = np.concatenate((obs, real_pred), axis=0)

    if all_three:
        x_obs = np.tile(np.linspace(1, 8, num=8, endpoint=True), (3,1))
        y_obs = np.zeros((3, 8))
        ##Real Predictions
        x_pred = np.tile(np.linspace(9, 16, num=8, endpoint=True), (3,1))
        y_pred = np.zeros((3, 8))

        ##1st Mode (Straight Line)
        ##2nd Mode (Up Slant Line)
        y_pred[1, :] = 1*np.linspace(1, 8, num=8, endpoint=True)
        ##3rd Mode (Down Slant Line)
        y_pred[2, :] = -1*np.linspace(1, 8, num=8, endpoint=True)


        ax.plot(fake_pred[:, 0], fake_pred[:, 1], 'g', label='Predicted')
        ax.plot(x_obs[0],  y_obs[0],  'b',  label='Observed')
        ax.plot(x_pred[0], y_pred[0], 'r', label='Real Pred 1')
        if 'single' not in args.dataset_name:
            ax.plot(x_pred[1], y_pred[1], 'r', label='Real Pred 2')
            ax.plot(x_pred[2], y_pred[2], 'r', label='Real Pred 3')

    else:
        # pass
        ax.plot(fake_pred[:, 0], fake_pred[:, 1], 'g')