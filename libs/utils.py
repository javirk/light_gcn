import os
import torch
import math
import yaml
import shutil
import numpy as np
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

wrap_to_2pi = lambda x: x % (2 * np.pi) + (2 * np.pi) * (x == 0)


def batch_dot(v1, v2):
    return torch.einsum('b d, b d -> b', v1, v2)


def transfer_batch_to_device(batch, device):
    for k, v in batch:
        if hasattr(v, 'to'):
            batch[k] = v.to(device)
    return batch


def gather_batch(input, index):
    b, t, p = index.shape
    _, _, c = input.shape
    output = torch.zeros((b, t, p, c), device=input.device)

    for i, (val, idx) in enumerate(zip(input, index)):
        output[i] = val[idx]

    return output


def get_optimizer(p, parameters):
    if p['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(parameters, **p['optimizer_kwargs'])

    elif p['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(parameters, **p['optimizer_kwargs'])

    else:
        raise ValueError('Invalid optimizer {}'.format(p['optimizer']))

    return optimizer


def adjust_learning_rate(p, optimizer, epoch):
    lr = p['optimizer_kwargs']['lr']

    if p['scheduler'] == 'step':
        steps = np.sum(epoch > np.array(p['scheduler_kwargs']['lr_decay_epochs']))
        if steps > 0:
            lr = lr * (p['scheduler_kwargs']['lr_decay_rate'] ** steps)

    elif p['scheduler'] == 'poly':
        lambd = pow(1 - (epoch / p['train']['epochs']), 0.9)
        lr = lr * lambd

    elif p['scheduler'] == 'cosine':
        eta_min = lr * (p['scheduler_kwargs']['lr_decay_rate'] ** 3)
        lr = eta_min + (lr - eta_min) * (1 + math.cos(math.pi * epoch / p['epochs'])) / 2

    elif p['scheduler'] == 'constant':
        lr = lr

    else:
        raise ValueError('Invalid learning rate schedule {}'.format(p['scheduler']))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


def copy_file(src, dst):
    try:
        shutil.copy(src, dst)
    except shutil.SameFileError:
        pass


def read_config(path):
    with open(path) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    return data


def prepare_run(root_path, config_path):
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    tb_path = root_path.joinpath('runs/TL_{}'.format(current_time))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    writer = SummaryWriter(tb_path)
    copy_file(config_path, f'{tb_path}/config.yml')
    os.makedirs(f'runs/TL_{current_time}/imgs')
    return writer, device, current_time
