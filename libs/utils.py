import os

import meshio
import torch
import math
import yaml
import shutil
import numpy as np
from torch_geometric.data import Data
from libs.transforms import TetraToEdge
from torch_geometric.transforms import FaceToEdge
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter


def from_meshio(mesh, mesh_type='2D'):
    r"""Converts a :.msh file to a
    :class:`torch_geometric.data.Data` instance.

    Args:
        mesh (meshio.read): A :obj:`meshio` mesh.
    """

    if meshio is None:
        raise ImportError('Package `meshio` could not be found.')

    pos = torch.from_numpy(mesh.points).to(torch.float)
    if mesh_type == '3D':
        tetra = torch.from_numpy(mesh.cells_dict['tetra']).to(torch.long).t().contiguous()
        return Data(pos=pos, tetra=tetra)
    elif mesh_type == '2D':
        face = torch.from_numpy(mesh.cells_dict['triangle']).to(torch.long).t().contiguous()
        return Data(pos=pos[:, :2], face=face)


def to_meshio(data):
    r"""Converts a :class:`torch_geometric.data.Data` instance to a
    :obj:`.msh format`.

    Args:
        data (torch_geometric.data.Data): The data object.
    """

    if meshio is None:
        raise ImportError('Package `meshio` could not be found.')

    points = data.pos.detach().cpu().numpy()
    tetra = data.tetra.detach().t().cpu().numpy()

    cells = [("tetra", tetra)]

    return meshio.Mesh(points, cells)


def read_mesh(mesh_path, mesh_type):
    mesh = meshio.read(
        mesh_path,  # string, os.PathLike, or a buffer/open file
    )
    data = from_meshio(mesh, mesh_type=mesh_type)
    if mesh_type == '2D':
        data = FaceToEdge(remove_faces=False)(data)
    else:
        data = TetraToEdge()(data)

    return data


def transfer_batch_to_device(batch, device):
    for k, v in batch:
        if hasattr(v, 'to'):
            batch[k] = v.to(device)
    return batch


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
        lambd = pow(1 - (epoch / p['epochs']), 0.9)
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