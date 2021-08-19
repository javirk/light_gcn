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


def from_meshio(mesh, dimensions=2):
    r"""Converts a :.msh file to a
    :class:`torch_geometric.data.Data` instance.

    Args:
        mesh (meshio.read): A :obj:`meshio` mesh.
    """

    if meshio is None:
        raise ImportError('Package `meshio` could not be found.')

    pos = torch.from_numpy(mesh.points).to(torch.float)
    if dimensions == 3:
        tetra = torch.from_numpy(mesh.cells_dict['tetra']).to(torch.long).t().contiguous()
        face = torch.from_numpy(mesh.cells_dict['triangle']).to(torch.long).t().contiguous()
        return Data(pos=pos, tetra=tetra, face=face)
    elif dimensions == 2:
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


def read_mesh(mesh_path, dimensions):
    mesh = meshio.read(
        mesh_path,  # string, os.PathLike, or a buffer/open file
    )
    data = from_meshio(mesh, dimensions=dimensions)
    if dimensions == 2:
        data = FaceToEdge(remove_faces=False)(data)
    elif dimensions == 3:
        data = TetraToEdge(remove_tetras=False)(data)
    else:
        raise ValueError(f'{dimensions} not supported')
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

def trace_ray(o, d, graph):
    if len(o.shape) == 1:
        o = o.unsqueeze(0)
    else:
        o = o.unsqueeze(1)

    dist_r_c = torch.norm(graph.centroids - o, dim=-1)  # [B x nodes]
    o_triangle = dist_r_c.argmin(1)
    triangles = graph.face.permute(1, 0)

    idx_origin_triangle = graph.triangle[o_triangle]

    combs = [[0, 1, 2], [0, 2, 1], [1, 2, 0]]
    idx_vertex = torch.zeros((o.shape[0], 2))  # Batch size
    for idx_p1, idx_p2, out in combs:
        p1 = graph.pos[idx_origin_triangle[:, idx_p1]]
        p2 = graph.pos[idx_origin_triangle[:, idx_p2]]

        intersection = line_ray_intersection_point(graph.centroids[o_triangle], d, p1, p2)

        if intersection.any():
            idx_vertex += intersection.unsqueeze(1) * idx_origin_triangle[:, [idx_p1, idx_p2]]

    next_triangle = torch.zeros((o.shape[0]), dtype=torch.long)
    for i, r in enumerate(idx_vertex):
        next_triangle[i] = torch.nonzero((triangles == r[0]).any(1) * (triangles == r[1]).any(1) *
                                         (triangles != idx_origin_triangle[i]).any(1))[0, 0]

    triangles_visited = o_triangle.clone().unsqueeze(1).long()
    # distances = torch.zeros_like(triangles_visited)

    triangles_visited = torch.cat([triangles_visited, next_triangle.unsqueeze(1)], dim=-1)
    # distances = torch.cat(
    #     [distances, torch.norm(centroids[o_triangle] - centroids[next_triangle.long()], dim=-1, keepdim=True)])

    graph.probs_triangle = graph.probs[triangles].mean(axis=1).permute(1, 0)

    vis = evolve(graph, triangles_visited.clone(), d)

    return vis


def line_ray_intersection_point(ray_origin, ray_direction, point1, point2):
    # Ray-Line Segment Intersection Test in 2D
    # http://bit.ly/1CoxdrG
    v1 = (ray_origin - point1)[:,:-1]
    v2 = (point2 - point1)[:,:-1]
    v3 = np.stack([-np.array(ray_direction[:, 1]), np.array(ray_direction[:, 0])], axis=1)
    dotprod = np.einsum('ij, ij -> i', v2, v3)
    t1 = np.cross(v2, v1, axis=1) / dotprod
    t2 = np.einsum('ij, ij -> i', v1, v3) / dotprod
    return torch.tensor((t1 >= 0) * (t2 >= 0) * (t2 <= 1))


def evolve(graph, visited, ray_direction):
    while True and visited.shape[-1] < 100:
        # last_triangle = visited[:, -2]
        current_triangle = visited[:, -1]
        current_neigh = graph.neighbours[current_triangle]

        if -1 in current_neigh:
            break

        y = graph.centroids[current_neigh] - graph.centroids[current_triangle].unsqueeze(1)  # B x Vecinos x Componentes
        if visited.shape[-1] > 10:
            momentum_vector = graph.centroids[current_triangle] - graph.centroids[visited[:, -10]]  # B x Componentes
        else:
            momentum_vector = ray_direction

        momentum = torch.einsum('bc,bvc->bv', momentum_vector, y) / (
                    momentum_vector.norm(dim=1) * y.norm(dim=(1, 2))).unsqueeze(1)

        probs_neighbours = torch.gather(graph.probs_triangle, 1, current_neigh) * momentum

        next_triangle = torch.gather(current_neigh, 1, probs_neighbours.argmax(1).unsqueeze(1)).squeeze(1)

        visited = torch.cat([visited, next_triangle.unsqueeze(1)], dim=-1)
    return visited