import meshio
import torch
import numpy as np
from torch_geometric.data import Data
from einops import rearrange
from libs.utils import wrap_to_2pi, batch_dot, gather_batch


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
        # face = torch.from_numpy(mesh.cells_dict['triangle']).to(torch.long).t().contiguous()
        return Data(pos=pos, tetra=tetra)
    elif mesh_type == '2D':
        face = torch.from_numpy(mesh.cells_dict['triangle']).to(torch.long).t().contiguous()
        return Data(pos=pos, face=face)


def to_meshio(graph):
    r"""Converts a :class:`torch_geometric.data.Data` instance to a
    :obj:`.msh format`.

    Args:
        graph (torch_geometric.data.Data): The data object.
    """

    if meshio is None:
        raise ImportError('Package `meshio` could not be found.')

    points = graph.pos.detach().cpu().numpy()
    tetra = graph.tetra.detach().t().cpu().numpy()

    cells = [("tetra", tetra)]

    return meshio.Mesh(points, cells)


def read_mesh(mesh_path, dimensions, transforms):
    mesh = meshio.read(
        mesh_path,  # string, os.PathLike, or a buffer/open file
    )
    graph = from_meshio(mesh, mesh_type=dimensions)
    for t in transforms:
        graph = t(graph)
    return graph


def get_neighbour_tetra(graph, current_tetra, exit_face_local):
    exit_face = graph.tetra_face[current_tetra, exit_face_local]  # Face number 2 in the local space of the tetrahedron
    return graph.face_tetra[exit_face]


def find_tetrahedron_point(data, point):
    # Very much from https://stackoverflow.com/questions/25179693/how-to-check-whether-the-point-is-in-the-tetrahedron-or-not
    n_p = point.shape[0]
    orir = torch.repeat_interleave(data.origin.unsqueeze(-1), n_p, dim=2)
    newp = torch.einsum('imk, kmj -> kij', data.ort_tetra, point.T - orir)
    val = torch.all(newp >= 0, dim=1) & torch.all(newp <= 1, dim=1) & (torch.sum(newp, dim=1) <= 1)
    id_tet, id_p = torch.nonzero(val, as_tuple=True)

    res = - torch.ones(n_p, dtype=id_tet.dtype, device=id_tet.device)  # Sentinel value
    res[id_p] = id_tet
    return res


def compute_vars(graph, n_index):
    num_nodes = graph.pos.shape[0]
    num_tetra = graph.tetra.shape[1]

    coords_vertex_tetra = rearrange(graph.pos[graph.tetra], 'v t c -> t c v')
    k = torch.cat((torch.ones((num_tetra, 1, 4)), coords_vertex_tetra), dim=1).inverse()
    assert n_index.shape[0] == num_nodes, 'n_index must have as many values as the number of nodes'

    ab = torch.bmm(torch.transpose(k, 1, 2), n_index[graph.tetra].T.unsqueeze(-1))
    a = ab[:, 0, 0]
    b = ab[:, 1:, 0]
    n = b / b.norm(dim=1, keepdim=True)
    graph.n = n
    # graph.k = k  # You don't need it actually
    graph.a = a
    graph.b = b
    graph.n_index = n_index
    return graph


def evolve_in_tetrahedron(graph, tetra_idx, rp, m):
    """ This is where the magic happens. Based on https://academic.oup.com/qjmam/article/65/1/87/1829302"""
    bs = m.shape[0]
    # k = graph.k[tetra_idx]
    a = graph.a[tetra_idx]
    b = graph.b[tetra_idx]
    # p = graph.pos[graph.tetra[:, tetra_idx]]
    n = graph.n[tetra_idx]
    # p = rearrange(p, 'p b c -> b p c')
    faces = graph.tetra_face[tetra_idx]
    faces = rearrange(faces, 'b f -> f b')

    mn = torch.cross(m, n)
    q = mn / torch.sqrt(batch_dot(mn, mn).unsqueeze(1))
    nq = torch.cross(n, q)
    mn_dot = batch_dot(m, n).unsqueeze(1)
    rc = rp - (batch_dot(rp, n) + a / b.norm(dim=1)).unsqueeze(1) * (n - (mn_dot * nq) / batch_dot(m, nq).unsqueeze(1))
    R = rc - rp

    phiR = torch.zeros((bs, 4), device=R.device)
    for it, face in enumerate(faces):
        vtx_face = graph.face_vertex[face]
        pos_vtx_face = graph.pos[vtx_face]
        i = pos_vtx_face[:, 0]
        j = pos_vtx_face[:, 1]
        k = pos_vtx_face[:, 2]

        M_L0 = (j[:, 1] - i[:, 1]) * (k[:, 2] - i[:, 2]) - (k[:, 1] - i[:, 1]) * (j[:, 2] - i[:, 2])
        M_L1 = (j[:, 2] - i[:, 2]) * (k[:, 0] - i[:, 0]) - (k[:, 2] - i[:, 2]) * (j[:, 0] - i[:, 0])
        M_L2 = (j[:, 0] - i[:, 0]) * (k[:, 1] - i[:, 1]) - (k[:, 0] - i[:, 0]) * (j[:, 1] - i[:, 1])
        M_L = torch.stack((M_L0, M_L1, M_L2), dim=1)
        Q_L = - batch_dot(i, M_L)

        c1 = - batch_dot(M_L, R)
        c2 = R.norm(dim=1) * batch_dot(M_L, m)
        c3 = batch_dot(M_L, rc) + Q_L

        # if c1 == c3:
        #    phi1 = -2*torch.atan(c1/c2)
        #    phi2 = phi1
        # else:
        phi1 = 2 * torch.atan((c2 + torch.sqrt(c1.pow(2) + c2.pow(2) - c3.pow(2))) / (c1 - c3))
        phi2 = 2 * torch.atan((c2 - torch.sqrt(c1.pow(2) + c2.pow(2) - c3.pow(2))) / (c1 - c3))
        phi1 = wrap_to_2pi(phi1)
        phi2 = wrap_to_2pi(phi2)
        phi_cat = torch.stack([phi1, phi2], dim=-1)
        phiR[:, it] = phi_cat.min(dim=1).values

    phiR = torch.nan_to_num(phiR, nan=10)
    phiE, idx_face = phiR.min(dim=1, keepdim=True)

    phiE += phiE * 1 / 100  # This is to make sure that the next point starts inside the next tetrahedron

    # New direction and position
    re = rc - torch.cos(phiE) * R + R.norm(dim=1) * torch.sin(phiE) * m
    me = torch.cos(phiE) * m + torch.sin(phiE) / R.norm(dim=1) * R

    # Face number
    hit_face = faces[idx_face.squeeze()].diag()  # Get the diagonal

    # Next tetrahedron
    local_idx_next = (graph.face_tetra[hit_face] != tetra_idx.unsqueeze(1)).nonzero(as_tuple=True)

    next_tetra = graph.face_tetra[hit_face][local_idx_next]

    distance = torch.norm(rp - re, dim=1)
    return next_tetra, re, me, distance


def evolve(graph, r0, m0, max_steps, device='cpu'):
    r = r0.clone()
    m = m0.clone()
    tetra_idx = find_tetrahedron_point(graph, r0)

    i = 0
    r_hist = [r]
    m_hist = [m]
    distances = [torch.zeros(r.shape[0], device=r.device)]
    tetra_hist = [tetra_idx]
    while i < max_steps and (tetra_idx != -1).all():
        tetra_idx, r, m, d = evolve_in_tetrahedron(graph, tetra_idx, r, m)
        r_hist.append(r)
        m_hist.append(m)
        distances.append(d)
        tetra_hist.append(tetra_idx)
        i += 1

    r_hist = torch.stack(r_hist, dim=1)
    m_hist = torch.stack(m_hist, dim=1)
    distances = torch.stack(distances, dim=1)
    tetra_hist = torch.stack(tetra_hist, dim=1)

    distances = torch.cumsum(distances, dim=1)

    return {'r_hist': r_hist, 'm_hist': m_hist, 'distances': distances, 'tetra_hist': tetra_hist}


def sample_rays(r_hist, distances, z_vals):
    tminusd = z_vals - distances.unsqueeze(1)
    tminusd_pos = tminusd.clone()
    tminusd_neg = tminusd.clone()
    tminusd_pos[tminusd_pos < 0] = 10
    tminusd_neg[tminusd_neg > 0] = -10

    idx_top_pos = torch.topk(tminusd_pos, k=1, largest=False, sorted=False)  # Take the smallest
    idx_top_neg = torch.topk(tminusd_neg, k=1, largest=True, sorted=False)  # Take the largest

    indices = torch.cat([idx_top_pos.indices, idx_top_neg.indices], dim=-1)
    values = torch.cat([idx_top_pos.values, idx_top_neg.values], dim=-1)

    # idx_top = torch.topk(tminusd, k=2, largest=False, sorted=False)
    coords = gather_batch(r_hist, indices)
    m = (coords[:, :, 1] - coords[:, :, 0]) / z_vals

    m = m / m.norm(dim=-1, keepdim=True)
    final_coords = coords[:, :, 0] + values[..., :1].repeat_interleave(3, dim=-1) * m

    return final_coords
