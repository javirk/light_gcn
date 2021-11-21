import meshio
import torch
from torch_geometric.data import Data


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