import torch
from torch_geometric.utils import to_undirected


class TetraToEdge(object):
    r"""Converts mesh tetras :obj:`[4, num_tetras]` to edge indices
    :obj:`[2, num_edges]`.
    Args:
        remove_tetras (bool, optional): If set to :obj:`False`, the tetra tensor
            will not be removed.
    """

    def __init__(self, remove_tetras=True):
        self.remove_tetras = remove_tetras

    def __call__(self, data):
        if data.tetra is not None:
            tetra = data.tetra
            edge_index = torch.cat([tetra[:2], tetra[1:3, :], tetra[-2:], tetra[::2], tetra[::3], tetra[1::2]], dim=1)
            edge_index = to_undirected(edge_index, num_nodes=data.num_nodes)

            data.edge_index = edge_index
            if self.remove_tetras:
                data.tetra = None

        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)

class GetNeighbours2D(object):
    def __init__(self):
        pass

    def __call__(self, data):
        neighbours = torch.empty((data.triangle.shape[0], 3), dtype=torch.long)
        combs = [[0, 1, 2], [0, 2, 1], [1, 2, 0]]
        for i_face, face in enumerate(data.triangle):
            neighbours_triangle = []
            for i1, i2, out in combs:
                v1 = face[i1]
                v2 = face[i2]
                vout = face[out]

                edge_neighbour = torch.nonzero(
                    (data.triangle == v1).any(1) * (data.triangle == v2).any(1) * (data.triangle != vout).all(1))[:, 0]

                if len(edge_neighbour) == 0:
                    edge_neighbour = -1
                elif len(edge_neighbour) > 1:
                    raise ValueError('Something is wrong')
                else:
                    edge_neighbour = edge_neighbour.item()

                neighbours_triangle.append(edge_neighbour)
            neighbours[i_face] = torch.tensor(neighbours_triangle, dtype=torch.long)
        data.neighbours = neighbours
        return data

class GetCentroids2D(object):
    def __init__(self):
        pass

    def __call__(self, data):
        triangles_coords = data.pos[data.triangle]
        centroids = torch.sum(triangles_coords, dim=1) / 3
        data.centroids = centroids
        return data