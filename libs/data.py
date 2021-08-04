import torch
from torch_geometric.data import Dataset
from libs.utils import read_mesh


class PlanarDataset(Dataset):
    def __init__(self, mesh_path, dimensions):
        super().__init__()
        self.dimensions = dimensions
        self.data = read_mesh(mesh_path, dimensions)
        self.len_dataset = 10000

    def __len__(self):
        return self.len_dataset


    def __getitem__(self, item):
        data = self.data.clone()  # This is very important!
        o = torch.rand(self.dimensions) * (-2) + 1  # Range [-1, 1]
        d = torch.rand(self.dimensions) * (-2) + 1
        d = d / d.norm()

        features = torch.cat([o, d]).unsqueeze(0)
        features = features.repeat(data.num_nodes, 1)
        features = torch.cat([data.pos, features], dim=1)

        # Compute the distances (y). This is a shorthand for the cross product, which is not defined in 2D
        dirs = d.unsqueeze(0).repeat(data.num_nodes, 1)
        v = o - data.pos
        distn = torch.abs(dirs[:, 0] * v[:, 1] - dirs[:, 1] * v[:, 0]).unsqueeze(-1)  # So the final shape is [num_nodes, 1]

        data.x = features
        data.dist = distn

        return data


    def _download(self):
        pass

    def _process(self):
        pass

if __name__ == '__main__':
    dataset = PlanarDataset('../meshes/planar_coarse.msh', 2)
    a = dataset[0]