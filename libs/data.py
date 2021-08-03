import torch
from torch_geometric.data import Data, Batch, Dataset
from libs.utils import read_mesh


class PlanarDataset(Dataset):
    def __init__(self, mesh_path, mesh_type):
        super().__init__()
        self.data = read_mesh(mesh_path, mesh_type)
        self.len_dataset = 10000
        self.origins = torch.rand((self.len_dataset, 2)) * (-2) + 1  # Range [-1, 1]
        self.directions = torch.rand((self.len_dataset, 2)) * (-2) + 1
        self.directions = self.directions / self.directions.norm(dim=1, keepdim=True)

    def __len__(self):
        return self.len_dataset


    def __getitem__(self, item):
        item = 0
        data = self.data.clone()  # This is very important!
        o = self.origins[item]
        d = self.directions[item]

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
    dataset = PlanarDataset('../meshes/planar_coarse.msh', '2D')
    a = dataset[0]
    from libs.plots import plot_graph
    plot_graph(a)