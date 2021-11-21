import torch
import torch.nn as nn
import libs.transforms as t
from libs.model import EvolutionModel
from libs.plots import plot_trajectory

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

mesh_type = '3D'
filename = 'meshes/sphere_coarse.msh'
near = 0
far = 1.5
N_samples = 10
r0 = torch.tensor([[0.8, 0., 0.], [0.8, 0.2, 0.]], device=device)
m0 = torch.tensor([[-1, 0.1, 0.1], [-1, 0.1, 0.1]], dtype=torch.float32, device=device)
m0 = m0 / m0.norm(dim=1, keepdim=True)
tr = [t.TetraToEdge(remove_tetras=False), t.TetraToNeighbors(), t.TetraCoordinates()]

ev = EvolutionModel(filename, mesh_type, n_steps=25, n_samples=N_samples, transforms=tr)
ev.to(device)
ev.train()

n_index = - 0.5 * ev.graph.pos.norm(dim=1) + 1.5

ev.compute_vars(n_index)

final_coords = ev(r0, m0)

"""Test backpropagation"""
crit = nn.MSELoss()
real = torch.randn_like(final_coords)
loss = crit(final_coords, real)
loss.backward()
assert ev.graph.n_index.grad is not None

fig, ax = plot_trajectory(final_coords, b_pos=0)
fig.show()