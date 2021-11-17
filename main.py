import torch
from libs.mesh_utils import read_mesh, evolve, compute_vars, sample_rays
import libs.transforms as transforms
from libs.plots import plot_trajectory


# Preparation
t = [transforms.TetraToEdge(remove_tetras=False), transforms.TetraToNeighbors(), transforms.TetraCoordinates()]

mesh_type = '3D'
filename = 'meshes/sphere_coarse.msh'
near = 0
far = 1.5
N_samples = 10
r0 = torch.tensor([[0.8, 0., 0.]])
m0 = torch.tensor([[-1, 0.1, 0.1]], dtype=torch.float32)
m0 = m0 / m0.norm(dim=1)
N_rays = r0.shape[0]

graph = read_mesh(filename, mesh_type, t)

n_index = - 0.1 * graph.pos.norm(dim=1) + 1.1

# Evolution
graph = compute_vars(graph, n_index)
evolution = evolve(graph, r0, m0, 200)

r_hist = evolution['r_hist']
m_hist = evolution['m_hist']
d = evolution['distances']

# Sampling
t_vals = torch.linspace(0.1, 1., steps=10)
z_vals = near * (1. - t_vals) + far * t_vals
z_vals = z_vals.expand([N_rays, N_samples]).unsqueeze(-1)

final_coords = sample_rays(r_hist, d, z_vals)

fig, ax = plot_trajectory(final_coords, b_pos=0)
fig.show()