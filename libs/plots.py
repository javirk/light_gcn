import matplotlib.tri as tri
import matplotlib.pyplot as plt
import torch
from itertools import cycle
from einops import rearrange
from libs.utils import transfer_batch_to_device


def plot_graph(graph_data, fields=None, out_file=None):
    # assert graph_data.face is not None
    graph_data = transfer_batch_to_device(graph_data, 'cpu')

    triang = tri.Triangulation(graph_data.pos[:, 0], graph_data.pos[:, 1], graph_data.face.permute(1, 0))
    plt.figure()
    if fields is None:
        plt.tripcolor(triang, graph_data.y)
    else:
        fields = fields.to('cpu')
        plt.tripcolor(triang, fields[:, 0])

    if out_file is not None:
        plt.savefig(out_file)
        plt.close('all')


def plot_mesh_tb(graph_data, fields, writer, label, step):
    vertex = torch.zeros((1, 221, 3))
    vertex[0, :, :2] = graph_data.pos

    face = rearrange(graph_data.face, 'd (f b) -> b f d', b=1)

    colors = fields.clone().unsqueeze(0)
    colors = (colors.repeat(1, 1, 3).clip(min=0, max=1) * 255).int()

    writer.add_mesh(label, vertices=vertex, faces=face, colors=colors, global_step=step)


def plot_trajectory(r_hist, m_hist=None, b_pos=0):
    r_hist = r_hist.detach().cpu().numpy()
    cycol = cycle("bgrcmykw")

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    for i in range(r_hist.shape[1]):
        col = next(cycol)
        ax.scatter(r_hist[b_pos, i, 0], r_hist[b_pos, i, 1], r_hist[b_pos, i, 2], color=col)
        if m_hist is not None:
            ax.quiver(*r_hist[b_pos, i], m_hist[b_pos, i, 0] / 20, m_hist[b_pos, i, 1] / 20, m_hist[b_pos, i, 2] / 20,
                      colors=col)

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

    return fig, ax