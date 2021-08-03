import os
import torch
import torch.nn as nn
import argparse
from torch_geometric.data import DataLoader
from libs.model import Model
from libs.data import PlanarDataset
from libs.utils import transfer_batch_to_device, get_optimizer, prepare_run, read_config, adjust_learning_rate
from libs.plots import plot_mesh_tb, plot_graph
from pathlib import Path


def main():
    writer, device, current_time = prepare_run(root_path, FLAGS.config)
    ckpt_path = root_path.joinpath('ckpts', f'{current_time}.pth')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = PlanarDataset('meshes/planar_coarse.msh', '2D')
    train_loader = DataLoader(dataset, batch_size=config['train']['batch_size'], shuffle=True, num_workers=0)
    test_loader = DataLoader(dataset, batch_size=1)

    model = Model(config['model']['input_features'], config['model']['output_features'])
    print(f"Lets use {torch.cuda.device_count()} GPUs!")
    model = nn.DataParallel(model)
    model.to(device)
    model.train()

    opt = get_optimizer(config, model.parameters())
    criterion = nn.MSELoss()

    for epoch in range(config['train']['epochs']):
        lr = adjust_learning_rate(config, opt, epoch)
        running_loss = 0.
        for i, batch in enumerate(train_loader):
            opt.zero_grad()
            batch = transfer_batch_to_device(batch, device)
            out = model(batch)
            loss = criterion(out, batch.dist)
            loss.backward()
            opt.step()
            running_loss += loss.item()

        # Validation
        b_test = next(iter(test_loader))
        b_test = transfer_batch_to_device(b_test, device)
        with torch.no_grad():
            out_test = model(b_test)

        print(f'Epoch {epoch}: loss = {running_loss/i}')
        # plot_graph(b_test, out_test, out_file=f'runs/TL_{current_time}/imgs/epoch_{epoch}.png')
        writer.add_scalar('Train', running_loss / i, epoch)
        plot_mesh_tb(b_test, out_test, writer, 'Train', epoch)
        plot_mesh_tb(b_test, b_test.dist, writer, 'GT', epoch)

        ckpt = {'optimizer': opt.state_dict(), 'model': model.state_dict(), 'epoch': epoch + 1}
        torch.save(ckpt, ckpt_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--config',
                        default='configs/config.yml',
                        type=str,
                        help='Path to the config file')

    FLAGS, unparsed = parser.parse_known_args()
    config = read_config(FLAGS.config)

    root_path = Path(__file__).resolve().parents[0]
    main()