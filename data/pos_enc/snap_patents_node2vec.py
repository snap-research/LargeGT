
import argparse

import torch
from torch_geometric.nn import Node2Vec

from ogb.nodeproppred import PygNodePropPredDataset
import scipy


def save_embedding(model, dim, dataset_name):
    root = '.'
    torch.save(model.embedding.weight.data.cpu(), f'{root}/{dataset_name}_embedding_{dim}.pt')


def main():
    parser = argparse.ArgumentParser(description='OGBN-Products (Node2Vec)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--embedding_dim', type=int, default=64)
    parser.add_argument('--walk_length', type=int, default=40)
    parser.add_argument('--context_size', type=int, default=20)
    parser.add_argument('--walks_per_node', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--log_steps', type=int, default=1)
    args = parser.parse_args()

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    #########################################
    dataset_name = 'snap-patents'

    fulldata = scipy.io.loadmat(f'../snap_patents.mat')
    edge_index = torch.tensor(fulldata['edge_index'], dtype=torch.long)
    node_feat = torch.tensor(fulldata['node_feat'].todense(), dtype=torch.float)
    num_nodes = int(fulldata['num_nodes'])
    

    model = Node2Vec(edge_index, args.embedding_dim, args.walk_length,
                     args.context_size, args.walks_per_node,
                     sparse=True).to(device)

    loader = model.loader(batch_size=args.batch_size, shuffle=True,
                          num_workers=4)
    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=args.lr)

    model.train()
    for epoch in range(1, args.epochs + 1):
        for i, (pos_rw, neg_rw) in enumerate(loader):
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()

            if (i + 1) % args.log_steps == 0:
                print(f'Epoch: {epoch:02d}, Step: {i+1:03d}/{len(loader)}, '
                      f'Loss: {loss:.4f}')

            if (i + 1) % 1000 == 0:  # Save model every 100 steps.
                save_embedding(model, args.embedding_dim, dataset_name)
        save_embedding(model, args.embedding_dim, dataset_name)


if __name__ == "__main__":
    main()