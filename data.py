import torch

import numpy as np
import os
import os.path
import time

from data_utils import (
    get_ogbn_products_with_splits,
    get_snap_patents,
    get_ogbn_papers100M_with_splits,
    get_data_pt_file,
)


def rand_train_test_idx(label, train_prop=0.5, valid_prop=0.25, ignore_negative=True):
    """randomly splits label into train/valid/test splits"""
    if ignore_negative:
        labeled_nodes = torch.where(label != -1)[0]
    else:
        labeled_nodes = label

    n = labeled_nodes.shape[0]
    train_num = int(n * train_prop)
    valid_num = int(n * valid_prop)

    perm = torch.as_tensor(np.random.permutation(n))

    train_indices = perm[:train_num]
    val_indices = perm[train_num : train_num + valid_num]
    test_indices = perm[train_num + valid_num :]

    if not ignore_negative:
        return train_indices, val_indices, test_indices

    train_idx = labeled_nodes[train_indices]
    valid_idx = labeled_nodes[val_indices]
    test_idx = labeled_nodes[test_indices]

    return train_idx, valid_idx, test_idx


def even_quantile_labels(vals, nclasses=5, verbose=True):
    """partitions vals into nclasses by a quantile based split,
    where the first class is less than the 1/nclasses quantile,
    second class is less than the 2/nclasses quantile, and so on

    vals is np array
    returns an np array of int class labels
    """
    label = -1 * np.ones(vals.shape[0], dtype=int)
    interval_lst = []
    lower = -np.inf
    for k in range(nclasses - 1):
        upper = np.nanquantile(vals, (k + 1) / nclasses)
        interval_lst.append((lower, upper))
        inds = (vals >= lower) * (vals < upper)
        label[inds] = k
        lower = upper
    label[vals >= lower] = nclasses - 1
    interval_lst.append((lower, np.inf))
    if verbose:
        print("Class Label Intervals:")
        for class_idx, interval in enumerate(interval_lst):
            print(f"Class {class_idx}: [{interval[0]}, {interval[1]})]")
    return label


def get_dataset_with_splits(dataset):
    # print(dataset)
    if dataset == "ogbn-products":
        (
            adj,
            features,
            labels,
            idx_train,
            idx_val,
            idx_test,
        ) = get_ogbn_products_with_splits()
    elif dataset == "snap-patents":
        adj, features, labels, idx_train, idx_val, idx_test = get_snap_patents()
    elif dataset == "ogbn-papers100M":
        (
            adj,
            features,
            labels,
            idx_train,
            idx_val,
            idx_test,
        ) = get_ogbn_papers100M_with_splits()

    return (adj, features, labels, idx_train, idx_val, idx_test)


class LargeGTTokens(torch.utils.data.Dataset):
    def __init__(self, name, sample_node_len=50, seed=0):
        super(LargeGTTokens, self).__init__()

        start = time.time()
        print("[I] Loading dataset %s..." % (name))
        self.name = name
        self.sample_node_len = sample_node_len

        file_path = "data/" + self.name + ".pt"

        if os.path.exists(file_path):
            print("processed data file exists, loading...")
            data_list = torch.load(file_path)
        else:
            print("processed data file does not exists, preparing...")
            data_list = get_data_pt_file(
                name, get_dataset_with_splits(name.split("_")[0]), self.sample_node_len
            )

        try:
            self.nodes_in_seq = torch.tensor(data_list[0])
            self.X = torch.tensor(data_list[1], dtype=torch.float32)
            self.hop2token_feats = torch.tensor(data_list[2], dtype=torch.float32)
        except KeyError:
            self.nodes_in_seq = torch.tensor(data_list["nodes_in_seq"])
            self.X = torch.tensor(data_list["node_feat"])
            self.hop2token_feats = torch.load(
                file_path.replace(".pt", "_hop2token_feats.pt")
            )

            self.y = torch.tensor(data_list["label"])
            self.split_idx = data_list["split_idx"]

        self.token_len = self.nodes_in_seq.shape[1]
        self.input_dim = self.X.shape[-1]

        self.data_token_len = self.token_len * 3

        del data_list
        print("[I] Data load time: {:.4f}s".format(time.time() - start))

    def collate(self, samples, original_X=None):
        return self.collate_new(samples, original_X)

    def collate_new_slow(self, batch):
        mini_batch_size = len(batch)

        seq = torch.empty(mini_batch_size, self.token_len * 3, self.input_dim)

        for i, node in enumerate(batch):
            j = 0
            for sampled_node in self.nodes_in_seq[node]:
                seq[i, j] = self.X[sampled_node]
                seq[i, j + 1] = self.hop2token_feats[sampled_node, 0]
                seq[i, j + 2] = self.hop2token_feats[sampled_node, 1]
                j += 3

        return seq, torch.tensor(batch)

    def collate_new(self, batch, original_X):
        mini_batch_size = len(batch)
        seq = torch.empty(mini_batch_size, self.token_len * 3, self.input_dim)

        sampled_nodes = torch.stack([self.nodes_in_seq[node] for node in batch])

        i, j = torch.meshgrid(
            torch.arange(mini_batch_size), torch.arange(self.token_len), indexing="ij"
        )

        i = i.flatten()
        j = j.flatten() * 3
        sampled_nodes = sampled_nodes.flatten()

        if original_X is not None:
            seq[i, j] = original_X[sampled_nodes]
        else:
            seq[i, j] = self.X[sampled_nodes]
        seq[i, j + 1] = self.hop2token_feats[sampled_nodes, 0]
        seq[i, j + 2] = self.hop2token_feats[sampled_nodes, 1]

        return seq, torch.tensor(batch)
