import torch
from tqdm import tqdm
import random
import time
import numpy as np
import scipy.sparse as sp
from multiprocessing import Pool, cpu_count

from ogb.nodeproppred import DglNodePropPredDataset

# global vars used in data sampling steps in create_node_ids()
all_1hop_indices = None
all_2hop_indices = None
all_nodes_set = None
seq_length = None


def get_ogbn_products_with_splits():
    dataset = DglNodePropPredDataset(name="ogbn-products")

    g = dataset[0][0]
    split_idx = dataset.get_idx_split()
    adj = g.adj_external(scipy_fmt="csr")
    features = g.ndata["feat"]
    labels = torch.nn.functional.one_hot(
        dataset[0][1].view(-1), dataset[0][1].max() + 1
    )
    idx_train = split_idx["train"]
    idx_val = split_idx["valid"]
    idx_test = split_idx["test"]

    return adj, features, labels, idx_train, idx_val, idx_test


def get_snap_patents_with_splits():
    import scipy

    fulldata = scipy.io.loadmat(f"data/snap_patents.mat")
    edge_index = torch.tensor(fulldata["edge_index"], dtype=torch.long)

    num_nodes = int(fulldata["num_nodes"])
    features = torch.tensor(fulldata["node_feat"].todense(), dtype=torch.float)

    adj = sp.csr_matrix(
        (torch.ones(edge_index.shape[1]), edge_index),
        shape=(num_nodes, num_nodes),
        dtype=np.int64,
    )

    labels, idx_train, idx_val, idx_test = (
        torch.rand(1),
        torch.rand(1),
        torch.rand(1),
        torch.rand(1),
    )
    return adj, features, labels, idx_train, idx_val, idx_test


def get_ogbn_papers100M_with_splits():
    dataset = DglNodePropPredDataset(name="ogbn-papers100M")

    g = dataset[0][0]
    split_idx = dataset.get_idx_split()
    adj = g.adj_external(scipy_fmt="csr")
    features = g.ndata["feat"]

    labels = torch.rand(1)  # dummy value as it is not required for GOAT2 code
    idx_train = split_idx["train"]
    idx_val = split_idx["valid"]
    idx_test = split_idx["test"]

    return adj, features, labels, idx_train, idx_val, idx_test


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


# for one node; to be exectuted in parallel
def get_node_ids_for_all_seq(i):
    """Returns node ids for all seqs, sampled from 1/2 hop neighborhood"""
    # i, hop1indices, hop2indices, all_nodes_set, seq_length = args
    global all_1hop_indices, all_2hop_indices, all_nodes_set, seq_length

    hop1_neighbors = all_1hop_indices[i].tolist()
    hop2_neighbors = all_2hop_indices[i].tolist()

    all_applicable_neighbors = hop1_neighbors + hop2_neighbors

    if len(all_applicable_neighbors) < seq_length - 1:
        if len(all_applicable_neighbors) == 0:
            sampled_neighbors = random.sample(all_nodes_set, seq_length - 1)
        else:
            repeat_count = ((seq_length - 1) // len(all_applicable_neighbors)) + 1
            repeated_neighbor_list = all_applicable_neighbors * repeat_count
            sampled_neighbors = repeated_neighbor_list[: seq_length - 1]
    else:
        # for cases where node has >= 99 hop1 and hop2 neighbors
        sampled_neighbors = random.sample(all_applicable_neighbors, seq_length - 1)

    return [i] + sampled_neighbors


def create_node_ids(X, adj_matrix, for_nagphormer=False, sample_node_len=0):
    global all_1hop_indices, all_2hop_indices, all_nodes_set, seq_length

    N = X.size(0)
    all_nodes_set = list(set(range(N)))
    seq_length = sample_node_len

    print("multiset length for sampling: ", seq_length)

    t0 = time.time()
    print("multiplying csr matrix to itself...")

    adj_matrix_2hop = adj_matrix @ adj_matrix
    print("Done", time.time() - t0)
    print("getting all 1 hop indices...")
    all_1hop_indices = np.split(adj_matrix.indices, adj_matrix.indptr)[1:-1]
    print("Done\ngetting all 2 hop indices...")
    all_2hop_indices = np.split(adj_matrix_2hop.indices, adj_matrix_2hop.indptr)[1:-1]
    print("Done!")

    # Parallelize the loop and collect the results
    with Pool(cpu_count() - 1) as pool:
        args_list = [i for i in range(N)]
        node_ids_for_all_seq = list(
            tqdm(pool.imap(get_node_ids_for_all_seq, args_list), total=N)
        )

    print(
        "Retrieved node ids for all seqs, now preparing hop2token feats for hop=2... "
    )

    t0 = time.time()

    if for_nagphormer:
        hop2token_range = 10
    else:
        hop2token_range = 3

    # Use 1-hop 2-hop 3-hop feature information from hop2token of nagphormer
    hop2token_feats = torch.empty(X.size(0), 1, hop2token_range, X.size(1))

    renormalize = True  # required for preparing hop2token feats
    if renormalize:
        adj_matrix = adj_matrix + sp.eye(adj_matrix.shape[0])
        D1 = np.array(adj_matrix.sum(axis=1)) ** (-0.5)
        D2 = np.array(adj_matrix.sum(axis=0)) ** (-0.5)
        D1 = sp.diags(D1[:, 0], format="csr")
        D2 = sp.diags(D2[0, :], format="csr")

        A = adj_matrix.dot(D1)
        A = D2.dot(A)
        adj_matrix = A

    adj_matrix = sparse_mx_to_torch_sparse_tensor(adj_matrix)
    tmp = X + torch.zeros_like(X)
    for i in range(hop2token_range):
        # only preparing 3-hop for hop2token feats
        # expts could use upto 2-hop only
        tmp = torch.matmul(adj_matrix, tmp)
        for index in range(X.shape[0]):
            hop2token_feats[index, 0, i, :] = tmp[index]
    hop2token_feats = hop2token_feats.squeeze()

    print("DONE!", time.time() - t0)
    del adj_matrix, adj_matrix_2hop, tmp

    print("Saving now ...")
    return node_ids_for_all_seq, hop2token_feats


def get_data_pt_file(name, data_args, sample_node_len):
    # adj_matrix, X, labels, idx_train, idx_val, idx_test = data_args
    adj_matrix = data_args[0]
    X = torch.tensor(data_args[1], dtype=torch.float32)
    labels = torch.tensor(data_args[2])
    idx_train = torch.tensor(data_args[3])
    idx_val = torch.tensor(data_args[4])
    idx_test = torch.tensor(data_args[5])

    if name.split("_")[1] == "nagphormer":
        file_save_name = "dataset/" + name + ".pt"
        for_nagphormer = True
    else:
        for_nagphormer = False
        if name.split("_")[0] == "ogbn-products":
            file_save_name = (
                "data/ogbn-products"
                + "_sample_node_len_"
                + str(sample_node_len)
                + ".pt"
            )
        elif name.split("_")[0] == "snap-patents":
            file_save_name = (
                "data/snap-patents" + "_sample_node_len_" + str(sample_node_len) + ".pt"
            )
        elif name.split("_")[0] == "ogbn-papers100M":
            file_save_name = (
                "data/ogbn-papers100M"
                + "_sample_node_len_"
                + str(sample_node_len)
                + ".pt"
            )
        else:
            raise Exception

    t0 = time.time()
    node_ids_for_all_seq, hop2token_feats = create_node_ids(
        X, adj_matrix, for_nagphormer, sample_node_len
    )
    torch.save(
        (
            node_ids_for_all_seq,
            X,
            hop2token_feats,
            adj_matrix,
            labels,
            idx_train,
            idx_val,
            idx_test,
        ),
        file_save_name,
        pickle_protocol=4,
    )
    print("total time taken: ", time.time() - t0)

    return (
        node_ids_for_all_seq,
        X,
        hop2token_feats,
        adj_matrix,
        labels,
        idx_train,
        idx_val,
        idx_test,
    )
