import argparse
import torch
import torch.nn.functional as F

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

from model import LargeGT
from data import LargeGTTokens, rand_train_test_idx, even_quantile_labels

import sys
import time
import datetime
import scipy.io
from numpy import mean as npmean
from numpy import std as npstd
import wandb
from multiprocessing import cpu_count


def train(model, loader, x, pos_enc, y, optimizer, device, conv_type, evaluator=None):
    model.train()

    counter = 1
    total_loss, total_correct, total_count = 0, 0, 0

    if conv_type == "global":
        for node_idx in loader:
            batch_size = len(node_idx)

            feat = x[node_idx] if torch.is_tensor(x) else x(node_idx)
            input = feat.to(device), pos_enc[node_idx].to(device), node_idx

            optimizer.zero_grad()
            out = model.to(device).global_forward(*input)
            loss = F.cross_entropy(out, y[node_idx].to(device))
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_size
            total_correct += out.argmax(dim=-1).cpu().eq(y[node_idx]).sum().item()
            total_count += batch_size

            counter += 1

    else:
        y_pred, y_true = [], []
        for seq, node_idx in loader:
            batch_size = len(node_idx)

            feat = x[node_idx] if torch.is_tensor(x) else x(node_idx)
            input = (
                seq.to(device),
                feat.to(device),
                pos_enc[node_idx].to(device),
                node_idx,
            )

            optimizer.zero_grad()
            out = model.to(device)(*input)
            loss = F.cross_entropy(out, y[node_idx].long().to(device))
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_size
            total_correct += out.argmax(dim=-1).cpu().eq(y[node_idx]).sum().item()
            total_count += batch_size

            counter += 1

            y_pred.append(torch.argmax(out, dim=1, keepdim=True).cpu())
            y_true.append(y[node_idx].unsqueeze(1))

        if evaluator is not None:
            acc = evaluator.eval(
                {
                    "y_true": torch.cat(y_true, dim=0),
                    "y_pred": torch.cat(y_pred, dim=0),
                }
            )["acc"]

            return total_loss / total_count, acc

    return total_loss / total_count, total_correct / total_count


def test(
    model, loader, x, pos_enc, y, device, conv_type, fast_eval=False, evaluator=None
):
    model.eval()

    counter = 1
    total_correct, total_count = 0, 0

    if conv_type == "global":
        for node_idx in loader:
            batch_size = len(node_idx)

            feat = x[node_idx] if torch.is_tensor(x) else x(node_idx)
            out = model.to(device).global_forward(
                feat.to(device), pos_enc[node_idx].to(device), node_idx
            )

            total_correct += out.argmax(dim=-1).cpu().eq(y[node_idx]).sum().item()
            total_count += batch_size

            if fast_eval and counter == len(loader) // 10:
                if total_correct / total_count < 0.8:
                    return 0
            counter += 1

    else:
        y_pred, y_true = [], []
        for seq, node_idx in loader:
            batch_size = len(node_idx)

            feat = x[node_idx] if torch.is_tensor(x) else x(node_idx)
            out = model.to(device)(
                seq.to(device), feat.to(device), pos_enc[node_idx].to(device), node_idx
            )

            total_correct += out.argmax(dim=-1).cpu().eq(y[node_idx]).sum().item()
            total_count += batch_size

            if fast_eval and counter == len(loader) // 10:
                if total_correct / total_count < 0.8:
                    return 0
            counter += 1

            y_pred.append(torch.argmax(out, dim=1, keepdim=True).cpu())
            y_true.append(y[node_idx].unsqueeze(1))

        if evaluator is not None:
            acc = evaluator.eval(
                {
                    "y_true": torch.cat(y_true, dim=0),
                    "y_pred": torch.cat(y_pred, dim=0),
                }
            )["acc"]

            return acc

    return total_correct / total_count


def create_run_name_with_timestamp(args, timestamp):
    run_name = "largegt_"

    for arg_name, arg_value in vars(args).items():
        run_name += f"{arg_name}_{arg_value}_"

    run_name += "timestamp_" + timestamp
    return run_name


def main(tstamp=0):
    parser = argparse.ArgumentParser(description="large")

    # data loading
    parser.add_argument(
        "--dataset",
        type=str,
        default="ogbn-products",
        choices=["ogbn-products", "snap-patents", "ogbn-papers100M"],
    )
    parser.add_argument("--data_root", type=str, default="data")

    # training
    parser.add_argument("--hetero_train_prop", type=float, default=0.5)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--test_batch_size", type=int, default=256)
    parser.add_argument("--test_freq", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=cpu_count() - 1)

    # NN
    parser.add_argument(
        "--conv_type", type=str, default="full", choices=["local", "global", "full"]
    )
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--global_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--num_heads", type=int, default=1)
    parser.add_argument("--attn_dropout", type=float, default=0)
    parser.add_argument("--ff_dropout", type=float, default=0.5)
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument("--num_centroids", type=int, default=4096)
    parser.add_argument("--no_bn", action="store_true")
    parser.add_argument("--norm_type", type=str, default="batch_norm")

    # eval
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--eval_epoch", type=int, default=100)
    parser.add_argument("--save_ckpt", action="store_true")
    parser.add_argument("--save_path", type=str, default="checkpoints")

    parser.add_argument("--sample_node_len", type=int, default=100)
    parser.add_argument("--project_name", default="test")
    parser.add_argument("--budget_hour", type=int, default=48)

    args = parser.parse_args()

    print(args)

    run_name = create_run_name_with_timestamp(args, tstamp)
    wandb.init(
        project=args.project_name,
        config=args,
        name=run_name,
        resume="allow",
        id=wandb.util.generate_id(),
        settings=wandb.Settings(start_method="fork"),
    )

    # convert int to boolean:
    args.skip = args.skip > 0

    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    if args.eval:
        ckpt = torch.load(
            f"checkpoints/ckpt_epoch{args.eval_epoch}.pt", map_location=device
        )

    data_root = args.data_root

    if args.dataset.startswith("ogbn"):
        dataset = PygNodePropPredDataset(name=args.dataset, root=data_root)
        dataset_new_tokenizer = LargeGTTokens(
            args.dataset + "_sample_node_len_" + str(args.sample_node_len),
            sample_node_len=args.sample_node_len,
        )
        num_classes = dataset.num_classes
        data = dataset[0]

        try:
            split_idx = dataset_new_tokenizer.split_idx
            x = dataset_new_tokenizer.X
            y = dataset_new_tokenizer.y.squeeze()
            num_nodes = y.shape[0]
            original_X = data.x
        except:
            split_idx = dataset.get_idx_split()
            x = data.x
            y = data.y.squeeze()
            num_nodes = data.num_nodes

        # Convert split indices to boolean masks and add them to `data`.
        for key, idx in split_idx.items():
            mask = torch.zeros(num_nodes, dtype=torch.bool)
            mask[idx] = True
            data[f"{key}_mask"] = mask

        assert args.batch_size <= len(split_idx["train"])

        if args.dataset == "ogbn-papers100M":
            evaluator = Evaluator(name="ogbn-papers100M")
        else:
            evaluator = None

    elif args.dataset == "snap-patents":
        dataset_new_tokenizer = LargeGTTokens(
            args.dataset + "_sample_node_len_" + str(args.sample_node_len),
            sample_node_len=args.sample_node_len,
        )
        num_classes = 5
        fulldata = scipy.io.loadmat(f"data/snap_patents.mat")
        edge_index = torch.tensor(fulldata["edge_index"], dtype=torch.long)

        num_nodes = int(fulldata["num_nodes"])
        node_feat = torch.tensor(fulldata["node_feat"].todense(), dtype=torch.float)

        years = fulldata["years"].flatten()
        label = even_quantile_labels(years, num_classes, verbose=False)
        label = torch.tensor(label, dtype=torch.long)

        class MyObject:
            pass

        data = MyObject()
        x = data.x = node_feat
        y = data.y = label
        data.num_features = data.x.shape[-1]

        data.edge_index = edge_index
        data.num_nodes = num_nodes

        train_idx, valid_idx, test_idx = rand_train_test_idx(
            y, train_prop=args.hetero_train_prop
        )
        split_idx = {"train": train_idx, "valid": valid_idx, "test": test_idx}
        evaluator = None

    if args.dataset == "ogbn-papers100M":
        try:
            data.num_nodes = num_nodes
        except:
            pass

    model = LargeGT(
        num_nodes=data.num_nodes,
        in_channels=data.num_features,
        hidden_channels=args.hidden_dim,
        out_channels=num_classes,
        global_dim=args.global_dim,
        num_layers=args.num_layers,
        heads=args.num_heads,
        ff_dropout=args.ff_dropout,
        attn_dropout=args.attn_dropout,
        skip=args.skip,
        conv_type=args.conv_type,
        num_centroids=args.num_centroids,
        no_bn=args.no_bn,
        norm_type=args.norm_type,
        sample_node_len=args.sample_node_len,
    )

    print("total params:", sum(p.numel() for p in model.parameters()))

    if args.conv_type == "local":
        pos_enc = x
    else:
        dataset_name_input = args.dataset

        if dataset_name_input == "ogbn-papers100M" and args.sample_node_len == 50:
            pos_enc = torch.randn(data.num_nodes, args.global_dim)
            ogb_node2vec = torch.load(
                f"data/{dataset_name_input}_data_dict.pt", map_location="cpu"
            )  # 128 dim
            node2vec_embd = ogb_node2vec["node2vec_embedding"]

            # https://github.com/snap-stanford/ogb/blob/master/examples/nodeproppred/papers100M/node2vec.py
            # Using the mapping from ogb node2vec example to assign pos_enc only to the labeled nodes
            all_original_split_idx = torch.cat(
                (split_idx["train"], split_idx["valid"], split_idx["test"])
            ).tolist()
            for i, idx in enumerate(all_original_split_idx):
                pos_enc[idx] = node2vec_embd[i]
        elif dataset_name_input == "ogbn-papers100M" and args.sample_node_len == 100:
            ogb_node2vec = torch.load(
                f"data/{dataset_name_input}_data_dict.pt", map_location="cpu"
            )  # 128 dim
            pos_enc = ogb_node2vec["node2vec_embedding"]
        else:
            pos_enc = torch.load(
                f"data/{dataset_name_input}_embedding_{args.global_dim}.pt",
                map_location="cpu",
            )

    if args.conv_type == "global":
        train_loader = torch.utils.data.DataLoader(
            split_idx["train"],
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        )
        valid_loader = torch.utils.data.DataLoader(
            split_idx["valid"],
            batch_size=args.test_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
        test_loader = torch.utils.data.DataLoader(
            split_idx["test"],
            batch_size=args.test_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
    else:
        if args.dataset == "ogbn-papers100M" and args.sample_node_len == 100:
            from functools import partial

            custom_collate = partial(
                dataset_new_tokenizer.collate, original_X=original_X
            )
        else:
            custom_collate = dataset_new_tokenizer.collate

        train_loader = torch.utils.data.DataLoader(
            split_idx["train"],
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=custom_collate,
        )
        valid_loader = torch.utils.data.DataLoader(
            split_idx["valid"],
            batch_size=args.test_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=custom_collate,
        )
        test_loader = torch.utils.data.DataLoader(
            split_idx["test"],
            batch_size=args.test_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=custom_collate,
        )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    test_start_epoch = 0

    valid_acc_final, test_acc_final, test_acc_highest = 0, 0, 0

    whole_start = time.time()
    for epoch in range(1, 1 + args.epochs):
        if time.time() - whole_start >= args.budget_hour * 60 * 60:
            print("Budget runtime has passed. Exiting.")
            sys.exit(0)  # Exit the program
        start = time.time()

        train_loss, train_acc = train(
            model,
            train_loader,
            x,
            pos_enc,
            y,
            optimizer,
            device,
            args.conv_type,
            evaluator,
        )
        train_time = time.time() - start
        print(
            f"Epoch: {epoch}, Train loss:{train_loss:.4f}, Train acc:{100*train_acc:.2f}, Epoch time: {train_time:.4f}, Train Mem:{torch.cuda.max_memory_allocated(device=device)/1e6:.0f} MB"
        )

        wandb.log(
            {"loss_train": train_loss, "acc_train": train_acc, "time": train_time}
        )

        if epoch > test_start_epoch and epoch % args.test_freq == 0:
            if args.save_ckpt:
                ckpt = {}
                ckpt["model"] = model.state_dict()

                torch.save(
                    ckpt, f"{args.save_path}/{args.dataset}_ckpt_epoch{epoch}.pt"
                )
                # ckpt = model.load_state_dict(torch.load('model.pt'))

            else:
                start = time.time()
                valid_acc = test(
                    model,
                    valid_loader,
                    x,
                    pos_enc,
                    y,
                    device,
                    args.conv_type,
                    False,
                    evaluator,
                )

                wandb.log({"acc_val": valid_acc})

                if args.dataset == "ogbn-products" and valid_acc < 0.0:
                    pass
                else:
                    fast_eval_flag = args.dataset == "ogbn-products"
                    fast_eval_flag = False

                    test_acc = test(
                        model,
                        test_loader,
                        x,
                        pos_enc,
                        y,
                        device,
                        args.conv_type,
                        fast_eval_flag,
                        evaluator,
                    )
                    test_time = time.time() - start
                    print(
                        f"Test acc: {100 * test_acc:.2f}, Val+Test time used: {test_time:.4f}"
                    )

                    if valid_acc > valid_acc_final:
                        valid_acc_final = valid_acc
                        test_acc_final = test_acc
                    if test_acc > test_acc_highest:
                        test_acc_highest = test_acc

                    wandb.log(
                        {
                            "acc_test": test_acc,
                            "acc_test_best": test_acc_final,
                            "acc_test_highest": test_acc_highest,
                        }
                    )

    wandb.finish()
    return valid_acc_final, test_acc_final, time.time() - whole_start


if __name__ == "__main__":
    # running for multiple times
    all_valid_acc = []
    all_test_acc = []
    all_time = []

    total_runs = 4
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    for run_number in range(total_runs):
        val_score, test_score, timee = main(timestamp)
        all_valid_acc.append(val_score)
        all_test_acc.append(test_score)
        all_time.append(timee)

    print("Mean valid acc: ", npmean(all_valid_acc), "s.d.: ", npstd(all_valid_acc))
    print("Mean test acc: ", npmean(all_test_acc), "s.d.: ", npstd(all_test_acc))
    print("Avg time taken for 4 runs: ", npmean(timee))
