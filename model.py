import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear

from codebook import VectorQuantizerEMA
from einops import rearrange
from local_module import LocalModule


class LargeGTLayer(MessagePassing):
    def __init__(
        self,
        in_channels,
        out_channels,
        global_dim,
        num_nodes,
        heads=1,
        concat=True,
        beta=False,
        dropout=0.0,
        edge_dim=None,
        bias=True,
        skip=True,
        conv_type="local",
        num_centroids=None,
        sample_node_len=100,
        **kwargs,
    ):
        kwargs.setdefault("aggr", "add")
        super(LargeGTLayer, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.beta = beta and skip
        self.skip = skip
        self.concat = concat
        self.dropout = dropout
        self.edge_dim = edge_dim
        self.conv_type = conv_type
        self.num_centroids = num_centroids
        self._alpha = None

        self.sample_node_len = sample_node_len

        self.lin_key = Linear(in_channels, heads * out_channels)
        self.lin_query = Linear(in_channels, heads * out_channels)
        self.lin_value = Linear(in_channels, heads * out_channels)

        if concat:
            self.lin_skip = Linear(in_channels, heads * out_channels, bias=bias)
            if self.beta:
                self.lin_beta = Linear(3 * heads * out_channels, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter("lin_beta", None)
        else:
            self.lin_skip = Linear(in_channels, out_channels, bias=bias)
            if self.beta:
                self.lin_beta = Linear(3 * out_channels, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter("lin_beta", None)

        self.local_module = LocalModule(
            seq_len=self.sample_node_len * 3,
            input_dim=in_channels,
            n_layers=1,
            num_heads=heads,
            hidden_dim=out_channels,
        )

        if self.conv_type != "local":
            self.vq = VectorQuantizerEMA(num_centroids, global_dim, decay=0.99)
            c = torch.randint(0, num_centroids, (num_nodes,), dtype=torch.short)
            self.register_buffer("c_idx", c)
            self.attn_fn = F.softmax

            self.lin_proj_g = Linear(in_channels, global_dim)
            self.lin_key_g = Linear(global_dim * 2, heads * out_channels)
            self.lin_query_g = Linear(global_dim * 2, heads * out_channels)
            self.lin_value_g = Linear(global_dim, heads * out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_key.reset_parameters()
        self.lin_query.reset_parameters()
        self.lin_value.reset_parameters()
        self.lin_skip.reset_parameters()
        if self.beta:
            self.lin_beta.reset_parameters()

    def forward(self, seq, x, pos_enc=None, batch_idx=None):
        if self.conv_type == "local":
            out = self.local_forward(seq)

        elif self.conv_type == "global":
            out = self.global_forward(x[: len(batch_idx)], pos_enc, batch_idx)

        elif self.conv_type == "full":
            out_local = self.local_forward(seq)
            out_global = self.global_forward(x[: len(batch_idx)], pos_enc, batch_idx)
            out = torch.cat([out_local, out_global], dim=1)

        else:
            raise NotImplementedError

        return out

    def global_forward(self, x, pos_enc, batch_idx):
        d, h = self.out_channels, self.heads
        scale = 1.0 / math.sqrt(d)

        q_x = torch.cat([self.lin_proj_g(x), pos_enc], dim=1)

        k_x = self.vq.get_k()
        v_x = self.vq.get_v()

        q = self.lin_query_g(q_x)
        k = self.lin_key_g(k_x)
        v = self.lin_value_g(v_x)

        q, k, v = map(lambda t: rearrange(t, "n (h d) -> h n d", h=h), (q, k, v))
        dots = torch.einsum("h i d, h j d -> h i j", q, k) * scale

        c, c_count = self.c_idx.unique(return_counts=True)

        centroid_count = torch.zeros(self.num_centroids, dtype=torch.long).to(x.device)
        centroid_count[c.to(torch.long)] = c_count

        dots += torch.log(centroid_count.view(1, 1, -1))

        attn = self.attn_fn(dots, dim=-1)
        attn = F.dropout(attn, p=self.dropout, training=self.training)

        out = torch.einsum("h i j, h j d -> h i d", attn, v)
        out = rearrange(out, "h n d -> n (h d)")

        # Update the centroids
        if self.training:
            x_idx = self.vq.update(q_x)
            self.c_idx[batch_idx] = x_idx.squeeze().to(torch.short)

        return out

    def local_forward(self, seq):
        return self.local_module(seq)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.in_channels}, "
            f"{self.out_channels}, heads={self.heads})"
        )


class LargeGT(torch.nn.Module):
    def __init__(
        self,
        num_nodes,
        in_channels,
        hidden_channels,
        out_channels,
        global_dim,
        num_layers,
        heads,
        ff_dropout,
        attn_dropout,
        skip,
        conv_type,
        num_centroids,
        no_bn,
        norm_type,
        sample_node_len,
    ):
        super(LargeGT, self).__init__()

        if norm_type == "batch_norm":
            norm_func = nn.BatchNorm1d
        elif norm_type == "layer_norm":
            norm_func = nn.LayerNorm

        if no_bn:
            self.fc_in = nn.Sequential(
                nn.Linear(in_channels, hidden_channels),
                nn.ReLU(),
                nn.Dropout(ff_dropout),
                nn.Linear(hidden_channels, hidden_channels),
            )
            self.fc_in_seq = nn.Sequential(
                nn.Linear(in_channels, hidden_channels),
                nn.ReLU(),
                nn.Dropout(ff_dropout),
                nn.Linear(hidden_channels, hidden_channels),
            )
        else:
            self.fc_in = nn.Sequential(
                nn.Linear(in_channels, hidden_channels),
                norm_func(hidden_channels),
                nn.ReLU(),
                nn.Dropout(ff_dropout),
                nn.Linear(hidden_channels, hidden_channels),
            )
            self.fc_in_seq = nn.Sequential(
                nn.Linear(in_channels, hidden_channels),
                # norm_func(hidden_channels),
                nn.ReLU(),
                nn.Dropout(ff_dropout),
                nn.Linear(hidden_channels, hidden_channels),
            )
        self.convs = torch.nn.ModuleList()
        self.ffs = torch.nn.ModuleList()

        assert num_layers == 1
        for _ in range(num_layers):
            self.convs.append(
                LargeGTLayer(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    global_dim=global_dim,
                    num_nodes=num_nodes,
                    heads=heads,
                    dropout=attn_dropout,
                    skip=skip,
                    conv_type=conv_type,
                    num_centroids=num_centroids,
                    sample_node_len=sample_node_len,
                )
            )
            h_times = 2 if conv_type == "full" else 1

            if no_bn:
                self.ffs.append(
                    nn.Sequential(
                        nn.Linear(
                            h_times * hidden_channels * heads, hidden_channels * heads
                        ),
                        nn.ReLU(),
                        nn.Dropout(ff_dropout),
                        nn.Linear(hidden_channels * heads, hidden_channels),
                        nn.ReLU(),
                        nn.Dropout(ff_dropout),
                    )
                )
            else:
                self.ffs.append(
                    nn.Sequential(
                        nn.Linear(
                            h_times * hidden_channels * heads, hidden_channels * heads
                        ),
                        norm_func(hidden_channels * heads),
                        nn.ReLU(),
                        nn.Dropout(ff_dropout),
                        nn.Linear(hidden_channels * heads, hidden_channels),
                        norm_func(hidden_channels),
                        nn.ReLU(),
                        nn.Dropout(ff_dropout),
                    )
                )

        self.fc_out = torch.nn.Linear(hidden_channels, out_channels)

    def reset_parameters(self):
        self.fc_in.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for ff in self.ffs:
            ff.reset_parameters()
        self.fc_out.reset_parameters()

    def forward(self, seq, x, pos_enc, batch_idx):
        x = self.fc_in(x)
        seq = self.fc_in_seq(seq)

        for i, conv in enumerate(self.convs):
            x = conv(seq, x, pos_enc, batch_idx)
            x = self.ffs[i](x)
        x = self.fc_out(x)
        return x

    def global_forward(self, x, pos_enc, batch_idx):
        x = self.fc_in(x)
        for i, conv in enumerate(self.convs):
            x = conv.global_forward(x, pos_enc, batch_idx)
            x = self.ffs[i](x)
        x = self.fc_out(x)
        return x
