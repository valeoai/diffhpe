import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from copy import deepcopy
from functools import reduce

from diffhpe.data.utils.graph_utils import adj_mx_from_skeleton
from .pre_agg.any_gcn import _GraphConv, _GraphNonLocal
from .diffusion_emb import DiffusionEmbedding, DummyEmbedding


def Conv1d_with_init(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer


class LiftCTGCN(nn.Module):
    def __init__(
        self,
        config,
        seq_len,
        skeleton,
        n_input_coords=5,
        n_output_coords=3,
        debug_disable_diffusion=False,
    ):
        super().__init__()
        self.channels = config.channels
        self.n_inputs_coords = n_input_coords
        self.seq_len = seq_len
        self.n_output_coords = n_output_coords
        self.skeleton = deepcopy(skeleton)
        self.use_nonlocal = config.use_nonlocal

        self.adj = adj_mx_from_skeleton(self.skeleton, "default", "default")

        if debug_disable_diffusion:
            self.diffusion_embedding = DummyEmbedding(
                embedding_dim=config.diffusion_embedding_dim,
            )
        else:
            self.diffusion_embedding = DiffusionEmbedding(
                num_steps=config.num_steps,
                embedding_dim=config.diffusion_embedding_dim,
            )

        self.input_projection = Conv1d_with_init(
            n_input_coords, self.channels, 1
        )
        self.output_projection1 = Conv1d_with_init(
            self.channels, self.channels, 1
        )
        self.output_projection2 = Conv1d_with_init(
            self.channels, n_output_coords, 1
        )
        nn.init.zeros_(self.output_projection2.weight)

        if self.use_nonlocal:
            nodes_group = skeleton.joints_group()
            assert (
                nodes_group is not None
            ), "Non-Local layers require node groups"
            group_size = len(nodes_group[0])
            assert group_size > 1

            grouped_order = list(reduce(lambda x, y: x + y, nodes_group))
            restored_order = [0] * len(grouped_order)
            for i in range(len(restored_order)):
                for j in range(len(grouped_order)):
                    if grouped_order[j] == i:
                        restored_order[i] = j
                        break
        else:
            grouped_order = None
            restored_order = None
            group_size = 0

        self.residual_layers = nn.ModuleList(
            [
                GCNResidualBlock(
                    channels=self.channels,
                    seq_len=self.seq_len,
                    diffusion_embedding_dim=config.diffusion_embedding_dim,
                    adj=self.adj,
                    p_dropout=config.p_dropout,
                    use_nonlocal=self.use_nonlocal,
                    grouped_order=grouped_order,
                    restored_order=restored_order,
                    group_size=group_size,
                )
                for _ in range(config.layers)
            ]
        )

    def forward(self, x, diffusion_step):
        B, D, J, L = x.shape
        assert (
            D == self.n_inputs_coords
        ), f"Expected input dimension {self.n_inputs_coords}, got {D}."

        x = x.reshape(B, D, J * L)  # (B,C,J*L)
        x = self.input_projection(x)  # (B,C,J*L)
        x = F.relu(x)
        x = x.reshape(B, self.channels, J, L)  # (B,C,J,L)

        diffusion_emb = self.diffusion_embedding(diffusion_step)

        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x, diffusion_emb)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(
            len(self.residual_layers)
        )
        x = x.reshape(B, self.channels, -1)  # (B,channel,J*L)
        x = self.output_projection1(x)  # (B,channel,J*L)
        x = F.relu(x)
        x = self.output_projection2(x)  # (B,3,J*L)
        x = x.reshape(B, self.n_output_coords, J, L)  # (B,3, J,L)
        return x


class GCNResidualBlock(nn.Module):
    def __init__(
        self,
        channels,
        seq_len,
        diffusion_embedding_dim,
        adj,
        p_dropout,
        use_nonlocal,
        grouped_order,
        restored_order,
        group_size,
    ):
        super().__init__()
        self.use_nonlocal = use_nonlocal
        self.diffusion_projection = nn.Linear(
            diffusion_embedding_dim, channels
        )
        self.mid_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)

        self.time_layer = _GraphConv(
            adj=adj,
            input_dim=seq_len,
            output_dim=seq_len,
            p_dropout=p_dropout,
            gcn_type="dc_preagg",
        )
        self.feature_layer = _GraphConv(
            adj=adj,
            input_dim=channels,
            output_dim=channels,
            p_dropout=p_dropout,
            gcn_type="dc_preagg",
        )

        if self.use_nonlocal:
            self.non_local = _GraphNonLocal(
                hid_dim=channels,
                grouped_order=grouped_order,
                restored_order=restored_order,
                group_size=group_size,
            )

    def forward_time(self, y, base_shape):
        B, channel, J, L = base_shape
        if L == 1:
            return y
        # Treat channels independently and interprete time as "fake" channels
        y = y.reshape(B, channel, J, L).reshape(B * channel, J, L)
        y = self.time_layer(y)
        y = y.reshape(B, channel, J, L).reshape(B, channel, J * L)
        return y

    def forward_feature(self, y, base_shape):
        B, channel, J, L = base_shape
        if J == 1:
            return y
        y = (
            y.reshape(B, channel, J, L)
            .permute(0, 3, 2, 1)
            .reshape(B * L, J, channel)
        )
        y = self.feature_layer(y)
        y = (
            y.reshape(B, L, J, channel)
            .permute(0, 3, 2, 1)
            .reshape(B, channel, J * L)
        )
        return y

    def forward(self, x, diffusion_emb):
        B, channel, J, L = x.shape
        base_shape = x.shape
        x = x.reshape(B, channel, J * L)

        diffusion_emb = self.diffusion_projection(diffusion_emb).unsqueeze(
            -1
        )  # (B,channel,1)
        y = x + diffusion_emb

        y = self.forward_time(y, base_shape)
        y = self.forward_feature(y, base_shape)  # (B,channel,J*L)
        if self.use_nonlocal:
            y = (
                y.reshape(B, channel, J, L)
                .permute(0, 3, 2, 1)
                .reshape(B * L, J, channel)
            )
            y = self.non_local(y)
            y = (
                y.reshape(B, L, J, channel)
                .permute(0, 3, 2, 1)
                .reshape(B, channel, J * L)
            )
        y = self.mid_projection(y)  # (B,2*channel,J*L)

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)  # (B,channel,J*L)
        y = self.output_projection(y)

        residual, skip = torch.chunk(y, 2, dim=1)
        x = x.reshape(base_shape)
        residual = residual.reshape(base_shape)
        skip = skip.reshape(base_shape)
        return (x + residual) / math.sqrt(2.0), skip
