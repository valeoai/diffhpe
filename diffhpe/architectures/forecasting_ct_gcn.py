import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from copy import deepcopy
import numpy as np

from diffhpe.data.lifting.h36m_lifting import h36m_skeleton
from diffhpe.data.utils.graph_utils import adj_mx_from_skeleton
from .pre_agg.any_gcn import _GraphConv
from .diffusion_emb import DiffusionEmbedding


def Conv1d_with_init(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer


class ForecastCTGCN(nn.Module):
    def __init__(
        self, config, seq_len, features_to_keep, n_inputs=2, input_channels=3
    ):
        super().__init__()
        self.channels = config.channels
        self.n_inputs = n_inputs
        self.seq_len = seq_len
        self.input_channels = input_channels
        self.skeleton = deepcopy(h36m_skeleton)

        joint_to_keep = np.array(features_to_keep) // 3
        joints_to_remove = [
            i
            for i in range(len(self.skeleton.parents()))
            if i not in joint_to_keep
        ]
        self.skeleton.remove_joints(joints_to_remove)

        self.adj = adj_mx_from_skeleton(self.skeleton, "default", "default")

        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=config.num_steps,
            embedding_dim=config.diffusion_embedding_dim,
        )

        self.input_projection = Conv1d_with_init(
            n_inputs * input_channels, self.channels, 1
        )
        self.output_projection1 = Conv1d_with_init(
            self.channels, self.channels, 1
        )
        self.output_projection2 = Conv1d_with_init(
            self.channels, input_channels, 1
        )
        nn.init.zeros_(self.output_projection2.weight)

        self.residual_layers = nn.ModuleList(
            [
                GCNResidualBlock(
                    channels=self.channels,
                    seq_len=self.seq_len,
                    input_channels=self.input_channels,
                    diffusion_embedding_dim=config.diffusion_embedding_dim,
                    adj=self.adj,
                    p_dropout=config.p_dropout,
                    mask_conditioning=config.mask_conditioning,
                )
                for _ in range(config.layers)
            ]
        )

    def forward(self, x, cond_info, diffusion_step):
        B, n_inputs, K, L = x.shape
        assert n_inputs == self.n_inputs

        x = x.reshape(B, self.n_inputs * self.input_channels, -1)  # (B,6,J*L)
        x = self.input_projection(x)  # (B,C,J*L)
        x = F.relu(x)
        x = x.reshape(B, self.channels, -1, L)  # (B,C,J,L)

        diffusion_emb = self.diffusion_embedding(diffusion_step)

        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x, cond_info, diffusion_emb)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(
            len(self.residual_layers)
        )
        x = x.reshape(B, self.channels, -1)  # (B,channel,J*L)
        x = self.output_projection1(x)  # (B,channel,J*L)
        x = F.relu(x)
        x = self.output_projection2(x)  # (B,3,J*L)
        x = x.reshape(B, -1, L)  # (B,K,L)
        return x


class GCNResidualBlock(nn.Module):
    def __init__(
        self,
        channels,
        seq_len,
        input_channels,
        diffusion_embedding_dim,
        adj,
        p_dropout,
        mask_conditioning=True,
    ):
        super().__init__()
        self.mask_conditioning = mask_conditioning
        self.diffusion_projection = nn.Linear(
            diffusion_embedding_dim, channels
        )
        self.cond_projection = Conv1d_with_init(
            input_channels, 2 * channels, 1
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

    def forward(self, x, cond_info, diffusion_emb):
        B, channel, J, L = x.shape
        base_shape = x.shape
        x = x.reshape(B, channel, J * L)

        diffusion_emb = self.diffusion_projection(diffusion_emb).unsqueeze(
            -1
        )  # (B,channel,1)
        y = x + diffusion_emb

        y = self.forward_time(y, base_shape)
        y = self.forward_feature(y, base_shape)  # (B,channel,J*L)
        y = self.mid_projection(y)  # (B,2*channel,J*L)

        # TODO: We could probably just pass the cond_mask directly here, since
        # we are dropping 99% of cond_info
        if self.mask_conditioning:
            cond_mask = cond_info[:, -1, ...]  # Get just the mask
            cond_mask = cond_mask.reshape(B, -1, J * L)  # (B, 3, J*L)
            cond_mask = self.cond_projection(cond_mask)  # (B,2*channel,J*L)
            y = y + cond_mask

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)  # (B,channel,J*L)
        y = self.output_projection(y)

        residual, skip = torch.chunk(y, 2, dim=1)
        x = x.reshape(base_shape)
        residual = residual.reshape(base_shape)
        skip = skip.reshape(base_shape)
        return (x + residual) / math.sqrt(2.0), skip
