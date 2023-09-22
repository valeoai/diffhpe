import torch
from torch import nn
import torch.nn.functional as F


class DiffusionEmbedding(nn.Module):
    def __init__(
        self, num_steps, embedding_dim=128, projection_dim=None, project=True
    ):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.register_buffer(
            "embedding",
            self._build_embedding(num_steps, embedding_dim / 2),
            persistent=False,
        )
        self.project = project
        if project:
            self.projection1 = nn.Linear(embedding_dim, projection_dim)
            self.projection2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]
        if self.project:
            x = self.projection1(x)
            x = F.silu(x)
            x = self.projection2(x)
            x = F.silu(x)
        return x

    def _build_embedding(self, num_steps, dim=64):
        steps = torch.arange(num_steps).unsqueeze(1)  # (T,1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(
            0
        )  # (1,dim)
        table = steps * frequencies  # (T,dim)
        table = torch.cat(
            [torch.sin(table), torch.cos(table)], dim=1
        )  # (T,dim*2)
        return table


class DiffPoseDiffusionEmbedding(nn.Module):
    def __init__(
        self, num_steps, embedding_dim=128, projection_dim=None, project=True
    ):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.register_buffer(
            "embedding",
            self._build_embedding(num_steps, embedding_dim / 2),
            persistent=False,
        )
        self.project = project
        if project:
            self.projection1 = nn.Linear(embedding_dim, projection_dim)
            self.projection2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]
        if self.project:
            x = self.projection1(x)
            x = F.silu(x)
            x = self.projection2(x)
            x = F.silu(x)
        return x

    def _build_embedding(self, num_steps, dim=64):
        steps = torch.arange(num_steps).unsqueeze(1)  # (T,1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(
            0
        )  # (1,dim)
        table = steps / frequencies  # <---
        table = torch.cat(
            [torch.sin(table), torch.cos(table)], dim=1
        )  # (T,dim*2)
        return table


class DummyEmbedding(nn.Module):
    def __init__(
        self,
        num_steps=None,
        embedding_dim=128,
        projection_dim=None,
        dummy_value=0.0,
    ):
        super().__init__()
        self.projection_dim = (
            embedding_dim if projection_dim is None else projection_dim
        )
        self.dummy_value = dummy_value

    def forward(self, x):
        # Get the batch size from the input tensor
        batch_size = x.shape[0]
        # Create an output tensor with the correct dimensions, including the
        # batch size
        output_shape = (batch_size, self.projection_dim)
        output = torch.full(
            output_shape,
            fill_value=self.dummy_value,
            dtype=x.dtype,
            device=x.device,
        )
        return output
