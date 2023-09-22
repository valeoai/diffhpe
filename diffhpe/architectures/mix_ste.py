from functools import partial

import torch
import torch.nn as nn

from einops import rearrange  # equivalent to reshape, but reader-friendly
from timm.models.layers import DropPath

from .diffusion_emb import DiffusionEmbedding


class MixSTE(nn.Module):
    def __init__(
        self,
        num_frame=243,
        num_joints=17,
        in_chans=2,
        out_dim=3,  # output dimension is num_joints * 3
        embed_dim=512,
        depth=8,
        num_heads=8,
        mlp_ratio=2.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        norm_layer=None,
    ):
        """##########hybrid_backbone=None, representation_size=None,
        Args:
            num_frame (int, tuple): input frame number
            num_joints (int, tuple): joints number
            in_chans (int): number of input channels, 2D joints have 2
            embed_dim (int): embedding dimension ratio
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): query-key scaler (defaults to 1/dim**O.5)
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.embed_dim = embed_dim

        # spatial patch embedding
        self.Spatial_patch_to_embedding = nn.Linear(in_chans, embed_dim)
        self.Spatial_pos_embed = nn.Parameter(
            torch.zeros(1, num_joints, embed_dim)
        )

        self.Temporal_pos_embed = nn.Parameter(
            torch.zeros(1, num_frame, embed_dim)
        )

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.block_depth = depth

        # Double Attention Blocks
        self.STEblocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )

        self.TTEblocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    comb=False,
                    changedim=False,
                    currentdim=i + 1,
                    depth=depth,
                )
                for i in range(depth)
            ]
        )

        self.Spatial_norm = norm_layer(embed_dim)
        self.Temporal_norm = norm_layer(embed_dim)

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, out_dim),
        )

    def STE_forward(self, x, diffusion_step):
        B, C, J, L = x.shape
        # B batch size, L seq length, J joints, C channels
        x = rearrange(x, "B C J L  -> (B L) J C")

        # now x is [batch_size * seq len, joint_num, 2]
        x = self.Spatial_patch_to_embedding(x)
        # now x is [batch_size * seq len, joint_num, channels]

        x += self.Spatial_pos_embed
        x = self.pos_drop(x)

        blk = self.STEblocks[0]
        x = blk(x)

        x = self.Spatial_norm(x)
        x = rearrange(x, "(B L) J C -> (B J) L C", L=L)
        return x

    def TTE_foward(self, x):
        assert len(x.shape) == 3, "TTE input should be 3-dim."
        x += self.Temporal_pos_embed
        x = self.pos_drop(x)
        blk = self.TTEblocks[0]
        x = blk(x)

        x = self.Temporal_norm(x)
        return x

    def ST_foward(self, x):
        assert len(x.shape) == 4, "ST input should be 4-dim"
        B, L, J, C = x.shape
        for i in range(1, self.block_depth):
            x = rearrange(x, "B L J C -> (B L) J C")
            steblock = self.STEblocks[i]
            tteblock = self.TTEblocks[i]

            x = steblock(x)
            x = self.Spatial_norm(x)
            x = rearrange(x, "(B L) J C -> (B J) L C", L=L)

            x = tteblock(x)
            x = self.Temporal_norm(x)
            x = rearrange(x, "(B J) L C -> B L J C", J=J)

        return x

    def forward(self, x, diffusion_step=None):
        B, C, J, L = x.shape
        # now x is (B, C, J, L) = (batch_size, 2, joint num, seq lentgh)

        x = self.STE_forward(x, diffusion_step)
        # now x shape is (B*J, L, C)

        x = self.TTE_foward(x)

        x = rearrange(x, "(B J) L C -> B L J C", J=J)
        x = self.ST_foward(x)

        x = self.head(x)
        # now x shape is (B, L, J, C) -- but code says (B L (J * 3))...

        # XXX Commented as seems useless with current x shape
        #  x = x.view(b, f, n, -1)

        x = rearrange(x, "B L J C -> B C J L")

        return x


class DiffMixSTE(MixSTE):
    def __init__(
        self,
        num_frame=243,
        num_joints=17,
        in_chans=2,
        out_dim=3,
        embed_dim=512,
        depth=8,
        num_heads=8,
        mlp_ratio=2.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        norm_layer=None,
        num_diff_steps=50,
    ) -> None:
        super().__init__(
            num_frame=num_frame,
            num_joints=num_joints,
            in_chans=in_chans,
            out_dim=out_dim,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
        )

        # Diffusion step embedding
        # TODO: This should be outside, in diff_lifting.py
        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=num_diff_steps,
            embedding_dim=embed_dim,
        )

    def STE_forward(self, x, diffusion_step):
        B, C, J, L = x.shape
        # B batch size, L seq length, J joints, C channels
        x = rearrange(x, "B C J L  -> (B L) J C")

        # now x is [batch_size * seq len, joint_num, 2]
        x = self.Spatial_patch_to_embedding(x)
        # now x is [batch_size * seq len, joint_num, channels]

        # Conditioning on diffusion embedding
        diffusion_emb = self.diffusion_embedding(diffusion_step)
        x = rearrange(x, "(B L) J C -> B C J L", L=L)
        x += diffusion_emb[:, :, None, None]
        x = rearrange(x, "B C J L  -> (B L) J C")

        x += self.Spatial_pos_embed
        x = self.pos_drop(x)

        blk = self.STEblocks[0]
        x = blk(x)

        x = self.Spatial_norm(x)
        x = rearrange(x, "(B L) J C -> (B J) L C", L=L)
        return x


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
        changedim=False,
        currentdim=0,
        depth=0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)

        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        comb=False,
        vis=False,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually
        # to be compat with prev weights
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)
        self.comb = comb
        self.vis = vis

    def forward(self, x, vis=False):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        # Now x shape (3, B, heads, N, C//heads)
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )
        if self.comb:
            attn = (q.transpose(-2, -1) @ k) * self.scale
        elif not self.comb:
            attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        if self.comb:
            x = (attn @ v.transpose(-2, -1)).transpose(-2, -1)
            x = rearrange(x, "B H N C -> B N (H C)")
        elif not self.comb:
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        attention=Attention,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        comb=False,
        changedim=False,
        currentdim=0,
        depth=0,
        vis=False,
    ):
        super().__init__()

        self.changedim = changedim
        self.currentdim = currentdim
        self.depth = depth
        if self.changedim:
            assert self.depth > 0

        self.norm1 = norm_layer(dim)
        self.attn = attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            comb=comb,
            vis=vis,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better
        # than dropout here
        self.drop_path = (
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        if self.changedim and self.currentdim < self.depth // 2:
            self.reduction = nn.Conv1d(dim, dim // 2, kernel_size=1)
        elif self.changedim and depth > self.currentdim > self.depth // 2:
            self.improve = nn.Conv1d(dim, dim * 2, kernel_size=1)
        self.vis = vis

    def forward(self, x, vis=False):
        x = x + self.drop_path(self.attn(self.norm1(x), vis=vis))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        if self.changedim and self.currentdim < self.depth // 2:
            x = rearrange(x, "b t c -> b c t")
            x = self.reduction(x)
            x = rearrange(x, "b c t -> b t c")
        elif self.changedim and self.depth > self.currentdim > self.depth // 2:
            x = rearrange(x, "b t c -> b c t")
            x = self.improve(x)
            x = rearrange(x, "b c t -> b t c")
        return x
