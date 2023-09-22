import torch
from torch import nn

from diffhpe.architectures import MixSTE

from .raw_2d import mix_data_with_condition


class PTMixSTECond(MixSTE):
    def __init__(
        self,
        checkpoint_path,
        encode_3d=False,
        keep_pt_head=False,
        mix_mode="concat",
        num_frame=243,
        num_joints=17,
        embed_dim=512,
        depth=8,
        num_heads=8,
        mlp_ratio=2,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0,
        attn_drop_rate=0,
        norm_layer=None,
    ):
        super().__init__(
            num_frame=num_frame,
            num_joints=num_joints,
            in_chans=2,  # <--
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=0.0,  # <--
            norm_layer=norm_layer,
        )

        # Save how to mix the noisy 3D data with the condition
        self.mix_mode = mix_mode
        self.encode_3d = encode_3d
        self.keep_pt_head = keep_pt_head

        # Load pre-trained weights
        self.checkpoint_path = checkpoint_path
        checkpoint = torch.load(checkpoint_path)
        self.load_state_dict(checkpoint["model_pos"])

        self.out_dim = 3
        if not self.keep_pt_head:
            # remove 3D projection head (keep only normalization)
            self.head = self.head[0]
            self.out_dim = embed_dim

        if mix_mode == "sum":
            assert encode_3d != self.keep_pt_head, (
                "When summing the condition the 3D noisy data, they need to "
                "have the same dimension. If you encode the 3D data, the "
                "PTMixSTE head should be removed and vice-versa."
            )
        else:
            if encode_3d:
                self.out_dim += embed_dim
            else:
                self.out_dim += 3

        # 3D encoder
        if encode_3d:
            self.data_enc = nn.Linear(in_features=3, out_features=embed_dim)
        else:
            self.data_enc = nn.Identity()

        # Freeze everything but data_enc
        self.Spatial_patch_to_embedding.requires_grad_(False)
        self.Spatial_pos_embed.requires_grad_(False)
        self.Temporal_pos_embed.requires_grad_(False)
        self.STEblocks.requires_grad_(False)
        self.TTEblocks.requires_grad_(False)
        self.Spatial_norm.requires_grad_(False)
        self.Temporal_norm.requires_grad_(False)
        self.head.requires_grad_(False)

    def forward(
        self,
        noisy_data,
        data_2d,
        p_uncond=0.0,
    ):
        encoded_2d_data = super().forward(data_2d)  # B, C, J, L
        encoded_3d_data = self.data_enc(
            noisy_data.permute(0, 3, 2, 1)
        ).permute(0, 3, 2, 1)
        total_input = mix_data_with_condition(
            encoded_3d_data,
            encoded_2d_data,
            mix_mode=self.mix_mode,
            p_uncond=p_uncond,
        )
        return total_input
