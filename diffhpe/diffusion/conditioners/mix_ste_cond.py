from torch import nn

from diffhpe.architectures import MixSTE

from .raw_2d import mix_data_with_condition


class MixSTECond(MixSTE):
    def __init__(
        self,
        encode_3d=False,
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
        drop_path_rate=0.0,
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
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
        )
        self.mix_mode = mix_mode
        self.encode_3d = encode_3d
        assert not (encode_3d is False and mix_mode == "sum"), (
            "Cannot condition by sum unless 3d data is encoded with same dim"
            "as 2d. Either set encode_3d to True or mix_mode to 'concat'."
        )
        if mix_mode == "sum":
            self.out_dim = embed_dim
        elif encode_3d:
            self.out_dim = 2 * embed_dim
        else:
            self.out_dim = embed_dim + 3

        # remove 3D projection head
        self.head = nn.LayerNorm(self.embed_dim)

        # 3D encoder
        if encode_3d:
            self.data_enc = nn.Linear(in_features=3, out_features=embed_dim)
        else:
            self.data_enc = nn.Identity()

    def forward(
        self,
        noisy_data,
        data_2d,
        p_uncond=0.0,
    ):
        encoded_2d_data = super().forward(data_2d)
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
