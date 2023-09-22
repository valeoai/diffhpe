import torch.nn as nn


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        n_layers=1,
        act_layer=nn.ReLU,
        drop=0.0,
    ):
        super().__init__()
        self.act = act_layer()
        self.drop = nn.Dropout(drop)

        self.fc_in = nn.Linear(in_features, hidden_features)
        self.fcs = (
            nn.Sequential(
                *[
                    nn.Sequential(
                        *[
                            nn.Linear(hidden_features, hidden_features),
                            self.act,
                            self.drop,
                        ]
                    )
                    for _ in range(n_layers - 2)
                ]
            )
            if n_layers > 2
            else nn.Identity()
        )

        self.fc_out = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc_in(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fcs(x)
        x = self.fc_out(x)
        x = self.drop(x)
        return x


class MlpCoords(Mlp):
    def forward(self, x, diffusion_step=None):
        # x has shape B, D, J, L
        x = x.permute(0, 3, 2, 1)  # B, L, J, D
        x = super().forward(x)  # B, L, J, D'
        return x.permute(0, 3, 2, 1)  # B, D', J, L


class MlpJoints(Mlp):
    def forward(self, x, diffusion_step=None):
        B, D, J, L = x.shape
        x = x.permute(0, 3, 2, 1)  # B, L, J, D
        x = x.reshape(B, L, J * D)  # B, L, J x D
        x = super().forward(x)  # B, L, J x D'
        return x.reshape(B, L, J, -1).permute(0, 3, 2, 1)  # B, D', J, L


class MlpSeq(Mlp):
    def forward(self, x, diffusion_step=None):
        B, D, J, L = x.shape
        x = x.reshape(B, D * L * J)  # B, L x J x L
        x = super().forward(x)  # B, D' x J x L
        return x.reshape(B, -1, J, L)  # B, D', J, L
