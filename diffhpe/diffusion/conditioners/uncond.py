from torch import nn


class NoConditioning(nn.Module):
    """Dummy conditioner class which simply returns the noisy data, dropping
    the conditioning data. This allows to train an unconditioned diffusion
    model.
    """

    out_dim = 3

    def __init__(self, mix_mode="drop") -> None:
        super().__init__()
        self.mix_mode = mix_mode

    def forward(self, noisy_data, data_2d):
        return noisy_data
