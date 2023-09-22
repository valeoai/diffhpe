# %%
import os
from pathlib import Path
import pickle
import torch
from torch import nn
from omegaconf import OmegaConf

from diffhpe.architectures import MixSTE


# %%

data_dir = Path("/path/to/data/")
preproc_dataset_path = data_dir / "preproc_data_3d_h36m_17.pkl"

print("==> Loading preprocessed dataset...")
with open(preproc_dataset_path, "rb") as f:
    dataset = pickle.load(f)


# %%
config_diff = OmegaConf.load("../conf/config.yaml")
skeleton = dataset.skeleton()
seq_len = 81
embed_dim = 512

model = MixSTE(
    num_frame=seq_len,
    num_joints=skeleton.num_joints(),
    embed_dim=embed_dim,
    depth=config_diff.cond_layers,
    num_heads=config_diff.cond_nheads,
)

# %%
para_model = nn.DataParallel(model)
# %%
checkpoint_path = (
    "/path/to/pre-trained/models/prt_mixste_h36m_L81_C512.pt"
)
checkpoint = torch.load(checkpoint_path)
para_model.load_state_dict(checkpoint["model_pos"])
# %%
checkpoint["model_pos"] = para_model.module.state_dict()
# %%
os.rename(checkpoint_path, checkpoint_path + ".bckp")
# %%
torch.save(checkpoint, checkpoint_path)
# %%
