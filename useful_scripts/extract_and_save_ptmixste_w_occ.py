# %% - IMPORTS
import pickle
from pathlib import Path
from collections import OrderedDict

import torch
from omegaconf import OmegaConf

from diffhpe.diffusion.diff_lifting import LiftingDiffusionModel
from diffhpe.supervised.lifting import LiftingSupervisedDebugModel

# %% - LOAD DATASET

data_dir = Path("/path/to/data/")
preproc_dataset_path = data_dir / "preproc_data_3d_h36m_17.pkl"

print("==> Loading preprocessed dataset...")
with open(preproc_dataset_path, "rb") as f:
    dataset = pickle.load(f)


# %% - PREP CONFIG
# os.chdir("..")
config_diff = OmegaConf.load("../conf/diffusion/sup_mixste_seq27.yaml")
skeleton = dataset.skeleton()
seq_len = 27

# %% - INSTANTIATE A DUMMY SUPERVISED LIFTER
config = OmegaConf.load("../conf/config.yaml")
config.diffusion.update(config_diff)

diff_model = LiftingSupervisedDebugModel(
    config,
    "cuda",
    skeleton=dataset.skeleton(),
    seq_len=seq_len,
)

# %% LOADCHECKPOINT OF TRAINED DIFF MODEL
diff_model_ckpt_path = "/path/to/LiftingSupervisedDebugModel/checkpoint"
state_dict = torch.load(f"{diff_model_ckpt_path}/model.pth")

# %% TRY TO ADAPT STATE DICT FOR LOADING AS CONDITIONING
adapted_state_dict = OrderedDict(
    (k.replace("model.", ""), v) for k, v in state_dict.items()
)

# %% - INSTANTIATE A DUMMY DIFF LIFTER
config = OmegaConf.load("conf/config.yaml")

checkpoint = torch.load(config_diff.cond_ckpt)
checkpoint["model_pos"] = adapted_state_dict
new_checkpoint_path = "/path/to/pre-trained/models/prt_mixste_h36m_L27_C64.pt"
torch.save(checkpoint, new_checkpoint_path)

# %% TEST WE CAN LOAD IT
config_diff.cond_ckpt = new_checkpoint_path
config.diffusion.update(config_diff)

diff_model = LiftingDiffusionModel(
    config,
    "cuda",
    skeleton=dataset.skeleton(),
    seq_len=seq_len,
)


# %%
