import os
import torch
from collections import OrderedDict

import argparse
from preprocessing_model.guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    add_dict_to_argparser
)

import numpy as np
import random

def set_random_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def unnormalize(tens, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    # assume tensor of shape NxCxHxW
    return tens * torch.Tensor(std)[None, :, None, None] + torch.Tensor(mean)[None, :, None, None]


def load_checkpoint(model, weights):
    checkpoint = torch.load(weights)
    try:
        model.load_state_dict(checkpoint["state_dict"])
    except:
        state_dict = checkpoint["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        
        
def create_argparser():
    defaults = dict(
        images_dir="./test/stargan",
        recons_dir="./test_dire/recons",
        dire_dir="./test_dire/dire",
        clip_denoised=True,
        num_samples=1,
        batch_size=1,
        use_ddim=True,
        model_path="./weights/prepocessing/lsun_bedroom.pt",
        real_step=0,  #test
        continue_reverse=False,
        has_subfolder=False,
        has_subclasses=False,
        timestep_respacing="ddim20"
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser
