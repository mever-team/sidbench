import os
import torch
from collections import OrderedDict

import numpy as np
import random


def set_random_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
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
    # assume tensor of shape N x C x H x W 
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


def setup_device(gpus):
    if gpus == '-1':  # CPU
        device = torch.device('cpu')
    else:
        device_ids = list(map(int, gpus.split(',')))
        if len(device_ids) == 1:
            # Single GPU
            device = torch.device(f'cuda:{device_ids[0]}' if torch.cuda.is_available() else 'cpu')
        else:
            # Multiple GPUs (DataParallel)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            if not torch.cuda.is_available():
                print("CUDA is not available. Running on CPU.")
            else:
                torch.cuda.set_device(device_ids[0])  # Set default device for DataParallel
    return device
