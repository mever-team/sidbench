from models.CNNDetect import CNNDetect
from models.FreqDetect import FreqDetect
from models.RPTC import Net as RPTCNet
from models.PSM import PSM
from models.UnivFD import UnivFD
from models.GramNet import GramNet
from models.Rine import RineModel
from models.LGrad import LGrad
from models.DIMD import DIMD
from models.NPR import NPR
from models.Dire import Dire
from models.DeFake import DeFake

import re
import torch

from networks.blip.blip import blip_decoder
from preprocessing.lgrad.models import build_model
from utils.util import setup_device


VALID_MODELS = ['CNNDetect', 'FreqDetect', 'Fusing', 'GramNet', 'LGrad', 'UnivFD', 'RPTC', 'Rine', 'DIMD', 'NPR', 'Dire', 'DeFake']


def get_model(opt):
    
    model_name = opt.modelName
    assert model_name in VALID_MODELS
    
    device = setup_device(opt.gpus)
    print('Model: ', opt.modelName)
    print("Device: ", device)

    if model_name == 'CNNDetect':
        model = CNNDetect()
    elif model_name == 'FreqDetect':
        opt.dctMean = torch.load(opt.dctMean).permute(1,2,0).numpy()
        opt.dctVar = torch.load(opt.dctVar).permute(1,2,0).numpy()
        model = FreqDetect()
    elif model_name == 'Fusing':
        model = PSM()
    elif model_name == 'GramNet':
        model = GramNet()
    elif model_name == 'LGrad':
        opt.numThreads = int(0)
        opt.cropSize = 256
        gen_model = build_model(gan_type='stylegan', 
                                module='discriminator', 
                                resolution=256, 
                                label_size=0, 
                                image_channels=3)
        gen_model.load_state_dict(torch.load(opt.LGradGenerativeModelPath), strict=True)
        gen_model = gen_model.to(device)
        opt.LGradGenerativeModel = gen_model
        model = LGrad()
    elif model_name == 'UnivFD':
        model = UnivFD()
    elif model_name == 'RPTC':
        model = RPTCNet()
    elif model_name == 'DIMD':
        model = DIMD()
    elif model_name == 'Dire':
        from preprocessing.dire.guided_diffusion.script_util import (
            model_and_diffusion_defaults, 
            create_model_and_diffusion
        )
        from preprocessing.dire.guided_diffusion import dist_util

        opt.numThreads = int(0)
        dire_args = model_and_diffusion_defaults()
        diffusion_model, diffusion = create_model_and_diffusion(**dire_args)
        dire_defaults = dict(
            noise_type="resize",
            clip_denoised=True,
            num_samples=3998,
            batch_size=16,
            use_ddim=False,
            model_path=opt.DireGenerativeModelPath,
            real_step=0, 
            continue_reverse=False,
            has_subfolder=False,
            has_subclasses=False,
        )
        dire_args.update(dire_defaults)

        diffusion_model.load_state_dict(dist_util.load_state_dict(dire_args['model_path'], map_location="cpu"))
        diffusion_model = diffusion_model.to(device)
        if dire_args['use_fp16']:
            diffusion_model.convert_to_fp16()
        diffusion_model.eval()

        opt.diffusionModel, opt.diffusion, opt.direArgs = diffusion_model, diffusion, dire_args
        model = Dire()
    elif model_name == 'NPR':
        model = NPR()
    elif model_name == 'DeFake':
        opt.defakeBlip = blip_decoder(pretrained=opt.defakeBlipPath, image_size=224, vit='base')
        opt.defakeClipEncoder = torch.load(opt.defakeClipEncoderPath, map_location='cpu')
        model = DeFake(blip=opt.defakeBlip, encoder=opt.defakeClipEncoder)
    elif model_name == 'Rine':
        pattern = r'model_([^_]*)_trainable'
        match = re.search(pattern, opt.ckpt)
        if match:
            ncls = match.group(1)
        else:
            print("No ncls found")

        if ncls == '1class':
            nproj = 4
            proj_dim = 1024
        elif ncls == '2class':
            nproj = 4
            proj_dim = 128
        elif ncls == '4class':
            nproj = 2
            proj_dim = 1024
        elif ncls == "ldm":
            nproj = 4
            proj_dim = 1024
        model = RineModel(backbone=("ViT-L/14", 1024), nproj=nproj, proj_dim=proj_dim)

    model.load_weights(ckpt=opt.ckpt)
    model.eval()
    model = model.to(device)

    return model

