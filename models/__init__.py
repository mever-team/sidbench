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

import re

VALID_MODELS = ['CNNDetect', 'FreqDetect', 'Fusing', 'GramNet', 'LGrad', 'UnivFD', 'RPTC', 'Rine', 'DIMD', 'NPR']


def get_model(model_name, ckpt):
    
    assert model_name in VALID_MODELS
    
    if model_name == 'CNNDetect':
        model = CNNDetect()
    elif model_name == 'FreqDetect':
        model = FreqDetect()
    elif model_name == 'Fusing':
        model = PSM()
    elif model_name == 'GramNet':
        model = GramNet()
    elif model_name == 'LGrad':
        model = LGrad()
    elif model_name == 'UnivFD':
        model = UnivFD()
    elif model_name == 'RPTC':
        model = RPTCNet()
    elif model_name == 'DIMD':
        model = DIMD()
    elif model_name == 'NPR':
        model = NPR()
    elif model_name == 'Rine':
        pattern = r'model_([^_]*)_trainable'
        match = re.search(pattern, ckpt)
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

    model.load_weights(ckpt=ckpt)
    model.eval()

    return model

