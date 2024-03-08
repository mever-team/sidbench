from models.CNNDetect import CNNDetect
from models.FreDetect import FreDetect
from models.RPTC import Net as RPTCNet
from models.PSM import PSM
from models.UnivFD import UnivFD
from models.GramNet import GramNet
from models.Rine import RineModel
from models.LGrad import LGrad
from models.DIMD import DIMD
from models.NPR import NPR


VALID_MODELS = ['CNNDetect', 'FreDetect', 'Fusing', 'GramNet', 'LGrad', 'UnivFD', 'RPTC', 'Rine', 'DIMD', 'NPR']


def get_model(model_name, ckpt, ncls=None):
    
    assert model_name in VALID_MODELS
    
    if model_name == 'CNNDetect':
        model = CNNDetect()
    elif model_name == 'FreDetect':
        model = FreDetect()
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
        assert ncls is not None
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



MODELS = [
    {
        "model_name": "UnivFD",
        "trained_on": "progan",
        "ckpt": "./weights/univfd/fc_weights.pth"
    },
    {
        "model_name": "CNNDetect",
        "trained_on": "progan",
        "ckpt": "./weights/cnndetect/blur_jpg_prob0.5.pth"
    },
    {
        "model_name": "DIMD",
        "trained_on": "latent_diffusion",
        "ckpt": "./weights/dimd/corvi22_latent_model.pth"
    },
    {
        "model_name": "DIMD",
        "trained_on": "progan",
        "ckpt": "./weights/dimd/corvi22_progan_model.pth"
    },
    {
        "model_name": "DIMD",
        "trained_on": "stylegan2",
        "ckpt": "./weights/dimd/gandetection_resnet50nodown_stylegan2.pth"
    },
    {
        "model_name": "LGrad",
        "trained_on": "progan",
        "ckpt": "./weights/lgrad/LGrad.pth"
    },
    {
        "model_name": "FreDetect",
        "trained_on": "progan",
        "ckpt": "./weights/fredetect/DCTAnalysis.pth"
    },
    {
        "model_name": "Rine",
        "trained_on": "progan",
        "ckpt": "./weights/rine/model_1class_trainable.pth",
        "ncls": "1class"
    },
    {
        "model_name": "Rine",
        "trained_on": "latent_diffusion",
        "ckpt": "./weights/rine/model_ldm_trainable.pth",
        "ncls": "ldm"
    },
    {
        "model_name": "NPR",
        "trained_on": "progan",
        "ckpt": "./weights/npr/NPR.pth"
    },
    {
        "model_name": "RPTC",
        "trained_on": "progan",
        "ckpt": "./weights/rptc/RPTC.pth"
    },
    {
        "model_name": "Fusing",
        "trained_on": "progan",
        "ckpt": "./weights/fusing/PSM.pth"
    },
    {
        "model_name": "GramNet",
        "trained_on": "progan",
        "ckpt": "./weights/gramnet/Gram.pth"
    }
]



