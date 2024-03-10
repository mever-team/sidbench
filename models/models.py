
MODELS = [
    {
        "modelName": "UnivFD",
        "trainedOn": "progan",
        "ckpt": "./weights/univfd/fc_weights.pth"
    },
    {
        "modelName": "CNNDetect",
        "trainedOn": "progan",
        "ckpt": "./weights/cnndetect/blur_jpg_prob0.5.pth"
    },
    {
        "modelName": "DIMD",
        "trainedOn": "latent_diffusion",
        "ckpt": "./weights/dimd/corvi22_latent_model.pth"
    },
    {
        "modelName": "DIMD",
        "trainedOn": "progan",
        "ckpt": "./weights/dimd/corvi22_progan_model.pth"
    },
    {
        "modelName": "DIMD",
        "trainedOn": "stylegan2",
        "ckpt": "./weights/dimd/gandetection_resnet50nodown_stylegan2.pth"
    },
    {
        "modelName": "LGrad",
        "trainedOn": "progan",
        "ckpt": "./weights/lgrad/LGrad.pth"
    },
    {
        "modelName": "FreqDetect",
        "trainedOn": "progan",
        "ckpt": "./weights/freqdetect/DCTAnalysis.pth"
    },
    {
        "modelName": "Rine",
        "trainedOn": "progan",
        "ckpt": "./weights/rine/model_1class_trainable.pth",
        "ncls": "1class"
    },
    {
        "modelName": "Rine",
        "trainedOn": "latent_diffusion",
        "ckpt": "./weights/rine/model_ldm_trainable.pth",
        "ncls": "ldm"
    },
    {
        "modelName": "NPR",
        "trainedOn": "progan",
        "ckpt": "./weights/npr/NPR.pth"
    },
    {
        "modelName": "RPTC",
        "trainedOn": "progan",
        "ckpt": "./weights/rptc/RPTC.pth"
    },
    {
        "modelName": "Fusing",
        "trainedOn": "progan",
        "ckpt": "./weights/fusing/PSM.pth"
    },
    {
        "modelName": "GramNet",
        "trainedOn": "progan",
        "ckpt": "./weights/gramnet/Gram.pth"
    },
        {
        "modelName": "Dire",
        "trainedOn": "progan",
        "ckpt": "./weights/gramnet/Gram.pth"
    }
]

