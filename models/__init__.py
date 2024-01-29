from .clip_models import CLIPModel
from .imagenet_models import ImagenetModel
from .svm_models import SvmModel
from .knn_models import KnnModel


VALID_MODELS = ['linear', 'svm', 'knn']

BACKBONE_VALID_NAMES = [
    'Imagenet:resnet18',
    'Imagenet:resnet34',
    'Imagenet:resnet50',
    'Imagenet:resnet50nodown',
    'Imagenet:resnet101',
    'Imagenet:resnet152',
    'Imagenet:vgg11',
    'Imagenet:vgg19',
    'Imagenet:swin-b',
    'Imagenet:swin-s',
    'Imagenet:swin-t',
    'Imagenet:vit_b_16',
    'Imagenet:vit_b_32',
    'Imagenet:vit_l_16',
    'Imagenet:vit_l_32',
    'CLIP:RN50', 
    'CLIP:RN101', 
    'CLIP:RN50x4', 
    'CLIP:RN50x16', 
    'CLIP:RN50x64', 
    'CLIP:ViT-B/32', 
    'CLIP:ViT-B/16', 
    'CLIP:ViT-L/14', 
    'CLIP:ViT-L/14@336px',
    'CLIP:ViT-H-14'
]


def get_model(classifier, backbone, ckpt):
    
    assert classifier in VALID_MODELS
    assert backbone in BACKBONE_VALID_NAMES
    
    if classifier == 'linear':
        if backbone.startswith("Imagenet:"):
            model = ImagenetModel(backbone[9:])
            model.load_weights(ckpt)
            model.eval()
            return model
        elif backbone.startswith("CLIP:"):
            model =  CLIPModel(backbone[5:])  
            model.load_weights(ckpt)
            model.eval()
            return model
        else:
            assert False 
    elif classifier == 'knn':
        if backbone.startswith("CLIP:"):
            return KnnModel(clip_model=backbone[5:], classifier_model_file=ckpt)
        else:
            assert False
    elif classifier == 'svm':
        if backbone.startswith("CLIP:"):
            return SvmModel(clip_model=backbone[5:], classifier_model_file=ckpt)
        else:
            assert False  





MODELS = [
    {
        'name': 'ojha2022',
        'classifier': 'linear',
        'backbone': 'CLIP:ViT-L/14',
        'ckpt': './pretrained_weights/ojha2022/fc_weights.pth',
    },
    {
        'name': 'wang2020',
        'classifier': 'linear',
        'backbone': 'Imagenet:resnet50',
        'ckpt': './pretrained_weights/wang2020/blur_jpg_prob0.5.pth',
    },
    {
        'name': 'corvi2022_latent',
        'classifier': 'linear',
        'backbone': 'Imagenet:resnet50nodown',
        'ckpt': './pretrained_weights/corvi2022/latent_model_epoch_best.pth',
    },
    {
        'name': 'corvi2022_progan',
        'classifier': 'linear',
        'backbone': 'Imagenet:resnet50nodown',
        'ckpt': './pretrained_weights/corvi2022/progan_model_epoch_best.pth',
    },
    {
        'name': 'grag2021_stylegan2',
        'classifier': 'linear',
        'backbone': 'Imagenet:resnet50nodown',
        'ckpt': './pretrained_weights/grag2021/gandetection_resnet50nodown_stylegan2.pth',
    }
]

