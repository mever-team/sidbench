import copy
import cv2

from scipy import fftpack
from io import BytesIO 
from PIL import Image 

import numpy as np 

import torch 
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

from random import choice
from models import VALID_MODELS

from utils.util import setup_device

MEAN = { 
    "imagenet":[0.485, 0.456, 0.406], 
    "clip":[0.48145466, 0.4578275, 0.40821073] 
}

STD = { 
    "imagenet":[0.229, 0.224, 0.225], 
    "clip":[0.26862954, 0.26130258, 0.27577711] 
}


def psm_processing(img, opt):
    height, width = img.height, img.width

    input_img = copy.deepcopy(img)
    input_img = transforms.ToTensor()(input_img)
    input_img = transforms.Normalize(MEAN['imagenet'], STD['imagenet'])(input_img)

    if opt.loadSize is not None:
        img = transforms.Resize(opt.loadSize)(img)

    img = transforms.CenterCrop(opt.cropSize)(img)
    cropped_img = transforms.ToTensor()(img)
    cropped_img = transforms.Normalize(MEAN['imagenet'], STD['imagenet'])(cropped_img)

    scale = torch.tensor([height, width])

    return input_img, cropped_img, scale


def lgrad_processing(img, opt):
    
    device = setup_device(opt.gpus)

    gen_transforms = transforms.Compose([
        transforms.CenterCrop(opt.cropSize),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    img_list = []
    img_list.append(torch.unsqueeze(gen_transforms(img), 0))
    img = torch.cat(img_list, 0)

    img_cuda = img.to(torch.float32)
    img_cuda= img_cuda.to(device)
    img_cuda.requires_grad = True
    
    pre = opt.LGradGenerativeModel(img_cuda)
    opt.LGradGenerativeModel.zero_grad()
    grads = torch.autograd.grad(pre.sum(), img_cuda, create_graph=True, retain_graph=True, allow_unused=False)[0]
    
    for _, grad in enumerate(grads):
        img_grad = normalize(grad.permute(1,2,0).cpu().detach().numpy())
    
    retval, buffer = cv2.imencode(".png", img_grad)
    if retval:
        img = Image.open(BytesIO(buffer)).convert('RGB')

    return transforms.Compose([
            transforms.CenterCrop(opt.cropSize),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN['imagenet'], std=STD['imagenet'])
        ])(img)


def resnet_processing(img, opt):
    transformations = []

    if opt.loadSize is not None:
        transformations.append(transforms.Resize(size=(opt.loadSize, opt.loadSize)))
    
    if opt.isTrain:
        crop_func = transforms.RandomCrop(opt.cropSize)
    else:
        crop_func = transforms.CenterCrop(opt.cropSize)
    
    transformations.append(crop_func)
    transformations.append(transforms.ToTensor())
    transformations.append(transforms.Normalize(mean=MEAN['imagenet'], std=STD['imagenet']))

    transform = transforms.Compose(transformations)
    
    return transform(img)


def clip_processing(img, opt):
    transformations = []

    if opt.loadSize:
        transformations.append(transforms.Resize(size=(opt.loadSize, opt.loadSize)))

    if opt.isTrain:
        crop_func = transforms.RandomCrop(opt.cropSize)
    else:
        crop_func = transforms.CenterCrop(opt.cropSize)

    transformations.append(crop_func)
    transformations.append(transforms.ToTensor())
    transformations.append(transforms.Normalize(mean=MEAN['clip'], std=STD['clip']))

    transform = transforms.Compose(transformations)
    
    return transform(img)


def dct2_wrapper(image, mean, var, log=True, epsilon=1e-12):
    """ apply 2d-DCT to image of shape (H, W, C) uint8 """
    # dct
    image = np.array(image)
    image = fftpack.dct(image, type=2, norm="ortho", axis=0)
    image = fftpack.dct(image, type=2, norm="ortho", axis=1)
    # log scale
    if log:
        image = np.abs(image)
        image = np.log(image + epsilon)

    # normalize
    return (image - mean) / np.sqrt(var)


def freq_processing(img, opt):
    input_img = copy.deepcopy(img)
    input_img = transforms.ToTensor()(input_img)
    input_img = transforms.Normalize(mean=MEAN['imagenet'], std=STD['imagenet'])(input_img)

    if opt.loadSize is not None:
        img = transforms.Resize(opt.loadSize)(img)

    img = transforms.CenterCrop(opt.cropSize)(img)
    
    cropped_img = torch.from_numpy(dct2_wrapper(img, opt.dctMean, opt.dctVar)).permute(2,0,1).to(dtype=torch.float)
    
    return cropped_img


def rptc_processing(img, opt):
    num_block = int(pow(2, opt.patchNum))
    patchsize = int(opt.cropSize / num_block)
    randomcrop = transforms.RandomCrop(patchsize)
    
    minsize = min(img.size)
    if minsize < patchsize:
        img = transforms.Resize((patchsize,patchsize))(img)
    
    img = transforms.ToTensor()(img)

    imgori = img.clone().unsqueeze(0)
    img_template = torch.zeros(3, opt.cropSize, opt.cropSize)
    img_crops = []
    for i in range(num_block * num_block * 3):
        cropped_img = randomcrop(img)
        texture_rich = ED(cropped_img)
        img_crops.append([cropped_img, texture_rich])

    img_crops = sorted(img_crops, key=lambda x:x[1])

    count = 0
    for ii in range(num_block):
        for jj in range(num_block):
            img_template[:,ii*patchsize:(ii+1)*patchsize,jj*patchsize:(jj+1)*patchsize] = img_crops[count][0]
            count += 1
    img_poor = img_template.clone().unsqueeze(0)

    count = -1
    for ii in range(num_block):
        for jj in range(num_block):
            img_template[:,ii*patchsize:(ii+1)*patchsize,jj*patchsize:(jj+1)*patchsize] = img_crops[count][0]
            count -= 1
    img_rich = img_template.clone().unsqueeze(0)
    img = torch.cat((img_poor,img_rich),0)
    
    return img


def ED(img):
    r1, r2 = img[:, 0:-1, :], img[:, 1::, :]
    r3, r4 = img[:, :, 0:-1], img[:, :, 1::]
    r5, r6 = img[:, 0:-1, 0:-1], img[:, 1::, 1::]
    r7, r8 = img[:, 0:-1, 1::], img[:, 1::, 0:-1]
    s1 = torch.sum(torch.abs(r1 - r2)).item()
    s2 = torch.sum(torch.abs(r3 - r4)).item()
    s3 = torch.sum(torch.abs(r5 - r6)).item()
    s4 = torch.sum(torch.abs(r7 - r8)).item() 

    return s1 + s2 + s3 + s4


def reshape_image(imgs: torch.Tensor, image_size: int) -> torch.Tensor:
    if len(imgs.shape) == 3:
        imgs = imgs.unsqueeze(0)

    if imgs.shape[2] != imgs.shape[3]:
        crop_func = transforms.CenterCrop(image_size)
        imgs = crop_func(imgs)
    
    if imgs.shape[2] != image_size:
        imgs = F.resize(imgs, size=(image_size, image_size), interpolation=F.InterpolationMode.BICUBIC)
        # imgs = F.interpolate(imgs, size=(image_size, image_size), mode="bicubic")
    return imgs


def center_crop_arr(pil_image, image_size):
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


def dire_processing(img, opt):

    device = setup_device(opt.gpus)

    img = center_crop_arr(img, opt.cropSize)
    img = img.astype(np.float32) / 127.5 - 1
    img = torch.from_numpy(np.transpose(img, [2, 0, 1]))

    img_list = []
    img_list.append(torch.unsqueeze(img,0))
    img=torch.cat(img_list,0)

    reverse_fn = opt.diffusion.ddim_reverse_sample_loop
    img = reshape_image(img, opt.direArgs['image_size'])
    
    img = img.to(device)
    model_kwargs = {}

    latent = reverse_fn(
        opt.diffusionModel,
        (1, 3, opt.direArgs['image_size'], opt.direArgs['image_size']),
        noise=img,
        clip_denoised=opt.direArgs['clip_denoised'],
        model_kwargs=model_kwargs,
        real_step=opt.direArgs['real_step'],
    )

    sample_fn = opt.diffusion.p_sample_loop if not opt.direArgs['use_ddim'] else opt.diffusion.ddim_sample_loop
    recons = sample_fn(
        opt.diffusionModel,
        (1, 3, opt.direArgs['image_size'], opt.direArgs['image_size']),
        noise=latent,
        clip_denoised=opt.direArgs['clip_denoised'],
        model_kwargs=model_kwargs,
        # real_step=opt.direArgs['real_step'],
    )

    dire = torch.abs(img - recons)
    dire = (dire * 255.0 / 2.0).clamp(0, 255).to(torch.uint8)
    dire = dire.permute(0, 2, 3, 1)
    dire = dire.contiguous()
    for i in range(len(dire)):
        retval, buffer = cv2.imencode(".png", cv2.cvtColor(dire[i].cpu().numpy().astype(np.uint8), cv2.COLOR_RGB2BGR))
        if retval:
            img_dire = Image.open(BytesIO(buffer)).convert('RGB')

    return resnet_processing(img_dire, opt)


def custom_resize(img, opt):
    rz_dict = {
        'bilinear': Image.BILINEAR,
        'bicubic': Image.BICUBIC,
        'lanczos': Image.LANCZOS,
        'nearest': Image.NEAREST
    }

    interp = opt.rz_interp[0] if len(opt.rz_interp) == 1 else choice(opt.rz_interp)
    return F.resize(img, opt.loadSize, interpolation=rz_dict[interp])


def normalize(img):
    img -= img.min()
    if img.max() != 0: 
        img /= img.max()
    return img * 255. 


def defake_processing(img, opt):

    transformations = []

    if opt.loadSize:
        transformations.append(transforms.Resize(size=(opt.loadSize, opt.loadSize)))

    if opt.isTrain:
        crop_func = transforms.RandomCrop(opt.cropSize)
    else:
        crop_func = transforms.CenterCrop(opt.cropSize)

    transformations.append(crop_func)
    transformations.append(transforms.ToTensor())
    transformations.append(transforms.Normalize(mean=MEAN['clip'], std=STD['clip']))
    
    transform = transforms.Compose(transformations)
    img = transform(img)

    return img


def processing(img, opt, label, image_path):
    assert opt.modelName in VALID_MODELS

    if opt.modelName == 'Fusing':
        input_img, cropped_img, scale = psm_processing(img, opt)
        return input_img, cropped_img, scale, label, image_path
    
    if opt.modelName == 'LGrad':
        return lgrad_processing(img, opt), label, image_path
    
    if opt.modelName == 'CNNDetect':
        return resnet_processing(img, opt), label, image_path
    
    if opt.modelName == 'UnivFD':
        return clip_processing(img, opt), label, image_path
    
    if opt.modelName == 'RPTC':
        return rptc_processing(img, opt), label, image_path
    
    if opt.modelName == 'DIMD':
        return resnet_processing(img, opt), label, image_path
    
    if opt.modelName == 'NPR':
        return resnet_processing(img, opt), label, image_path
    
    if opt.modelName == 'Rine':
        return clip_processing(img, opt), label, image_path

    if opt.modelName == 'FreqDetect':
        return freq_processing(img, opt), label, image_path
    
    if opt.modelName == 'GramNet':
        return resnet_processing(img, opt), label, image_path
    
    if opt.modelName == 'Dire':
        return dire_processing(img, opt), label, image_path
    
    if opt.modelName == 'DeFake':
        return defake_processing(img, opt), label, image_path
    
    raise ValueError(f"Model {opt.modelName} not found")


