import copy
import json
from models import get_model

import torch
from torchvision import transforms


MEAN = { 
    "imagenet":[0.485, 0.456, 0.406], 
    "clip":[0.48145466, 0.4578275, 0.40821073] 
}

STD = { 
    "imagenet":[0.229, 0.224, 0.225], 
    "clip":[0.26862954, 0.26130258, 0.27577711] 
}


def psm_processing(img, cropSize, resizeSize):
    height, width = img.height, img.width

    input_img = copy.deepcopy(img)
    input_img = transforms.ToTensor()(input_img)
    input_img = transforms.Normalize(MEAN['imagenet'], STD['imagenet'])(input_img)

    if resizeSize is not None:
        img = transforms.Resize(resizeSize)(img)

    img = transforms.CenterCrop(cropSize)(img)
    cropped_img = transforms.ToTensor()(img)
    cropped_img = transforms.Normalize(MEAN['imagenet'], STD['imagenet'])(cropped_img)

    scale = torch.tensor([height, width])

    return input_img, cropped_img, scale


if __name__ == "__main__":

    with open("models.json", "r") as f:
        models = json.load(f)

    for model_metadata in models:
        if model_metadata["model_name"] != "Rine":
            continue

        print(model_metadata["model_name"])
        print(model_metadata["ckpt"])

        del model_metadata["trained_on"]

        model = get_model(**model_metadata)
        model.eval()

        model = model.to("cuda:0")
        img = torch.rand(1, 3, 224, 224).to("cuda:0")
        
        out = model.predict(img)

        print(out)
        print("------------------------------------")
