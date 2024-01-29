from networks.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from networks.resnet50nodown import resnet50nodown
from networks.vision_transformer import vit_b_16, vit_b_32, vit_l_16, vit_l_32

import torch.nn as nn 
import torch

model_dict = {
    'resnet18': resnet18,
    'resnet34': resnet34,
    'resnet50': resnet50,
    'resnet50nodown': resnet50nodown,
    'resnet101': resnet101,
    'resnet152': resnet152,
    'vit_b_16': vit_b_16,
    'vit_b_32': vit_b_32,
    'vit_l_16': vit_l_16,
    'vit_l_32': vit_l_32
}


CHANNELS = {
    "resnet50nodown" : 2048,
    "resnet50" : 2048,
    "vit_b_16" : 768,
}


class ImagenetModel(nn.Module):
    def __init__(self, name, num_classes=1):
        super(ImagenetModel, self).__init__()

        self.name = name
        # self.model = model_dict['resnet50'](pretrained=True)
        # self.model.fc = nn.Linear(CHANNELS['resnet50'], 1)

        self.model = model_dict[name](num_classes=num_classes)
        # self.model.fc = nn.Linear(CHANNELS[name], num_classes)
        # self.fc = nn.Linear(CHANNELS[name], num_classes) #manually define a fc layer here

    def forward(self, x):
        # feature = self.model(x)["penultimate"]
        # return self.fc(feature)

        return self.model(x) if self.name == 'resnet50nodown' else self.model(x)["logits"]

    def load_weights(self, ckpt):
        state_dict = torch.load(ckpt, map_location='cpu')
        self.model.load_state_dict(state_dict['model'])
        # self.fc.load_state_dict({'weight': state_dict['model']['fc.weight'], 'bias': state_dict['model']['fc.bias']})

    def predict(self, img):
        with torch.no_grad():
            logits = self.forward(img)
            return logits.sigmoid().flatten().tolist()



    