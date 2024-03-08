import torch.nn as nn 
import torch

from networks.resnet_gram import resnet18


class GramNet(nn.Module):
    def __init__(self, num_classes=1):
        super(GramNet, self).__init__()
        self.model = resnet18(num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

    def load_weights(self, ckpt):
        state_dict = torch.load(ckpt, map_location='cpu')
        try:
            self.model.load_state_dict(state_dict['netC'])
        except:
            self.model.load_state_dict(state_dict)

    def predict(self, img):
        with torch.no_grad():
            logits = self.forward(img)
            return logits.sigmoid().flatten().tolist()
        