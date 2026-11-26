import torch.nn as nn 
import torch

from networks.resnet_npr import resnet50


class NPR(nn.Module):
    def __init__(self):
        super(NPR, self).__init__()
        self.model = resnet50(num_classes=1)

    def forward(self, x):
        return self.model(x)

    def load_weights(self, ckpt):
        checkpoint = torch.load(ckpt, map_location="cpu")

        state_dict = checkpoint.get("model", checkpoint)

        # Remove DataParallel prefix if present
        if state_dict and all(k.startswith("module.") for k in state_dict):
            state_dict = {
                k.removeprefix("module."): v
                for k, v in state_dict.items()
            }

        self.model.load_state_dict(state_dict, strict=True)

    def predict(self, img):
        with torch.no_grad():
            return self.forward(img).sigmoid().flatten().tolist()
        