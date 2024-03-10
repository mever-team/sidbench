import torch
import torch.nn as nn
import numpy as np
from models.srm_filter_kernel import all_normalized_hpf_list


class TLU(nn.Module):
    def __init__(self, threshold):
        super(TLU, self).__init__()
        self.threshold = threshold

    def forward(self, input):
        output = torch.clamp(input, min=-self.threshold, max=self.threshold)
        return output


class HPF(nn.Module):
    def __init__(self):
        super(HPF, self).__init__()

        #Load 30 SRM Filters
        all_hpf_list_5x5 = []

        for hpf_item in all_normalized_hpf_list:
            if hpf_item.shape[0] == 3:
                hpf_item = np.pad(hpf_item, pad_width=((1, 1), (1, 1)), mode='constant')

            all_hpf_list_5x5.append(hpf_item)

        hpf_weight = nn.Parameter(torch.Tensor(np.array(all_hpf_list_5x5)).view(30, 1, 5, 5), requires_grad=False)

        self.hpf = nn.Conv2d(1, 30, kernel_size=5, padding=2, bias=False)
        self.hpf.weight = hpf_weight

        #Truncation, threshold = 3 
        self.tlu = TLU(3.0)

    def forward(self, input):
        output = self.hpf(input)
        return output


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.group1 = HPF()

        self.group1_b = nn.Sequential(
            nn.Conv2d(90, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.Hardtanh(min_val=-5,max_val=5),
        )

        self.group2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.AvgPool2d(kernel_size=3, padding=1, stride=2)
        )

        self.group3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.AvgPool2d(kernel_size=3, padding=1, stride=2)
        )

        self.group4 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.AvgPool2d(kernel_size=3, padding=1, stride=2)
        )

        self.group5 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.fc2 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.advpool = nn.AdaptiveAvgPool2d((1,1))

    def forward(self, input):
        img_poor = input[:,0,:,:,:]
        img_rich = input[:,1,:,:,:]
        
        a,b,c,d = img_poor.shape

        img_poor = img_poor.reshape(-1,1,c,d)
        img_poor = self.group1(img_poor)
        img_poor = img_poor.reshape(a,-1,c,d)
        img_poor = self.group1_b(img_poor)

        img_rich = img_rich.reshape(-1,1,c,d)
        img_rich = self.group1(img_rich)
        img_rich = img_rich.reshape(a,-1,c,d)
        img_rich = self.group1_b(img_rich)

        res = img_poor - img_rich
        # res = torch.cat((img_poor,img_rich),1)
        F_poor = img_poor.clone().view(a,-1)
        F_rich = img_rich.clone().view(a,-1)
        output = self.group2(res)
        F1 = output.clone()
        output = self.group3(output)
        F2 = output.clone()
        output = self.group4(output)
        F3 = output.clone()
        output = self.group5(output)
        F4 = output.clone()

        output = self.advpool(output)
        output = output.view(output.size(0), -1)
        F = output.detach().clone()
        out = self.fc2(output)

        return out

    def load_weights(self, ckpt):
        self.apply(initWeights)
    
        state_dict = torch.load(ckpt, map_location='cpu')
        try:
            self.load_state_dict(state_dict['netC'], strict=True)
        except:
            try:
                self.load_state_dict(state_dict, strict=True)
            except:
                self.load_state_dict({k.replace('module.', ''): v for k, v in state_dict['netC'].items()})


    def predict(self, img):
        with torch.no_grad():
            return self.forward(img).sigmoid().flatten().tolist()
    

def initWeights(module):
    if type(module) == nn.Conv2d:
        if module.weight.requires_grad:
            nn.init.kaiming_normal_(module.weight.data, mode='fan_in', nonlinearity='relu')

    if type(module) == nn.Linear:
        nn.init.normal_(module.weight.data, mean=0, std=0.01)
        nn.init.constant_(module.bias.data, val=0)
