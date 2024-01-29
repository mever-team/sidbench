import os

from io import BytesIO 
from PIL import Image 
from PIL import ImageFile
import numpy as np 

from scipy.ndimage import gaussian_filter
import torchvision.transforms as transforms
from torch.utils.data import Dataset

import random

ImageFile.LOAD_TRUNCATED_IMAGES = True


MEAN = { 
    "imagenet":[0.485, 0.456, 0.406], 
    "clip":[0.48145466, 0.4578275, 0.40821073] 
}

STD = { 
    "imagenet":[0.229, 0.224, 0.225], 
    "clip":[0.26862954, 0.26130258, 0.27577711] 
}


def png2jpg(img, quality):
    out = BytesIO()
    img.save(out, format='jpeg', quality=quality) # ranging from 0-95, 75 is default
    img = Image.open(out)
    # load from memory before ByteIO closes
    img = np.array(img)
    out.close()
    return Image.fromarray(img)


def gaussian_blur(img, sigma):
    img = np.array(img)

    gaussian_filter(img[:,:,0], output=img[:,:,0], sigma=sigma)
    gaussian_filter(img[:,:,1], output=img[:,:,1], sigma=sigma)
    gaussian_filter(img[:,:,2], output=img[:,:,2], sigma=sigma)

    return Image.fromarray(img)


class SyntheticImageDetectionDataset(Dataset):
    def __init__(self, data_paths, backbone, max_sample=None, is_train=False, jpeg_quality=None, gaussian_sigma=None):    
        assert backbone.lower().split(':')[0] in ["imagenet", "clip"]
        
        self.backbone = backbone
        self.jpeg_quality = jpeg_quality
        self.gaussian_sigma = gaussian_sigma

        self.is_train = is_train

        self.real_list, self.fake_list = self.read_paths(data_paths, max_sample)
        self.total_list = self.real_list + self.fake_list

        # labels
        self.labels_dict = {}
        for i in self.real_list:
            self.labels_dict[i] = 0
        for i in self.fake_list:
            self.labels_dict[i] = 1
        
        stat_from = "imagenet" if backbone.lower().startswith("imagenet") else "clip"

        if is_train:
            crop_func = transforms.RandomCrop(224)
        else:
            crop_func = transforms.CenterCrop(224)

        self.transform = transforms.Compose([
            crop_func,
            transforms.ToTensor(),
            transforms.Normalize( mean=MEAN[stat_from], std=STD[stat_from] ),
        ])

    def merge(self, other_dataset):
        assert self.encoder == other_dataset.encoder
        assert self.jpeg_quality == other_dataset.jpeg_quality
        assert self.gaussian_sigma == other_dataset.gaussian_sigma
        assert self.is_train == other_dataset.is_train

        self.total_list += other_dataset.total_list
        self.labels_dict.update(other_dataset.labels_dict)
        self.real_list += other_dataset.real_list
        self.fake_list += other_dataset.fake_list

    def remove(self, other_dataset):
        assert self.encoder == other_dataset.encoder
        assert self.jpeg_quality == other_dataset.jpeg_quality
        assert self.gaussian_sigma == other_dataset.gaussian_sigma
        assert self.is_train == other_dataset.is_train

        for i in other_dataset.real_list:
            self.real_list.remove(i)

        for i in other_dataset.fake_list:
            self.fake_list.remove(i)

        self.total_list = self.real_list + self.fake_list

        # labels
        self.labels_dict = {}
        for i in self.real_list:
            self.labels_dict[i] = 0
        for i in self.fake_list:
            self.labels_dict[i] = 1


    def read_paths(self, paths, max_sample=None):
        real_list = []
        fake_list = []
        for pth in set(paths):
            real_list += SyntheticImageDetectionDataset.get_list(pth, must_contain='0_real')
            fake_list += SyntheticImageDetectionDataset.get_list(pth, must_contain='1_fake')

        if max_sample is not None:
            random.shuffle(real_list)
            real_list = real_list[0:max_sample] if max_sample < len(real_list) else real_list

            random.shuffle(fake_list)
            fake_list = fake_list[0:max_sample] if max_sample < len(fake_list) else fake_list

        return real_list, fake_list

    @staticmethod
    def get_list(path, must_contain='', exts=["png", "jpg", "JPEG", "jpeg", "bmp"]):
        image_list = [] 
        for r, _, f in os.walk(path):
            for file in f:
                if (file.split('.')[1] in exts) and (must_contain in os.path.join(r, file)):
                    image_list.append(os.path.join(r, file))

        return image_list
    
    def real_len(self):
        return len(self.real_list)
    
    def fake_len(self):
        return len(self.fake_list)
    
    def __len__(self):
        return len(self.total_list)

    def __getitem__(self, idx):
        img_path = self.total_list[idx]

        label = self.labels_dict[img_path]
        img = Image.open(img_path).convert("RGB")

        if self.gaussian_sigma is not None:
            img = gaussian_blur(img, self.gaussian_sigma) 

        if self.jpeg_quality is not None:
            img = png2jpg(img, self.jpeg_quality)

        img = self.transform(img)

        return img, label
    

import torch
import time

if __name__ == '__main__':
     
    base_path = '/fssd8/user-data/manosetro/sid_bench/test'

    # real_path = os.path.join(base_path, 'ojha2023/laion')   
    # fake_path = os.path.join(base_path, 'ojha2023/dalle') 

    # /fssd8/user-data/manosetro/sid_bench/test/wang2020/biggan/
    real_path = os.path.join(base_path, 'wang2020/biggan')
    fake_path = os.path.join(base_path, 'wang2020/biggan')
    full_dataset = SyntheticImageDetectionDataset([real_path, fake_path], backbone='CLIP:ViT-L/14', max_sample=150)
    print('Full dataset: %i' % len(full_dataset))
    print('%i real images' % full_dataset.real_len())
    print('%i fake images' % full_dataset.fake_len())

    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

    print('Train: %i' % len(train_dataset))
    # print('Train: %i real images' % train_dataset.real_len())
    # print('Train: %i fake images' % train_dataset.fake_len())

    print('Test: %i' % len(test_dataset))
    # print('Test: %i real images' % test_dataset.real_len())
    # print('Test: %i fake images' % test_dataset.fake_len())

    print(test_dataset.real_list[0:10])
    loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)
    for img, label in loader:
        print(label)

#     print('-'*20)

    # real_path = os.path.join(base_path, 'synthbuster/raise')   
    # fake_path = os.path.join(base_path, 'synthbuster/dalle3') 
    # dataset2 = SyntheticImageDetectionDataset(real_path, fake_path, encoder='CLIP:ViT-L/14', max_sample=100)
    # print(len(dataset1))
    # print('%i real images' % dataset1.real_len())
    # print('%i fake images' % dataset1.fake_len())
    # print('-'*20)

    # dataset1.merge(dataset2)
    # print(len(dataset1))
    # print('%i real images' % dataset1.real_len())
    # print('%i fake images' % dataset1.fake_len())
    # print('-'*20)

    # dataset1.remove(dataset2)
    # print(len(dataset1))
    # print('%i real images' % dataset1.real_len())
    # print('%i fake images' % dataset1.fake_len())
    # print('-'*20)

    # loader = torch.utils.data.DataLoader(dataset1, batch_size=128, shuffle=False, num_workers=8)
    # print ("Batches: %d" %(len(loader)))

    # t1 = time.time()
    # print ("Start iterating")
    # for img, label in iter(loader):
    #     print (img.shape)

    # print ("Time: %f" %(time.time() - t1))