import os

from io import BytesIO 
from PIL import Image 
from PIL import ImageFile
import numpy as np 

from scipy.ndimage import gaussian_filter

import torch
from torch.utils.data import Dataset

import random


ImageFile.LOAD_TRUNCATED_IMAGES = True


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

    gaussian_filter(img[:, :, 0], output=img[:, :, 0], sigma=sigma)
    gaussian_filter(img[:, :, 1], output=img[:, :, 1], sigma=sigma)
    gaussian_filter(img[:, :, 2], output=img[:, :, 2], sigma=sigma)

    return Image.fromarray(img)


IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp')


def get_list(path, exts=IMAGE_EXTENSIONS):
    """All images under path, searching every subfolder, sorted."""
    image_list = []
    for r, _, f in os.walk(path):
        for file in f:
            # The extension is the part after the last dot, in any case: img.v2.PNG -> .png
            if os.path.splitext(file)[1].lower() in exts:
                image_list.append(os.path.join(r, file))

    return sorted(image_list)


def sample(images, max_sample):
    """A random max_sample of the images (all of them if max_sample is None), without changing the list given."""
    images = list(images)
    if max_sample is not None and max_sample < len(images):
        random.shuffle(images)
        images = images[:max_sample]
    return images


class SyntheticImagesDataset(Dataset):
    def __init__(self, real_list, fake_list, opt, process_fn):
        """real_list and fake_list are the image paths, e.g. from dataset.sources.list_images()."""

        self.jpeg_quality = opt.jpegQuality
        self.gaussian_sigma = opt.gaussianSigma

        self.opt = opt
        self.process_fn = process_fn

        self.real_list = sample(real_list, opt.maxSample)
        self.fake_list = sample(fake_list, opt.maxSample)
        self.total_list = self.real_list + self.fake_list

        # labels
        self.labels_dict = {}
        for i in self.real_list:
            self.labels_dict[i] = 0
        for i in self.fake_list:
            self.labels_dict[i] = 1

    def merge(self, other_dataset):
        assert self.jpeg_quality == other_dataset.jpeg_quality
        assert self.gaussian_sigma == other_dataset.gaussian_sigma

        self.total_list += other_dataset.total_list
        self.labels_dict.update(other_dataset.labels_dict)
        self.real_list += other_dataset.real_list
        self.fake_list += other_dataset.fake_list

    def remove(self, other_dataset):
        assert self.jpeg_quality == other_dataset.jpeg_quality
        assert self.gaussian_sigma == other_dataset.gaussian_sigma

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

        return self.process_fn(img, self.opt, label, img_path)

    
class RecursiveImageDataset(Dataset):
    def __init__(self, data_path, opt, process_fn):
        """
        Args:
            data_path (string): Directory with all the images, including subdirectories.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_path = data_path
        self.process_fn = process_fn
        self.opt = opt

        self.image_paths = []
        self._load_image_paths(data_path)

    def _load_image_paths(self, dir_path):
        """Recursively load all image paths from the directory."""
        self.image_paths.extend(get_list(dir_path))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')

        return self.process_fn(image, self.opt, 0, img_path)
