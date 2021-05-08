import os
import sys
import copy
import random
from abc import ABC, abstractmethod

import scipy
import pickle
from PIL import ImageFilter
import numpy as np

import torch.nn as nn
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import kornia.augmentation as K
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate

sys.path.append("..")
from util import io

from logging import getLogger
logger = getLogger()

def get_custom_dataset(dataset_type, root, split, download=False, return_target_word=False):
    if dataset_type == "Cifar100":
        return Cifar100Dataset(root, split, download, return_target_word)
    elif dataset_type == 'STL10':
        return STL10Dataset(root, split, download, return_target_word)
    else:
        raise NotImplementedError("{} is not implemented yet".format(dataset_type))

class BaseDataset(Dataset, ABC):
    def __init__(self, root, split, download, return_target_word):
        super(BaseDataset,self).__init__()
        self.root = root
        self.split = split
        self.download = download
        self.return_target_word = return_target_word
        
        if self.download==True:
            io.mkdir_if_not_exists(self.root)
        
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0., 0., 0.), (1., 1., 1.))
        ])
        
        self.transforms_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0., 0., 0.), (1., 1., 1.))
        ])

    @abstractmethod
    def load_labels(self):
        '''fill in acccording to the chosen dataset. used by <__getitem__>'''
        pass
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self,i):
        img,idx = self.dataset[i]
        if self.split=='train':
            img = self.transforms(img)
        else:
            img = self.transforms_test(img)

        if self.return_target_word:
            label = self.labels[idx]

            to_return = {
                'img': img,
                'label_idx': idx,
                'label': label
            }
            return to_return
        else:
            return img, idx

    def normalize(self,imgs,mean,std):
        imgs = (imgs-mean) / std
        return imgs

    def get_collate_fn(self):
        def collate_fn(batch):
            batch = [s for s in batch if s is not None]
            return default_collate(batch)

        return collate_fn


class STL10Dataset(BaseDataset):
    def __init__(self, root, split, download, return_target_word):
        super().__init__(root, split, download, return_target_word)
        self.dataset = torchvision.datasets.STL10(
            self.root,
            split=split,
            download=self.download
        )
        self.labels = self.load_labels()
        self.transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(96,padding=2),
            transforms.ToTensor(),
            transforms.Normalize((0., 0., 0.), (1., 1., 1.))
        ])
        
        self.transforms_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0., 0., 0.), (1., 1., 1.))
        ])

    def load_labels(self):
        meta_file = os.path.join(
            self.root,
            'stl10_binary/class_names.txt')
        with open(meta_file,'r') as mf:
            labels = mf.readlines()
            labels = np.array([lb.strip() for lb in labels])
        return labels

class Cifar100Dataset(BaseDataset):
    def __init__(self, root, split, download, return_target_word):
        super().__init__(root, split, download, return_target_word)
        self.dataset = torchvision.datasets.CIFAR100(
            self.root,
            self.split == 'train',
            download=self.download)
        self.labels = self.load_labels()
        self.transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32,padding=2),
            transforms.ToTensor(),
            transforms.Normalize((0., 0., 0.), (1., 1., 1.))
        ])
        
        self.transforms_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0., 0., 0.), (1., 1., 1.))
        ])
        
    def load_labels(self):
        meta_file = os.path.join(
            self.root,
            'cifar-100-python/meta')
        fo = open(meta_file,'rb')
        labels = pickle.load(fo,encoding='latin1')['fine_label_names']
        return labels

class MultiCropDataset(datasets.ImageFolder):
    def __init__(
        self,
        data_path,
        size_crops,
        nmb_crops,
        min_scale_crops,
        max_scale_crops,
        size_dataset=-1,
        return_index=False,
    ):
        super(MultiCropDataset, self).__init__(data_path)
        assert len(size_crops) == len(nmb_crops)
        assert len(min_scale_crops) == len(nmb_crops)
        assert len(max_scale_crops) == len(nmb_crops)
        if size_dataset >= 0:
            self.samples = self.samples[:size_dataset]
        self.return_index = return_index

        color_transform = [get_color_distortion(), PILRandomGaussianBlur()]
        mean = [0.485, 0.456, 0.406]
        std = [0.228, 0.224, 0.225]
        trans = []
        for i in range(len(size_crops)):
            randomresizedcrop = transforms.RandomResizedCrop(
                size_crops[i],
                scale=(min_scale_crops[i], max_scale_crops[i]),
            )
            trans.extend([transforms.Compose([
                randomresizedcrop,
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Compose(color_transform),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)])
            ] * nmb_crops[i])
        self.trans = trans

    def __getitem__(self, index):
        path, _ = self.samples[index]
        image = self.loader(path)
        multi_crops = list(map(lambda trans: trans(image), self.trans))
        if self.return_index:
            return index, multi_crops
        return multi_crops

class KorniaAugmentationPipeline(nn.Module):
    def __init__(self,
                s_color=0.5, 
                p_color=0.8, 
                p_flip=0.5,
                p_gray=0.2, 
                p_blur=0.5, 
                kernel_min=0.1, 
                kernel_max=2.) -> None:
        super(KorniaAugmentationPipeline, self).__init__()
        
        T_hflip = K.RandomHorizontalFlip(p=p_flip) 
        T_gray = K.RandomGrayscale(p=p_gray)
        T_color = K.ColorJitter(p_color, 0.8*s_color, 0.8*s_color, 0.8*s_color, 0.2*s_color)

        radius = kernel_max*2  # https://dsp.stackexchange.com/questions/10057/gaussian-blur-standard-deviation-radius-and-kernel-size
        kernel_size = int(radius*2 + 1) # needs to be odd.
        kernel_size = (kernel_size, kernel_size)
        T_blur = K.GaussianBlur(kernel_size=kernel_size, sigma=(kernel_min, kernel_max), p=p_blur)
        #T_blur = KorniaRandomGaussianBlur(kernel_size=kernel_size, sigma=(kernel_min, kernel_max), p=p_blur)

        self.transform = nn.Sequential(
            T_hflip,
            T_color,
            T_gray,
            T_blur
        )

    def forward(self, input):
        out = self.transform(input)
        return out

class PILRandomGaussianBlur(object):
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = np.random.rand() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )

def get_color_distortion(s=1.0):
    # s is the strength of color distortion.
    color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([rnd_color_jitter, rnd_gray])
    return color_distort

'''

class VOCDetectionDataset(BaseDataset):
    def __init__(self,root, download=False):
        super().__init__(root, download)
        self.dataset = torchvision.datasets.STL10(
            self.root,
            split='train' if self.train else 'test',
            download=self.download
        )
        self.labels = self.load_labels()
        self.transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(96,padding=2),
            transforms.ToTensor(),
            transforms.Normalize((0., 0., 0.), (1., 1., 1.))
        ])
        
        self.transforms_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0., 0., 0.), (1., 1., 1.))
        ])

    def load_labels(self):
        meta_file = os.path.join(
            self.root,
            'stl10_binary/class_names.txt')
        with open(meta_file,'r') as mf:
            labels = mf.readlines()
            labels = np.array([lb.strip() for lb in labels])
        return labels


class KorniaRandomGaussianBlur(nn.Module):
    """
    Apply Gaussian Blur to the PIL image. Take the radius and probability of
    application as the parameter.
    This transform was used in SimCLR - https://arxiv.org/abs/2002.05709
    """

    def __init__(self, kernel_size, sigma, p):
        super(KorniaRandomGaussianBlur, self).__init__()
        self.prob = p
        self.sigma = sigma
        self.kernel_size = kernel_size
        self.blur = KorniaGaussianBlur(kernel_size, sigma)

    def forward(self, img):
        do_it = np.random.rand() <= self.prob
        if not do_it:
            return img
        return self.blur(img)
'''

