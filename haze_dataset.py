import os
import sys
import numpy as np
import torch.utils.data as data
from PIL import Image, ImageEnhance
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF


class HazeDataset(data.Dataset):
    def __init__(self, opt, transform = False):
        self.root = opt['root']
        self.image_format = opt['image_format']
        self.path_haze = opt['root'] + "/" + opt['haze']
        self.path_trans = opt['root'] + "/" + opt['trans']
        self.path_atmos = opt['root'] + "/" + opt['atmos']
        self.path_image = opt['root'] + "/" + opt['image']
        self.length = 0        
        self.image_paths_haze = sorted([f for f in os.listdir(self.path_haze) if f[-4:] == self.image_format])
        self.image_paths_trans = sorted([f for f in os.listdir(self.path_trans) if f[-4:] == self.image_format])
        self.image_paths_atmos = sorted([f for f in os.listdir(self.path_atmos) if f[-4:] == self.image_format])
        self.image_paths_image = sorted([f for f in os.listdir(self.path_image) if f[-4:] == self.image_format])
        self.return_trans = bool(opt['trans_on'])
        self.return_atmos = bool(opt['atmos_on'])
        self.return_image = bool(opt['image_on'])
        self.length = len(self.image_paths_haze)
    def __getitem__(self, idx):
        haze = np.array(Image.open(self.path_haze + "/" + self.image_paths_haze[idx])).astype(np.float32)/255
        haze = np.rollaxis(haze, 2, 0)
        trans = 0
        atmos = 0
        image = 0
        if self.return_trans:
            trans = np.array(Image.open(self.path_trans+"/"+self.image_paths_trans[idx]).convert('L')).astype(np.float32)/255
        if self.return_atmos:
            atmos = np.array(Image.open(self.path_atmos+"/"+self.image_paths_atmos[idx]).convert('L')).astype(np.float32)/255
        if self.return_image:
            image = np.array(Image.open(self.path_image+"/"+self.image_paths_image[idx])).astype(np.float32)/255
        return haze,image,trans,atmos
    
    def __len__(self):
        return self.length 
