import os
import sys
import numpy as np
import torch.utils.data as data
import random

from PIL import Image, ImageEnhance
from scipy.ndimage import rotate


class NITREDataset(data.Dataset):

    def __init__(self, opt):

        ''' set appropriate folder paths and image file type '''
        self.root = opt['root']
        self.path_haze = opt['root'] + "/" + opt['haze']
        self.length = 0

        self.image_paths_haze = sorted([f for f in os.listdir(self.path_haze)])
        self.is_train = bool(opt['is_train'])
        if self.is_train:
            self.path_clean = opt['root'] + "/" + opt['image']
            self.image_paths_clean = sorted([f for f in os.listdir(self.path_clean)])
            self.path_trans = opt['root'] + "/" + opt['trans']
            self.image_paths_trans = sorted([f for f in os.listdir(self.path_trans)])
            self.path_atmos = opt['root'] + "/" + opt['atmos']
            self.image_paths_atmos = sorted([f for f in os.listdir(self.path_atmos)])

            self.augment = bool(opt['augment'])
            if self.augment:
                self.transforms = opt['transforms']

        self.length = len(self.image_paths_haze)

    def __len__(self):
        return self.length 

    def transform_data(self, images):
        rand = np.random.randint(2, size=2)

        if "saturation" in self.transforms and rand[0] == 1:
            ''' Saturation '''
            enhance_factor = np.random.uniform(0.0,1.0)
            i = 0
            while i < len(images):
                enhancer = ImageEnhance.Color(Image.fromarray(images[i]))
                images[i] = np.array(enhancer.enhance(enhance_factor))
                i += 1
        if "rotation" in self.transforms and rand[1] == 1:
            ''' Rotation (with reflective padding) '''
            angle = random.randrange(0,360)
            i = 0
            while i < len(images):
                images[i] = rotate(images[i], angle, reshape=False, mode='reflect')
                i += 1

        return images

    def __getitem__(self, idx):
        input_image = np.array(Image.open(self.path_haze + "/" + self.image_paths_haze[idx]))
        label_image = 0
        label_trans = 0
        label_atmos = 0
        if self.is_train:
            label_image = np.array(Image.open(self.path_clean + "/" + self.image_paths_clean[idx]))
            label_trans = np.array(Image.open(self.path_trans + "/" + self.image_paths_trans[idx]))
            label_atmos = np.array(Image.open(self.path_atmos + "/" + self.image_paths_atmos[idx]))
            if self.augment:
                input_image, label_image, label_trans, label_atmos = self.transform_data([input_image, label_image, label_trans, label_atmos])
            label_image = np.rollaxis(label_image, 2, 0).astype(np.float32)/255
            label_trans = label_trans.astype(np.float32)
            label_atmos = label_atmos.astype(np.float32)
        input_image = np.rollaxis(input_image, 2, 0).astype(np.float32)/255
        return input_image, label_image, label_trans, label_atmos
