import os
import sys
import numpy as np
import torch.utils.data as data
from PIL import Image, ImageEnhance


class NITREDataset(data.Dataset):

    def __init__(self, opt):

        ''' set appropriate folder paths and image file type '''
        self.root = opt['root']
        self.path_in = opt['root'] + "/" + opt['haze']
        self.length = 0

        self.image_paths_haze = sorted([f for f in os.listdir(self.path_in)])
        self.is_train = bool(opt['is_train'])
        if self.is_train:
            self.path_out = opt['root'] + "/" + opt['image']
            self.image_paths_clean = sorted([f for f in os.listdir(self.path_out)])
        self.length = len(self.image_paths_haze)

        
    def __len__(self):
        return self.length 

    def __getitem__(self, idx):
        input_image = np.array(Image.open(self.path_in + "/" + self.image_paths_haze[idx]))
        input_image = np.rollaxis(input_image, 2, 0).astype(np.float32)/255
        label_image = 0
        if self.is_train:
            label_image = np.array(Image.open(self.path_out + "/" + self.image_paths_clean[idx]))
            label_image = np.rollaxis(label_image, 2, 0).astype(np.float32)/255
        return input_image, label_image
