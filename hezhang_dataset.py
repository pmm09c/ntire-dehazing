import os
import sys
import numpy as np
import torch.utils.data as data
from PIL import Image, ImageEnhance
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import h5py

class HeZhangDataset(data.Dataset):
    
    def __init__(self, opt):
        self.root = opt['root']
        self.path_haze = opt['root'] + "/" + opt['haze']
        self.length = 0        
        self.image_paths_haze = sorted([f for f in os.listdir(self.path_haze) ])
        self.length = len(self.image_paths_haze)
        
    def __getitem__(self, idx):
        f=h5py.File(self.path_haze+"/"+self.image_paths_haze[idx],'r')
        haze =  np.rollaxis(f['haze'][:],2,0).astype(np.float32)
        image = np.rollaxis(f['gt'][:],2,0).astype(np.float32)
        trans = np.rollaxis(f['trans'][:],2,0).astype(np.float32)
        atmos = np.rollaxis(f['ato'][:],2,0).astype(np.float32)
        return haze,image,trans,atmos
    
    def __len__(self):
        return self.length 
