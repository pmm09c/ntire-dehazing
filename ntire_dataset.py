import os
import sys
import numpy as np
import torch.utils.data as data
import random

from PIL import Image, ImageEnhance
from scipy.ndimage import rotate
from scipy import interpolate

def crop_and_resample_2darray(arr, x_crop, y_crop, resample):
    """Crop a 2darray and resize the data"""
    
    len_x_crop = x_crop[1]-x_crop[0]
    len_y_crop = y_crop[1]-y_crop[0]

    arr_crop0 = arr[x_crop[0]:x_crop[1], y_crop[0]:y_crop[1],0]
    arr_crop1 = arr[x_crop[0]:x_crop[1], y_crop[0]:y_crop[1],1]
    arr_crop2 = arr[x_crop[0]:x_crop[1], y_crop[0]:y_crop[1],2]

    f0 = interpolate.interp2d(np.arange(len_y_crop), 
                              np.arange(len_x_crop), 
                              arr_crop0)
    f1 = interpolate.interp2d(np.arange(len_y_crop), 
                              np.arange(len_x_crop), 
                              arr_crop1)
    f2 = interpolate.interp2d(np.arange(len_y_crop), 
                              np.arange(len_x_crop), 
                              arr_crop2)
    result = np.array([f0(np.arange(len_x_crop, step=len_x_crop/resample[1]), 
                          np.arange(len_y_crop, step=len_y_crop/resample[0])),
                       f1(np.arange(len_x_crop, step=len_x_crop/resample[1]), 
                          np.arange(len_y_crop, step=len_y_crop/resample[0])),
                       f2(np.arange(len_x_crop, step=len_x_crop/resample[1]), 
                          np.arange(len_y_crop, step=len_y_crop/resample[0]))]).astype(np.uint8)
    result = np.rollaxis(result,0,3) 
    return result

class NTIREDataset(data.Dataset):

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
        self.size = opt['size']

    def __len__(self):
        return self.length 

    def transform_data(self, images):
        #rand = np.random.randint(2, size=3)
        rand = np.random.random((3,1))
        if "saturation" in self.transforms.keys() and rand[0] < self.transforms['saturation']:
            ''' Saturation '''
            enhance_factor = np.random.uniform(0.0,1.0)
            i = 0
            while i < len(images):
                enhancer = ImageEnhance.Color(Image.fromarray(images[i]))
                images[i] = np.array(enhancer.enhance(enhance_factor))
                i += 1
        if "rotation" in self.transforms.keys() and rand[1] < self.transforms['rotation']:
            ''' Rotation (with reflective padding) '''
            angle = random.randrange(0,360)
            i = 0
            while i < len(images):
                images[i] = rotate(images[i], angle, reshape=False, mode='reflect')
                i += 1
        if "crop" in self.transforms.keys() and rand[2] < self.transforms['crop']:
            '''crop and resize'''
            i = 0
            while i < len(images):
                x_crop = int(np.random.randint(400,high=(images[i].shape[0]-400)))
                y_crop = int(np.random.randint(400,high=(images[i].shape[1]-400)))
                x_crop = [x_crop,int(x_crop+400)]
                y_crop = [y_crop,int(y_crop+400)]
                resample = [images[i].shape[0],images[i].shape[1]]
                images[i] = crop_and_resample_2darray(images[i], x_crop, y_crop, resample)
                i += 1
        return images

    def check_size(self, im):
        if self.is_train and im.size != self.size:
            im = im.resize(self.size, Image.BILINEAR)
        return im

    def __getitem__(self, idx):
        input_image = np.array(self.check_size(Image.open(self.path_haze + "/" + self.image_paths_haze[idx])))
        label_image = 0
        label_trans = 0
        label_atmos = 0
        if self.is_train:
            label_image = np.array(self.check_size(Image.open(self.path_clean + "/" + self.image_paths_clean[idx])))
            label_trans = np.array(self.check_size(Image.open(self.path_trans + "/" + self.image_paths_trans[idx])))
            label_atmos = np.array(self.check_size(Image.open(self.path_atmos + "/" + self.image_paths_atmos[idx])))
            if self.augment:
                input_image, label_image, label_trans, label_atmos = self.transform_data([input_image, label_image, label_trans, label_atmos])
            label_image = np.rollaxis(label_image, 2, 0).astype(np.float32)/255
            label_trans = np.rollaxis(label_trans, 2, 0).astype(np.float32)/255
            label_atmos = np.rollaxis(label_atmos, 2, 0).astype(np.float32)/255
        input_image = np.rollaxis(input_image, 2, 0).astype(np.float32)/255
        return input_image, label_image, label_trans, label_atmos
