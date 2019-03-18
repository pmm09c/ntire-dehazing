import os
import sys
import json
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from time import time

# internal libraries
from models.models import LinkNet,FullNet
from nitre_dataset import NITREDataset

# Load config file 
opt_file = open(sys.argv[1], "r")
opt = json.load(opt_file)
opt_file.close()

# Verify weights directory exists, if not create it
if not os.path.isdir(opt['results_path']):
    os.makedirs(opt['results_path'])

print('Running with following config:')
print(json.dumps(opt, indent=4, sort_keys=True))
device = torch.device(opt['device'])
model = FullNet().to(device)
model = nn.DataParallel(model)
model.load_state_dict(torch.load(sys.argv[2]))

# Dataset
train_dataset = NITREDataset(opt)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset)
total_step = len(train_loader)
pad = nn.ReflectionPad2d((0,0,8,8))
crop = nn.ReflectionPad2d((0,0,-8,-8)).to(device)
for i, (haze,_) in enumerate(train_loader):
    haze = haze.to(device)
    haze = pad(haze)
    t = time()
    output,trans,atmos,dehaze = model(haze)
    output=crop(output)
    t = time() - t
    print ("Process Time {:.4f} Step [{}/{}]".format(t, i+1, total_step))
    output = np.clip(np.rollaxis(output.cpu().detach().numpy(),1,4)*255,0,255)
    trans = np.clip(np.rollaxis(trans.cpu().detach().numpy(),1,4)*255,0,255)
    atmos = np.clip(np.rollaxis(atmos.cpu().detach().numpy(),1,4)*255,0,255)
    dehaze = np.clip(np.rollaxis(dehaze.cpu().detach().numpy(),1,4)*255,0,255)
    image = Image.fromarray(output[0].astype(np.uint8))
    image.save(opt['results_path']+"/"+str(i)+".png")
    image = Image.fromarray(trans[0].astype(np.uint8))
    image.save(opt['results_path']+"/"+str(i)+"_trans.png")
    image = Image.fromarray(atmos[0].astype(np.uint8))
    image.save(opt['results_path']+"/"+str(i)+"_atmos.png")
    image = Image.fromarray(dehaze[0].astype(np.uint8))
    image.save(opt['results_path']+"/"+str(i)+"_dehaze.png")
    

               


            




