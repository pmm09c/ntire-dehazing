import os
import sys
import json
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from time import time

# internal libraries
from models.models import LinkNet,FastNet,FastNet50,DualFastNet
from hezhang_dataset import HeZhangDataset
from ntire_dataset import NTIREDataset

# Load config file 
opt_file = open(sys.argv[1], "r")
opt = json.load(opt_file)
opt_file.close()

# Verify weights directory exists, if not create it
if not os.path.isdir(opt['results_path']):
    os.makedirs(opt['results_path'])

print('Running with following config:')
print(json.dumps(opt, indent=4, sort_keys=True))

# Mode
MODE = opt['mode'].upper()
device = torch.device(opt['device'])
if MODE == 'TRANS':
    model = LinkNet().to(device)
    model.load_state_dict(torch.load(sys.argv[2]))
elif MODE == 'ATMOS':
    model = LinkNet().to(device)
    model.load_state_dict(torch.load(sys.argv[2]))
elif MODE == 'DUAL':
    model = DualFastNet().to(device)
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(sys.argv[2]))
elif MODE == "FAST":
    model = FastNet().to(device)
    model.load_state_dict(torch.load(sys.argv[2]))
elif MODE == "FAST50":
    model = FastNet50().to(device)
    model.load_state_dict(torch.load(sys.argv[2]))
else:
    print('MODE INCORRECT : TRANS or ATMOS or FULL')
    exit()

# Dataset
if opt['dataset'].upper() == 'NTIRE':
    train_dataset = NTIREDataset(opt)
else:
    train_dataset = HeZhangDataset(opt)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset)
total_step = len(train_loader)

def get_pad(size):
    pad = [0, 0, 0, 0]
    x_pad = 32 - (size[0] % 32)
    y_pad = 32 - (size[1] % 32)
    if x_pad != 32:
        if x_pad % 2 != 0:
            pad[0] = (x_pad - 1)/2
            pad[1] = (x_pad - 1)/2 + 1
        else:
            pad[0] = x_pad/2
            pad[1] = x_pad/2
    if y_pad != 32:
        if y_pad % 2 != 0:
            pad[2] = (y_pad - 1)/2
            pad[3] = (y_pad - 1)/2 + 1
        else:
            pad[2] = y_pad/2
            pad[3] = y_pad/2
    pad = [int(p) for p in pad]
    return pad

for i, (haze,_,_,_) in enumerate(train_loader):
    pad_coords = get_pad([haze.shape[3], haze.shape[2]]) 
    pad = nn.ReflectionPad2d((pad_coords[0],pad_coords[1],pad_coords[2],pad_coords[3])).to(device)
    crop = nn.ReflectionPad2d((-pad_coords[0],-pad_coords[1],-pad_coords[2],-pad_coords[3])).to(device)

    haze = haze.to(device)
    haze = pad(haze)
    t = time()
    if MODE == 'TRANS':
        output = model(haze)
    elif MODE == 'ATMOS':
        output = model(haze)
    else:
        # All other models return dehazed image as first output
        output = model(haze)
        output = output[0]
    t = time() - t
    output=crop(output)

    print ("Process Time {:.4f} Step [{}/{}]".format(t, i+1, total_step))
    output = np.clip(np.rollaxis(output.cpu().detach().numpy(),1,4)*255,0,255)
    print(output.shape)
    image = Image.fromarray(output[0].astype(np.uint8))
    image.save(opt['results_path']+"/"+str(i)+".png")
