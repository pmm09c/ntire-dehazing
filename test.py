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
from hezhang_dataset import HeZhangDataset

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
elif MODE == 'FULL':
    model = FullNet().to(device)
    model.load_state_dict(torch.load(sys.argv[2]))
else:
    print('MODE INCORRECT : TRANS or ATMOS or FULL')
    exit()

# Dataset
train_dataset = HeZhangDataset(opt)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset)
total_step = len(train_loader)
for i, (haze,_,_,_) in enumerate(train_loader):
    haze = haze.to(device)
    t = time()
    if MODE == 'TRANS' :
        output = model(haze)
    elif MODE == 'ATMOS' :x
        output = model(haze)
    else :
        output,trans,atmos,dehaze = model(haze)
    t = time() - t
    print ("Process Time {:.4f} Step [{}/{}]".format(t, i+1, total_step))
    output = np.clip(np.rollaxis(output.cpu().detach().numpy(),1,4)*255,0,255)
    print(output.shape)
    image = Image.fromarray(output[0].astype(np.uint8))
    image.save(opt['results_path']+"/"+str(i)+".png")
    
    

               


            




