import os
import sys
import json
import torch
import torch.nn as nn
import numpy as np
import pytorch_msssim

# internal libraries
from models.models import LinkNet,FullNet
from nitre_dataset import NITREDataset

# Load config file 
opt_file = open(sys.argv[1], "r")
opt = json.load(opt_file)
opt_file.close()

# Verify weights directory exists, if not create it
if not os.path.isdir(opt['weights_path']):
    os.makedirs(opt['weights_path'])

print('Running with following config:')
print(json.dumps(opt, indent=4, sort_keys=True))

# Hyper-Parameters
num_epochs = opt['num_epochs']
learning_rate = opt['learning_rate']

# Model
device = torch.device(opt['device'])
model = FullNet().to(device)
model = nn.DataParallel(model)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
try:
    model.trans.load_state_dict(torch.load(sys.argv[2]))
    model.atmos.load_state_dict(torch.load(sys.argv[3]))
except Exception as e:
    try:
        model.load_state_dict(torch.load(sys.argv[2]))
    except Exception as e:
        print("No weights. Training from scratch.")

# Dataset
train_dataset = NITREDataset(opt)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=opt['batch_size'], 
                                           shuffle=True)
total_step = len(train_loader)
image_loss = nn.MSELoss()
    
best_loss = np.inf
pad = nn.ReflectionPad2d((0,0,8,8)).to(device)
crop = nn.ReflectionPad2d((0,0,-8,-8)).to(device)
msssim = pytorch_msssim.MSSSIM()
for epoch in range(num_epochs):
    epoch_loss = 0
    for i, (haze,image) in enumerate(train_loader):
        haze = haze.to(device)
        haze = pad(haze)
        image = image.to(device)
        # compute required losses
        loss_msg = ''
        output,trans,atmos,dehaze = model(haze)
        output=crop(output)
        # TODO: Add Flag MSE
        loss = image_loss(output,image)
        loss_msg += ' Image Loss : {:.4f}'.format(loss.item())
        # TODO: Add Flag MSSSIM
        loss_msssim = (1-msssim(output,image))/2
        loss_msg += ' MS-SSIM Loss : {:.4f}'.format(loss_msssim.item())
        # TODO: Add SSIM
        # TODO: Add discriminator
        loss += loss_msssim
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        print ("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f} Avg Epoch Loss: {:.4f}".format(epoch+1, num_epochs, i+1, total_step, loss.item(),epoch_loss/(i+1))+loss_msg)
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        torch.save(model.state_dict(), opt['weights_path'] + "/NITRE_" + str(epoch) + ".ckpt")
        


            




