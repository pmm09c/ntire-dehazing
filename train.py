import os
import sys
import json
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

# internal libraries
from models.models import LinkNet,FullNet,Discriminator
from hezhang_dataset import HeZhangDataset
from nitre_dataset import NITREDataset

# dependencies
from pytorch_ssim import ssim
from pytorch_msssim import MSSSIM


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

# Mode
MODE = opt['mode'].upper()
device = torch.device(opt['device'])
if MODE == 'TRANS':
    model = LinkNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    try:
        model.load_state_dict(torch.load(sys.argv[2]))
    except Exception as e:
        print("No weights. Training from scratch.")
elif MODE == 'ATMOS':
    model = LinkNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    try:
        model.load_state_dict(torch.load(sys.argv[2]))
    except Exception as e:
        print("No weights. Training from scratch.")
elif MODE == 'FULL' or ( MODE == 'GAN' and len(opt['loss_discr']) ):
    model = FullNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    try:
        model.trans.load_state_dict(torch.load(sys.argv[2]))
        model.atmos.load_state_dict(torch.load(sys.argv[3]))
        for param in model.trans.parameters():
            param.requires_grad = False
        for param in model.atmos.parameters():
            param.requires_grad = False
        model = nn.DataParallel(model)
    except Exception as e:
        try:
            model = nn.DataParallel(model)
            model.load_state_dict(torch.load(sys.argv[2]))
        except Exception as e:
            print("No weights. Training from scratch.")
    if MODE == 'GAN':
        model_d = Discriminator().to(device)
        optimizer_d = torch.optim.Adam(model_d.parameters(), lr=learning_rate)
        try:
            model_d.load_state_dict(torch.load(sys.argv[3]))
            model_d = nn.DataParallel(model_d)
        except Exception as e:
            print("No weights. Training from scratch.")
else:
    print('MODE INCORRECT : TRANS or ATMOS or FULL or GAN')
    exit()

# Loss
def sdim(output,target):
    return (1.-ssim(output,target))/2.

def mssdim(output,target):
    return (1.-MSSSIM(output,target))/2.
criterion = {'L1':nn.L1Loss(),'MSE':nn.MSELoss(),'BCE':nn.BCELoss(),'Huber':nn.SmoothL1Loss(),'SSIM':sdim,'MSSSIM':mssdim}
trans_loss = [x.upper() for x in opt['loss_trans']]
atmos_loss = [x.upper() for x in opt['loss_atmos']]
image_loss = [x.upper() for x in opt['loss_image']]
dhaze_loss = [x.upper() for x in opt['loss_dhaze']]
discr_loss = [x.upper() for x in opt['loss_discr']]
trans_criterion = [ criterion[x] for x in trans_loss ]
atmos_criterion = [ criterion[x] for x in atmos_loss ]
image_criterion = [ criterion[x] for x in image_loss ]
dhaze_criterion = [ criterion[x] for x in dhaze_loss ]
discr_criterion = [ criterion[x] for x in discr_loss ]

# Dataset
if opt['dataset'].upper() == 'NITRE':
    train_dataset = NITREDataset(opt)
else:
    train_dataset = HeZhangDataset(opt)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=opt['batch_size'], 
                                           shuffle=True)
total_step = len(train_loader)    
best_loss = np.inf

pad = nn.ReflectionPad2d((0,0,opt['pad'][0],opt['pad'][1])).to(device)
crop = nn.ReflectionPad2d((0,0,-opt['pad'][0],-opt['pad'][1])).to(device)

# Training Loop
for epoch in range(num_epochs):
    epoch_loss = 0

    for i, (haze,image,image_trans,image_atmos) in enumerate(train_loader):
        haze = haze.to(device)
        haze = pad(haze)
        loss_msg = ''       

        # copy required data to device
        if len(image_loss):
            image = image.to(device)
        if len(trans_loss):
            image_trans = image_trans.to(device)
        if len(atmos_loss):
            image_atmos = image_atmos.to(device)

        # Train transmission map network
        if MODE == 'TRANS':
            output = model(haze)
            output = crop(output)
            loss = sum([ c(output, image_trans) for c in trans_criterion ])
            loss_msg += ' Trans Loss : {:.4f}'.format(loss.item())

        # Train atmospheric light estimation network
        elif MODE == 'ATMOS':
            output = model(haze)
            output=crop(output)
            loss = sum([ c(output, image_atmos) for c in atmos_criterion ])
            loss_msg += ' Atmos Loss : {:.4f}'.format(loss.item())

        # Train full network (light atmospheric estimation, transmission map, dehazed image; with or without GAN loss)
        elif MODE == 'FULL' or MODE == 'GAN':
            output,trans,atmos,dehaze = model(haze)
            output = crop(output)
            trans = crop(trans)
            atmos = crop(atmos)
            dehaze = crop(dehaze)
            tloss = sum([ c(trans, image_trans)*w for c,w in zip(trans_criterion,opt['loss_trans_w'])])
            aloss = sum([ c(atmos, image_atmos)*w for c,w in zip(atmos_criterion,opt['loss_atmos_w'])])
            dloss = sum([ c(dehaze, image)*w for c,w in zip(dhaze_criterion,opt['loss_dhaze_w'])])
            iloss = sum([ c(output, image)*w for c,w in zip(image_criterion,opt['loss_image_w'])])
            loss = iloss + tloss + aloss + dloss
            if len(trans_loss):
                loss_msg += ' T : {:.4f}'.format(tloss.item())
            if len(atmos_loss):
                loss_msg += ' A : {:.4f}'.format(aloss.item())
            if len(dhaze_loss):
                loss_msg += ' J : {:.4f}'.format(dloss.item())
            if len(image_loss):
                loss_msg += ' I : {:.4f}'.format(iloss.item())
            if MODE == 'GAN':
                ones_const = Variable(torch.ones(image.shape[0], 1)).to(device)
                target_real = Variable(torch.rand(image.shape[0],1)*0.1 + 0.9).to(device)
                target_fake = Variable(torch.rand(image.shape[0],1)*0.1).to(device)
                real = Variable(image)                
                dloss_r = sum([ c(model_d(image), target_real)*w for c,w in zip(discr_criterion,opt['loss_discr_w'])])
                dloss_f = sum([ c(model_d(Variable(output)), target_fake)*w for c,w in zip(discr_criterion,opt['loss_discr_w'])])

                loss_d = dloss_r + dloss_f
                optimizer_d.zero_grad()
                dloss.backward(retain_graph=True)
                optimizer_d.step()
                gloss = sum([ c(model_d(Variable(output)), ones_const)*w for c,w in zip(discr_criterion,opt['loss_discr_w'])])
                loss_msg += ' advs : {:.4f}'.format(gloss.item())
                loss_msg += ' dreal : {:.4f}'.format(dloss_r.item())
                loss_msg += ' dfake : {:.4f}'.format(dloss_f.item())
                loss += gloss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        print ("Epoch [{}/{}], Step [{}/{}] Avg Epoch Loss: {:.4f} Loss: {:.4f}".format(epoch+1, num_epochs, i+1, total_step, epoch_loss/(i+1), loss.item())+loss_msg)

    # Save weights
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        torch.save(model.state_dict(), opt['weights_path'] + "/" + MODE + "_" + str(epoch) + ".ckpt")
        if MODE == 'GAN':
            torch.save(model_d.state_dict(), opt['weights_path'] + "/" + MODE + "_D_" + str(epoch) + ".ckpt")
