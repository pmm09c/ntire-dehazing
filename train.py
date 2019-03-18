import os
import sys
import json
import torch
import torch.nn as nn
import numpy as np

# internal libraries
from models.linknet import LinkNet,FullNet
from hezhang_dataset import HeZhangDataset

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
elif MODE == 'FULL':
    model = FullNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    try:
        model.trans.load_state_dict(torch.load(sys.argv[2]))
        model.atmos.load_state_dict(torch.load(sys.argv[3]))
    except Exception as e:
        try:
            model.load_state_dict(torch.load(sys.argv[2]))
        except Exception as e:
            print("No weights. Training from scratch.")

else:
    print('MODE INCORRECT : TRANS or ATMOS or FULL')
    exit()

# Loss 
trans_loss = [x for x in opt['loss_trans'].upper()]
atmos_loss = [x for x in opt['loss_atmos'].upper()]
image_loss = [x for x in opt['loss_image'].upper()]
## TODO
## the loss should be incorporated like this
## Loss = MSE(output,target) if ('MSE' in trans_loss) else Loss
## the above will add MSE if it exists in trans_loss else do nothing


# Dataset
train_dataset = HeZhangDataset(opt)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=opt['batch_size'], 
                                           shuffle=True)
total_step = len(train_loader)

if trans_loss == 'MSE':
    trans_loss = nn.MSELoss()
if atmos_loss == 'MSE':
    atmos_loss = nn.MSELoss()
if image_loss == 'MSE':
    image_loss = nn.MSELoss()
    
best_loss = np.inf
for epoch in range(num_epochs):
    epoch_loss = 0
    for i, (haze,image,image_trans,image_atmos) in enumerate(train_loader):
        haze = haze.to(device)
        
        # copy required data to device
        if image_loss :
            image = image.to(device)
        if trans_loss :
            image_trans = image_trans.to(device)
        if atmos_loss :
            image_atmos = image_atmos.to(device)

        # compute required losses
        loss_msg = ''
        iloss = 0
        tloss = 0
        aloss = 0
        if trans_loss :
            output = model(haze)
            tloss = trans_loss(output,image_trans)
            loss_msg += ' Trans Loss : {:.4f}'.format(tloss.item())
        elif atmos_loss :
            output = model(haze)
            aloss = atmos_loss(output,image_atmos)
            loss_msg += ' Atmos Loss : {:.4f}'.format(aloss.item())
        elif image_loss :
            output,trans,atmos,dehaze = model(haze)
            iloss = image_loss(output,image)
            loss_msg += ' Image Loss : {:.4f}'.format(iloss.item())
        loss = tloss + aloss + iloss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        print ("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f} Avg Epoch Loss: {:.4f}".format(epoch+1, num_epochs, i+1, total_step, loss.item(),epoch_loss/(i+1))+loss_msg)
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        torch.save(model.state_dict(), opt['weights_path'] + "/" + MODE + "_" + str(epoch) + ".ckpt")
        
torch.save(model.state_dict(), opt['weights_path'] + "/" + MODE + "_" + str(epoch) + ".ckpt")          


            




