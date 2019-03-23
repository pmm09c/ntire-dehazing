'''
References :
 Linknet from e-lab
     https://github.com/e-lab/pytorch-linknet
 Refinement layers from He,Zhang
     https://github.com/hezhangsprinter/DCPDN
 Discrimiantor is taken from Aitor Ruano's implementation of SRGAN
     https://github.com/aitorzip/PyTorch-SRGAN
     https://arxiv.org/abs/1609.04802
     https://arxiv.org/abs/1710.05941
'''
import torch
import torch.nn as nn

from torch.nn import functional as F
from torchvision.models import resnet, vgg16

from collections import namedtuple

class ContentLoss(nn.Module):
    def __init__(self, device):
        super(ContentLoss, self).__init__()
        self.vgg = Vgg16(requires_grad=False).to(device)
        self.loss = nn.MSELoss()
    def forward(self, output, target, weight=1):
        f_output = self.vgg(output).relu2_2
        f_target = self.vgg(target).relu2_2
        return weight * self.loss(f_output, f_target)

class Vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg16, self).__init__()
        vgg_pretrained_features = vgg16(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
        return out

class Decoder(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=False):
        # TODO bias=True
        super(Decoder, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_planes, in_planes//4, 1, 1, 0, bias=bias),
                                nn.BatchNorm2d(in_planes//4),
                                nn.ReLU(inplace=True),)
        self.tp_conv = nn.Sequential(nn.ConvTranspose2d(in_planes//4, in_planes//4, kernel_size, stride, padding, output_padding, bias=bias),
                                nn.BatchNorm2d(in_planes//4),
                                nn.ReLU(inplace=True),)
        self.conv2 = nn.Sequential(nn.Conv2d(in_planes//4, out_planes, 1, 1, 0, bias=bias),
                                nn.BatchNorm2d(out_planes),
                                nn.ReLU(inplace=True),)

    def forward(self, x):
        x = self.conv1(x)
        x = self.tp_conv(x)
        x = self.conv2(x)

        return x

class LinkNet50(nn.Module):
    def __init__(self, n_classes=3,pad=(0,0,0,0)):
        """
        Model initialization
        :param x_n: number of input neurons
        :type x_n: int
        """
        super(LinkNet50, self).__init__()

        base = resnet.resnet50(pretrained=True)

        self.in_block = nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu,
            base.maxpool
        )
        self.pad = nn.ReflectionPad2d(pad)
        self.pad_size = [int(x/2) for x in pad]
        self.encoder1 = base.layer1
        self.encoder2 = base.layer2
        self.encoder3 = base.layer3
        self.encoder4 = base.layer4

        self.decoder1 = Decoder(256, 64, 3, 1, 1, 0)
        self.decoder2 = Decoder(512, 256, 3, 2, 1, 1)
        self.decoder3 = Decoder(1024, 512, 3, 2, 1, 1)
        self.decoder4 = Decoder(2048, 1024, 3, 2, 1, 1)

        self.tp_conv1 = nn.Sequential(nn.ConvTranspose2d(64, 32, 3, 2, 1, 1),
                                      nn.BatchNorm2d(32),
                                      nn.ReLU(inplace=True),)
        self.conv2 = nn.Sequential(nn.Conv2d(32, 32, 3, 1, 1),
                                   nn.BatchNorm2d(32),
                                   nn.ReLU(inplace=True),)
        self.tp_conv2 = nn.ConvTranspose2d(32, n_classes, 2, 2, 0)
        self.lsm = nn.ReLU() #nn.LogSoftmax(dim=1)

    def forward(self, x):
        # Initial block
        x = self.pad(x)
        x = self.in_block(x)

        # Encoder blocks
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Decoder blocks
        d4 = e3 + self.decoder4(e4)
        d3 = e2 + self.decoder3(d4)
        d2 = e1 + self.decoder2(d3)
        d1 = x + self.decoder1(d2)
        y = self.tp_conv1(d1)
        y = self.conv2(y)
        y = self.tp_conv2(y)

        y = self.lsm(y) #relu
        y = y[:,
              :,
              self.pad_size[2]:(y.shape[2]-self.pad_size[2]),
              self.pad_size[3]:(y.shape[3]-self.pad_size[3])]
        return y  
    
class LinkNet(nn.Module):
    """
    Generate Model Architecture
    """

    def __init__(self, n_classes=3,pad=(0,0,0,0)):
        """
        Model initialization
        :param x_n: number of input neurons
        :type x_n: int
        """
        super(LinkNet, self).__init__()

        base = resnet.resnet18(pretrained=True)

        self.in_block = nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu,
            base.maxpool
        )
        self.pad = nn.ReflectionPad2d(pad)
        self.pad_size = [int(x/2) for x in pad]
        self.encoder1 = base.layer1
        self.encoder2 = base.layer2
        self.encoder3 = base.layer3
        self.encoder4 = base.layer4

        self.decoder1 = Decoder(64, 64, 3, 1, 1, 0)
        self.decoder2 = Decoder(128, 64, 3, 2, 1, 1)
        self.decoder3 = Decoder(256, 128, 3, 2, 1, 1)
        self.decoder4 = Decoder(512, 256, 3, 2, 1, 1)

        self.tp_conv1 = nn.Sequential(nn.ConvTranspose2d(64, 32, 3, 2, 1, 1),
                                      nn.BatchNorm2d(32),
                                      nn.ReLU(inplace=True),)
        self.conv2 = nn.Sequential(nn.Conv2d(32, 32, 3, 1, 1),
                                nn.BatchNorm2d(32),
                                nn.ReLU(inplace=True),)
        self.tp_conv2 = nn.ConvTranspose2d(32, n_classes, 2, 2, 0)
        self.lsm = nn.ReLU()

    def forward(self, x):
        # Initial block
        x = self.pad(x)
        x = self.in_block(x)

        # Encoder blocks
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Decoder blocks
        d4 = e3 + self.decoder4(e4)
        d3 = e2 + self.decoder3(d4)
        d2 = e1 + self.decoder2(d3)
        d1 = x + self.decoder1(d2)

        y = self.tp_conv1(d1)
        y = self.conv2(y)
        y = self.tp_conv2(y)

        y = self.lsm(y) #relu
        y = y[:,
              :,
              self.pad_size[2]:(y.shape[2]-self.pad_size[2]),
              self.pad_size[3]:(y.shape[3]-self.pad_size[3])]
        return y

    
class DualFastNet(nn.Module):
    
    def __init__(self):
        
        super(DualFastNet, self).__init__()
        self.trans = LinkNet()
        self.atmos = LinkNet()        
        self.tanh=nn.Tanh()
        self.refine1= nn.Conv2d(6, 20, kernel_size=3,stride=1,padding=1)
        self.refine2= nn.Conv2d(20, 20, kernel_size=3,stride=1,padding=1)
        self.threshold=nn.Threshold(0.1, 0.1)
        self.conv1010 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1020 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1030 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1040 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.refine3= nn.Conv2d(20+4, 3, kernel_size=3,stride=1,padding=1)
        self.upsample = F.upsample_nearest
        self.relu1=nn.LeakyReLU(0.2)
        self.relu2=nn.LeakyReLU(0.2)
        self.relu3=nn.LeakyReLU(0.2)
        self.relu4=nn.LeakyReLU(0.2)
        self.relu5=nn.LeakyReLU(0.2)
        self.relu6=nn.LeakyReLU(0.2)
        
    def forward(self,I):
        t = torch.clamp(self.trans(I),min=.01,max=0.99)
        A = torch.clamp(self.atmos(I),min=.01,max=0.99)
        J = torch.add(I,-A)
        J = torch.div(J,t)
        J = torch.clamp(J,min=0,max=1)
        J = torch.add(J,A)
        # Adapted from He Zhang https://github.com/hezhangsprinter/DCPDN
        dehaze=torch.cat([J,I],1)
        dehaze=self.relu1((self.refine1(dehaze)))
        dehaze=self.relu2((self.refine2(dehaze)))
        shape_out = dehaze.data.size()
        shape_out = shape_out[2:4]
        x101 = F.avg_pool2d(dehaze, 32)
        x1010 = F.avg_pool2d(dehaze, 32)
        x102 = F.avg_pool2d(dehaze, 16)
        x1020 = F.avg_pool2d(dehaze, 16)
        x103 = F.avg_pool2d(dehaze, 8)
        x104 = F.avg_pool2d(dehaze, 4)
        x1010 = self.upsample(self.relu3(self.conv1010(x101)),size=shape_out)
        x1020 = self.upsample(self.relu4(self.conv1020(x102)),size=shape_out)
        x1030 = self.upsample(self.relu5(self.conv1030(x103)),size=shape_out)
        x1040 = self.upsample(self.relu6(self.conv1040(x104)),size=shape_out)
        dehaze = torch.cat((x1010, x1020, x1030, x1040, dehaze), 1)
        dehaze= self.tanh(self.refine3(dehaze))
        return dehaze,t,A,J

class FastNet(nn.Module):
    
    def __init__(self):
        
        super(FastNet, self).__init__()
        self.trans = LinkNet(n_classes=32)
        self.tanh=nn.Tanh()
        self.refine0= nn.Conv2d(3, 32, kernel_size=3,stride=1,padding=1)
        self.refine1= nn.Conv2d(64, 20, kernel_size=3,stride=1,padding=1)
        self.refine2= nn.Conv2d(20, 20, kernel_size=3,stride=1,padding=1)
        self.threshold=nn.Threshold(0.1, 0.1)
        self.conv1010 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1020 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1030 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1040 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.refine3= nn.Conv2d(20+4, 3, kernel_size=3,stride=1,padding=1)
        self.upsample = F.upsample_nearest
        self.relu0=nn.LeakyReLU(0.2)
        self.relu1=nn.LeakyReLU(0.2)
        self.relu2=nn.LeakyReLU(0.2)
        self.relu3=nn.LeakyReLU(0.2)
        self.relu4=nn.LeakyReLU(0.2)
        self.relu5=nn.LeakyReLU(0.2)
        self.relu6=nn.LeakyReLU(0.2)
        
    def forward(self,I):
        t = self.trans(I)
        # Adapted from He Zhang https://github.com/hezhangsprinter/DCPDN
        # Bring I to feature space for concatenation
        I = self.relu0((self.refine0(I)))
        dehaze=torch.cat([t,I],1)
        dehaze=self.relu1((self.refine1(dehaze)))
        dehaze=self.relu2((self.refine2(dehaze)))
        shape_out = dehaze.data.size()
        shape_out = shape_out[2:4]
        x101 = F.avg_pool2d(dehaze, 32)
        x1010 = F.avg_pool2d(dehaze, 32)
        x102 = F.avg_pool2d(dehaze, 16)
        x1020 = F.avg_pool2d(dehaze, 16)
        x103 = F.avg_pool2d(dehaze, 8)
        x104 = F.avg_pool2d(dehaze, 4)
        x1010 = self.upsample(self.relu3(self.conv1010(x101)),size=shape_out)
        x1020 = self.upsample(self.relu4(self.conv1020(x102)),size=shape_out)
        x1030 = self.upsample(self.relu5(self.conv1030(x103)),size=shape_out)
        x1040 = self.upsample(self.relu6(self.conv1040(x104)),size=shape_out)
        dehaze = torch.cat((x1010, x1020, x1030, x1040, dehaze), 1)
        dehaze= self.tanh(self.refine3(dehaze))
        return dehaze,t
    
class FastNet50(nn.Module):
    
    def __init__(self):
        
        super(FastNet50, self).__init__()
        self.trans = LinkNet50(n_classes=32)
        self.tanh=nn.Tanh()
        self.refine0= nn.Conv2d(3, 32, kernel_size=3,stride=1,padding=1)
        self.refine1= nn.Conv2d(64, 20, kernel_size=3,stride=1,padding=1)
        self.refine2= nn.Conv2d(20, 20, kernel_size=3,stride=1,padding=1)
        self.threshold=nn.Threshold(0.1, 0.1)
        self.conv1010 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1020 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1030 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1040 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.refine3= nn.Conv2d(20+4, 3, kernel_size=3,stride=1,padding=1)
        self.upsample = F.upsample_nearest
        self.relu0=nn.LeakyReLU(0.2)
        self.relu1=nn.LeakyReLU(0.2)
        self.relu2=nn.LeakyReLU(0.2)
        self.relu3=nn.LeakyReLU(0.2)
        self.relu4=nn.LeakyReLU(0.2)
        self.relu5=nn.LeakyReLU(0.2)
        self.relu6=nn.LeakyReLU(0.2)
        
    def forward(self,I):
        t = self.trans(I)
        # Adapted from He Zhang https://github.com/hezhangsprinter/DCPDN
        # Bring I to feature space for concatenation
        I = self.relu0((self.refine0(I)))
        dehaze=torch.cat([t,I],1)
        dehaze=self.relu1((self.refine1(dehaze)))
        dehaze=self.relu2((self.refine2(dehaze)))
        shape_out = dehaze.data.size()
        shape_out = shape_out[2:4]
        x101 = F.avg_pool2d(dehaze, 32)
        x1010 = F.avg_pool2d(dehaze, 32)
        x102 = F.avg_pool2d(dehaze, 16)
        x1020 = F.avg_pool2d(dehaze, 16)
        x103 = F.avg_pool2d(dehaze, 8)
        x104 = F.avg_pool2d(dehaze, 4)
        x1010 = self.upsample(self.relu3(self.conv1010(x101)),size=shape_out)
        x1020 = self.upsample(self.relu4(self.conv1020(x102)),size=shape_out)
        x1030 = self.upsample(self.relu5(self.conv1030(x103)),size=shape_out)
        x1040 = self.upsample(self.relu6(self.conv1040(x104)),size=shape_out)
        dehaze = torch.cat((x1010, x1020, x1030, x1040, dehaze), 1)
        dehaze= self.tanh(self.refine3(dehaze))
        return dehaze,t
    
def swish(x):
    return x * F.sigmoid(x)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1)        
        self.conv2 = nn.Conv2d(64, 64, 3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, 3, stride=2, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.conv7 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
        self.bn7 = nn.BatchNorm2d(512)
        self.conv8 = nn.Conv2d(512, 512, 3, stride=2, padding=1)
        self.bn8 = nn.BatchNorm2d(512)
        
        # Replaced original paper FC layers with FCN
        self.conv9 = nn.Conv2d(512, 1, 1, stride=1, padding=1)

    def forward(self, x):
        x = swish(self.conv1(x))

        x = swish(self.bn2(self.conv2(x)))
        x = swish(self.bn3(self.conv3(x)))
        x = swish(self.bn4(self.conv4(x)))
        x = swish(self.bn5(self.conv5(x)))
        x = swish(self.bn6(self.conv6(x)))
        x = swish(self.bn7(self.conv7(x)))
        x = swish(self.bn8(self.conv8(x)))

        x = self.conv9(x)
        return F.sigmoid(F.avg_pool2d(x, x.size()[2:])).view(x.size()[0], -1)
