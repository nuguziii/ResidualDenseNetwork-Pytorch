
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np
from torch.autograd import Variable

class sub_pixel(nn.Module):
    def __init__(self, scale, act=False):
        super(sub_pixel, self).__init__()
        modules = []
        modules.append(nn.PixelShuffle(scale))
        self.body = nn.Sequential(*modules)
    def forward(self, x):
        x = self.body(x)
        return x
        
class make_dense(nn.Module):
  def __init__(self, nChannels, growthRate, kernel_size=3):
    super(make_dense, self).__init__()
    self.conv = nn.Conv2d(nChannels, growthRate, kernel_size=kernel_size, padding=(kernel_size-1)//2, bias=False)
  def forward(self, x):
    out = F.relu(self.conv(x))
    out = torch.cat((x, out), 1)
    return out

# Residual dense block (RDB) architecture
class RDB(nn.Module):
  def __init__(self, nChannels, nDenselayer, growthRate):
    super(RDB, self).__init__()
    nChannels_ = nChannels
    modules = []
    for i in range(nDenselayer):    
        modules.append(make_dense(nChannels_, growthRate))
        nChannels_ += growthRate 
    self.dense_layers = nn.Sequential(*modules)    
    self.conv_1x1 = nn.Conv2d(nChannels_, nChannels, kernel_size=1, padding=0, bias=False)
  def forward(self, x):
    out = self.dense_layers(x)
    out = self.conv_1x1(out)
    out = out + x
    return out
# Residual Dense Network
class RDN(nn.Module):
    def __init__(self, args):
        super(RDN, self).__init__()
        nChannel = args.nChannel
        nDenselayer = args.nDenselayer
        nFeat = args.nFeat
        scale = args.scale
        growthRate = args.growthRate
        self.args = args

        # F-1
        self.conv1 = nn.Conv2d(nChannel, nFeat, kernel_size=3, padding=1, bias=True)
        # F0
        self.conv2 = nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=True)
        # RDBs 3
        self.RDB1 = RDB(nFeat, nDenselayer, growthRate)
        self.RDB2 = RDB(nFeat, nDenselayer, growthRate)
        self.RDB3 = RDB(nFeat, nDenselayer, growthRate)
        self.RDB4 = RDB(nFeat, nDenselayer, growthRate)
        self.RDB5 = RDB(nFeat, nDenselayer, growthRate)
        self.RDB6 =RDB(nFeat, nDenselayer, growthRate)
        self.RDB7 =RDB(nFeat, nDenselayer, growthRate)
        self.RDB8 =RDB(nFeat, nDenselayer, growthRate)
        self.RDB9 =RDB(nFeat, nDenselayer, growthRate)
        self.RDB10 =RDB(nFeat, nDenselayer, growthRate)
        self.RDB11 =RDB(nFeat, nDenselayer, growthRate)
        self.RDB12 =RDB(nFeat, nDenselayer, growthRate)
        self.RDB13 =RDB(nFeat, nDenselayer, growthRate)
        self.RDB14 =RDB(nFeat, nDenselayer, growthRate)
        self.RDB15 =RDB(nFeat, nDenselayer, growthRate)
        self.RDB16 =RDB(nFeat, nDenselayer, growthRate)
        # global feature fusion (GFF)
        self.GFF_1x1 = nn.Conv2d(nFeat*16, nFeat, kernel_size=1, padding=0, bias=True)
        self.GFF_3x3 = nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=True)
        # Upsampler
        self.conv_up = nn.Conv2d(nFeat, nFeat*scale*scale, kernel_size=3, padding=1, bias=True)
        self.upsample = sub_pixel(scale)
        # conv 
        self.conv3 = nn.Conv2d(nFeat, nChannel, kernel_size=3, padding=1, bias=True)
    def forward(self, x):

        F_  = self.conv1(x)
        F_0 = self.conv2(F_)
        F_1 = self.RDB1(F_0)
        F_2 = self.RDB2(F_1)
        F_3 = self.RDB3(F_2)
        F_4 = self.RDB4(F_3)
        F_5 = self.RDB5(F_4)
        F_6 = self.RDB6(F_5)
        F_7 = self.RDB7(F_6)
        F_8 = self.RDB8(F_7)
        F_9 = self.RDB9(F_8)
        F_10 = self.RDB10(F_9)
        F_11 = self.RDB11(F_10)
        F_12 = self.RDB12(F_11)
        F_13 = self.RDB13(F_12)
        F_14 = self.RDB14(F_13)
        F_15 = self.RDB15(F_14)
        F_16 = self.RDB16(F_15)
        FF = torch.cat((F_1, F_2, F_3, F_4, F_5, F_6, F_7, F_8, F_9, F_10, F_11, F_12, F_13, F_14, F_15, F_16), 1)
        FdLF = self.GFF_1x1(FF)         
        FGF = self.GFF_3x3(FdLF)
        FDF = FGF + F_
        #us = self.conv_up(FDF)
        #us = self.upsample(us)
        us = self.conv3(FDF)
        output = us + x

        return output
