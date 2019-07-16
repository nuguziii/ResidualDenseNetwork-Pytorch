import os
import os.path
import random
import numpy as np
import cv2
import torch
import torch.utils.data as data

def RGB_np2Tensor(imgTar):
    ts = (2, 0, 1)
    imgTar = torch.Tensor(np.transpose(imgTar,ts).astype(float)).mul_(1.0)
    return imgTar

def augment(imgTar):
    if random.random() < 0.3:
        imgTar = imgTar[:, ::-1, :]
    if random.random() < 0.3:
        imgTar = imgTar[::-1, :, :]
    return imgTar

def getPatch(imgTar, args, scale):
    (ih, iw, c) = imgTar.shape
    (th, tw) = (scale * ih, scale * iw)
    tp = args.patchSize
    ip = tp // scale
    ix = random.randrange(0, iw - tp)
    iy = random.randrange(0, ih - tp)
    (tx, ty) = (scale * ix, scale * iy)
    #imgIn = imgIn[iy:iy + ip, ix:ix + ip, :]
    imgTar = imgTar[iy:iy + tp, ix:ix + tp, :]
    return imgTar

def getPair(imgTar, args):
    sigma = torch.randint(0, 55, size=(imgTar.size(0),), dtype=imgTar.dtype)
    noise = torch.randn(imgTar.size()).mul_(sigma / 255.0)
    imgIn = imgTar + noise
    return imgIn, imgTar

class DIV2K(data.Dataset):
    def __init__(self, args):
        self.args = args
        self.scale = args.scale
        apath = args.dataDir
        #dirHR = 'HR'
        #dirLR = 'LR'
        self.dirTar = args.dataDir
        #self.dirIn = os.path.join(apath, dirLR) #input
        #self.dirTar = os.path.join(apath, dirHR) #target
        self.fileList= os.listdir(self.dirTar)
        self.nTrain = len(self.fileList)
        
    def __getitem__(self, idx):
        scale = self.scale
        #nameIn, nameTar = self.getFileName(idx)
        nameTar = self.getFileName(idx)
        #imgIn = cv2.imread(nameIn)
        imgTar = cv2.imread(nameTar, cv2.IMREAD_GRAYSCALE)
        imgTar = np.expand_dims(imgTar, axis=2)
        if self.args.need_patch:
            #imgIn, imgTar = getPatch(imgIn, imgTar, self.args, scale)
            imgTar = getPatch(imgTar, self.args, scale)
        imgTar = augment(imgTar)
        imgTar = RGB_np2Tensor(imgTar)
        return getPair(imgTar, self.args)

    def __len__(self):
        return self.nTrain   
        
    def getFileName(self, idx):
        name = self.fileList[idx]
        nameTar = os.path.join(self.dirTar, name)
        #name = name[0:-4] + 'x3' + '.png'
        #nameIn = os.path.join(self.dirIn, name)
        #return nameIn, nameTar
        return nameTar
