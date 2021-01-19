import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

import numpy as np
import torchvision
import torchvision.transforms as transforms

dataset_dir ='C:/Users/jonghyun/Desktop/use_mini/train' 



class my_can(nn.Module):
    def __init__(self):
        super(my_can,self).__init__()
        #사용할 함수들 정의할 장소

    def forward(self,x):
        #함수들 사용해 forward 정의할 장소
        return x




img = 0
for n, (img, labels) in enumerate(trainloader):
    print(n, img.shape, labels.shape)
    imgs = img
    break

my_net = my_can()
out = my_can(Variable(imgs))
print(out.shape)

########################################loss 랑 backprop 없음

