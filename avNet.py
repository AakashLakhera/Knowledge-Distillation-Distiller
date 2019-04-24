# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 09:04:05 2019

@author: Aakash
"""
import torch
import vgg
import resnet
import torch.nn as nn

class AvNet(nn.Module):
    def __init__(self, model1='VGG16.pt', model2='Resnet18.pt', w1=0.28, w2=0.72):
        super(AvNet, self).__init__()
        self.net1 = vgg.VGG('VGG16')
        self.net2 = resnet.ResNet18()
        self.net1.load_state_dict(torch.load(model1))
        self.net2.load_state_dict(torch.load(model2))
        self.w1 = w1
        self.w2 = w2

    def forward(self, x):
        a1 = self.net1(x)
        a2 = self.net2(x)
        x = (self.w1*a1 + self.w2*a2)
        return x
