# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 09:20:26 2019

@author: Aakash
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import resnet
import vgg

class jointNet(nn.Module): # Joint train a resnet and vgg
    def __init__(self, lastLayers1=1, lastLayers2=1, firstLayers1=0, firstLayers2=0, net1 = 'VGG16.pt', net2 = 'Resnet18.pt', wts = (0.28, 0.72)):
        super(jointNet, self).__init__()
        self.net1 = vgg.VGG('VGG16')
        self.net2 = resnet.ResNet18()
        self.alpha = wts[0]
        self.beta = wts[1]
        if net1 is not None:
            self.net1.load_state_dict(torch.load(net1))
        if net2 is not None:
            self.net2.load_state_dict(torch.load(net2))
        count1 = 0
        count2 = 0
        for x in self.net1.modules():
            if type(x) in (nn.Conv2d, nn.Linear):
                count1 += 1
        
        for x in self.net2.modules():
            if type(x) in (resnet.BasicBlock, nn.Linear):
                count2 += 1
        for param in self.net1.parameters():
            param.requires_grad = False
        for param in self.net2.parameters():
            param.requires_grad = False
    
        i = 1
        for x in self.net1.modules():
            if type(x) in (nn.Conv2d, nn.Linear):
                if i <= firstLayers1:
                    for param in x.parameters():
                        param.requires_grad = True    
                elif i > count1-lastLayers1:
                    for param in x.parameters():
                        param.requires_grad = True
            i += 1
        
        i=1
        for x in self.net2.modules():
            if type(x) in (resnet.BasicBlock, nn.Linear):
                if i <= firstLayers2:
                    for param in x.parameters():
                        param.requires_grad = True    
                elif i > count2-lastLayers2:
                    for param in x.parameters():
                        param.requires_grad = True
            i += 1
        self.fc1 = nn.Linear(10, 10)
        self.fc2 = nn.Linear(10, 10) # Extra layers, not used
        self.fc3 = nn.Linear(10, 10) # Extra layers, not used
        self.z1 = None
        self.z2 = None
        self.z3 = None
    
    def forward(self, x):
        #a1 = nn.functional.softmax(self.net1(x), dim=1)
        #a2 = nn.functional.softmax(self.net2(x), dim=1)
        a1 = self.net1(x)
        a2 = self.net2(x)
        x = self.alpha*a1 + self.beta*a2
        x = nn.functional.softmax(x, dim=1)
        self.z1 = self.fc1(x)
        self.z2 = a1
        self.z3 = a2
        #x = self.fc2(nn.functional.softmax(x1, dim=1) + nn.functional.softmax(x, dim=1))
        return self.z1
    
    def loss_fn(self, labels):
        c1 = nn.CrossEntropyLoss()
        c2 = nn.CrossEntropyLoss()
        c3 = nn.CrossEntropyLoss()
        l1 = c1(self.z1, labels)
        l2 = c2(self.z2, labels)
        l3 = c3(self.z3, labels)
        return l1+l2+l3