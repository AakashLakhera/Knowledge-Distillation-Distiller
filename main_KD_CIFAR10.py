# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 14:58:41 2019

@author: Aakash
"""

import sys
import resnet
import vgg
import cnn5
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import train
import avNet
import netJoin

def get_platform():
    platforms = {
        'linux1' : 'Linux',
        'linux2' : 'Linux',
        'darwin' : 'OS X',
        'win32' : 'Windows'
    }
    if sys.platform not in platforms:
        return sys.platform
    
    return platforms[sys.platform]

resDict = {'Resnet18': resnet.ResNet18, 
                'Resnet34': resnet.ResNet34,
                'Resnet50': resnet.ResNet50,
                'Resnet101': resnet.ResNet101,
                'Resnet152': resnet.ResNet152}


models = ['Resnet18', 'Resnet34', 'Resnet50', 'Resnet101', 'Resnet152', 
              'VGG6AS','VGG6AM','VGG6A','VGG6','VGG7','VGG8','VGG11','VGG13','VGG16','VGG16A','VGG19', '5-CNN']


def createResnet(ResType):
    global resDict    
    f = resDict[ResType]
    return f()

def createVGG(VGGType):
    return vgg.VGG(VGGType)

def create5_CNN():
    return cnn5.CNN_5()

def createNet(modelType):
    global models
    global jointModels
    
    for i in range(len(models)):
        if modelType.lower() == models[i].lower():
            if i <= 4:
                return createResnet(models[i])
            elif i <= len(models)-2:
                return createVGG(models[i])
            else:
                return create5_CNN()
    
    if modelType.lower() == 'avg':
        return avNet.AvNet()
    
    if modelType.lower() == 'jn':
        return netJoin.jointNet()
    
    return None

def main():
    
    global models
    
    trainModel = train.trainModel
    trainModelKD = train.trainModelKD
    testModel = train.testModel
    
    if(len(sys.argv) not in [5, 11, 15]):
        print('Not enough arguments given. Aborting.')
        return
    
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("CUDA Device Detected. GPU will be used.")
    else:
        device = torch.device("cpu")
        print("No CUDA supported GPU detected. CPU will be used.")
    
    print("Dataset- CIFAR-10")
    
    if get_platform() == 'Windows':
        workers = 0
    else:
        workers = 2
    
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    try:
        testset = torchvision.datasets.CIFAR10(root=sys.argv[1], train=False,
                                           download=False, transform=transform)
    except:
        testset = torchvision.datasets.CIFAR10(root=sys.argv[1], train=False,
                                           download=True, transform=transform)
        
    testloader = torch.utils.data.DataLoader(testset, batch_size=1024,
                                             shuffle=False, num_workers=workers)
    
    if sys.argv[2].lower() == 'tst':
       model = createNet(sys.argv[3])
       if model is None:
           print('Invalid Model Type. Aborting.')
           return
       
       model = model.to(device)
       
       try:
           model.load_state_dict(torch.load(sys.argv[4]))
       except Exception as e:
           print(e)
           print('Model could not be loaded. Aborting.')
           return
       testModel(model, device, testloader)
       return
   
    elif sys.argv[2].lower() != 'trn':
        print('Invalid Operation. Aborting.')
        return
    
    if sys.argv[3].lower() == 'y':
        transform_train =  transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            #transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])
    
    elif sys.argv[3].lower() == 'n':
        transform_train = transform
    
    else:
        print('Invalid argument for specifying augmentation. Aborting.')
        return
    
    if sys.argv[4].lower() == 'adam':
        opt = optim.Adam
    elif sys.argv[4].lower() == 'sgd':
        opt = optim.SGD
    else:
        print('Invalid argument for specifying optimizer. Aborting.')
        return
    
    try:
        totalEpochs = int(sys.argv[5])
    except:
        print('Invalid argument for specifying total epochs. Aborting.')
        return
    
    try:
        startEpoch = int(sys.argv[6])
    except:
        print('Invalid argument for specifying starting epoch. Aborting.')
        return
    
    try:
        batchSize = int(sys.argv[7])
    except:
        print('Invalid argument for specifying mini-batch size. Aborting.')
        return
    
    trainset = torchvision.datasets.CIFAR10(root=sys.argv[1], train=True,
                                            download=False, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchSize,
                                              shuffle=True, num_workers=workers)
    
    
    if sys.argv[8].lower() == 'y': # Use distillation
        try:
            alpha = float(sys.argv[9])
            T = float(sys.argv[10])
        except:
            print('Invalid Arguments for temperature or alpha. Aborting')
            return
        model = createNet(sys.argv[11])
        if model is None:
            print('Invalid Model Type. Aborting.')
            return
        model = model.to(device)    
        try:
            model.load_state_dict(torch.load(sys.argv[12]))
            accuracy = train.checkAccuracy(model, device, testloader)
        except:
            accuracy = 0
            
        teacher = createNet(sys.argv[13])
        if teacher is None:
            print('Invalid Model Type for Teacher. Aborting.')
            return
        teacher = teacher.to(device)
        try:
            teacher.load_state_dict(torch.load(sys.argv[14]))
        except Exception as e:
            print(e)
            print('The Teacher model does not exists. Aborting.')
            return
        for param in teacher.parameters():
            param.requires_grad = False
        teacher.eval()
        trainModelKD(model, sys.argv[12], teacher, device, trainloader, testloader, alpha, T, opt, startEpoch, totalEpochs, accuracy)
        return
    elif sys.argv[8].lower() == 'n':
        model = createNet(sys.argv[9])
        if model is None:
            print('Invalid Model Type. Aborting.')
            return
        model = model.to(device)
        try:
            model.load_state_dict(torch.load(sys.argv[10]))
            accuracy = train.checkAccuracy(model, device, testloader)
        except:
            accuracy = 0
            
        trainModel(model, sys.argv[10], device, trainloader, testloader, opt, startEpoch, totalEpochs, accuracy)
        return
    else:
        print('Invalid argument for specifying Knowledge Distillation. Aborting.')
        return
        
if  __name__ == '__main__':
    main()