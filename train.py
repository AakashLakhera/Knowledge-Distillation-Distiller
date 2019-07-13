# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 04:00:04 2019

@author: Aakash
"""

import torch
import distiller
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import time

classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def countParameters(net):
    params = 0
    for par in net.parameters():
        k = 1
        for x in par.size():
            k *= x
        params += k
    return params

def checkAccuracy(model, device, testloader):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100.0 * correct / total
    print('Accuracy of the network on the 10000 test images: %.2f %%' % (
        accuracy))
    model.train()
    return accuracy

def testModel(net, device, testloader):
    global classes
    correct = 0
    total = 0
    net.eval()
    print('Parameters:', countParameters(net))
    t1 = time.time()
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    t2 = time.time()
    print('Accuracy of the network on the 10000 test images: %.2f %%' % (
        100 * correct / total))
    print('Average Latency for 10000 test images:', (t2-t1)/10000,'seconds')
    
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    
    for i in range(10):
        print('Accuracy of %5s : %2f %%' % (
            classes[i], 100.0 * class_correct[i] / class_total[i]))

def trainModel(net, modelLocation, device, trainloader, testloader, opt, startEpoch, totalEpochs, accuracy = 0):
    
    criterion = nn.CrossEntropyLoss()
    bestAccuracy = accuracy
    bestEpoch = startEpoch
    torch.save(net.state_dict(), modelLocation)
    if opt == optim.SGD:
        scheme = 1
    else:
        scheme = 0
    for epoch in range(startEpoch, totalEpochs):  # loop over the dataset multiple times
        if scheme == 1:
            if epoch < 150:
                optimizer = opt(net.parameters(), lr=0.1, momentum = 0.9, weight_decay=5e-4)
            elif epoch < 250:
                optimizer = opt(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
            else:
                optimizer = opt(net.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
        else:
            optimizer = opt(net.parameters(), lr=0.001, weight_decay=5e-4)
        running_loss = 0.0
        
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data[0].to(device), data[1].to(device)
    
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # print statistics
            running_loss += loss.item()
            
            if i % 128 == 127:    # print every 128 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 128))
                running_loss = 0.0
        
        accuracy = checkAccuracy(net, device, testloader)
        if accuracy >= bestAccuracy:    
            torch.save(net.state_dict(), modelLocation)
            bestAccuracy = accuracy
            bestEpoch = epoch+1
        print('Best Accuracy of', bestAccuracy,'at epoch',bestEpoch)
        
    print('Finished Training Model.')
    try:
        net.load_state_dict(torch.load(modelLocation))
    except:
        pass
    
    testModel(net, device, testloader)

def trainModelKD(model, modelLocation, teacher, device, trainloader, testloader, alpha, T, opt, startEpoch, totalEpochs, accuracy = 0):
    
    criterion = nn.CrossEntropyLoss()
    dlw = distiller.DistillationLossWeights(alpha*T*T, 1-alpha, 0.0)
    kd_policy = distiller.KnowledgeDistillationPolicy(model, teacher, T, dlw)
    kd_policy.active = True
    bestAccuracy = accuracy
    bestEpoch = startEpoch
    torch.save(model.state_dict(), modelLocation)
    if opt == optim.SGD:
        scheme = 1
    else:
        scheme = 0
    for epoch in range(startEpoch, totalEpochs):  # loop over the dataset multiple times
        if scheme == 1:
            if epoch < 150:
                optimizer = opt(model.parameters(), lr=0.1, momentum = 0.9, weight_decay=5e-4)
            elif epoch < 250:
                optimizer = opt(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
            else:
                optimizer = opt(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
        else:
            optimizer = opt(model.parameters(), lr=0.001, weight_decay=5e-4)
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            output = kd_policy.forward(inputs)
            loss = criterion(output, labels)
            loss = kd_policy.before_backward_pass(model, epoch, None, None, loss, None).overall_loss        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 128 == 127:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 128))
                running_loss = 0.0
        accuracy = checkAccuracy(model, device, testloader)
        if accuracy > bestAccuracy:
            torch.save(model.state_dict(), modelLocation)
            bestAccuracy = accuracy
            bestEpoch = epoch+1
        print('Best Accuracy of', bestAccuracy,'at epoch',bestEpoch)
    
    print('Finished Training Student.')
    try:
        model.load_state_dict(torch.load(modelLocation))
    except:
        pass
    
    testModel(model, device, testloader)