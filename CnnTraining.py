#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 12:03:14 2024

@author: tarakuruvila
"""

import torch
import numpy as np
from torchvision import transforms 
from PIL import Image
import torchvision.datasets as datasets
from torch.utils.data import DataLoader , random_split
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import torch.nn as nn
from cnn import SecondCNN
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

#dataset path
datasetPath = "/Users/tarakuruvila/.spyder-py3/dataset2"

#---------  Data Cleaning transformations to apply ----------#

data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomRotation(degrees=(-10, 10)),
    transforms.RandomResizedCrop(size=(224, 224), scale=(0.9, 1.1), ratio=(0.9, 1.1)),
    transforms.ColorJitter(brightness=(0.5,1.5),contrast=(0.5,1.5),saturation=(0.5,1.5),hue=(-0.1,0.1)),
    transforms.GaussianBlur(kernel_size=3, sigma=2),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Standard from ImageNet
])

#transforming the data
dataset = datasets.ImageFolder(datasetPath, transform=data_transforms)
print("Dataset classes: ", dataset.classes)
#----------------------------------------------------------------#



#Split into traing and test sets
trainSet, testSet, validationSet = random_split(dataset, [0.7,0.15,0.15])

trainLoader = DataLoader(trainSet, shuffle=True, batch_size=32)
testLoader = DataLoader(testSet, shuffle=True, batch_size=1000)
ValidationLoader = DataLoader(testSet, shuffle=True, batch_size=1000)

num_epochs = 1
num_classes = 4
learning_rate = 0.001

classes = dataset.classes



def save_checkpoint(state_dict, filename='/Users/tarakuruvila/documents/testmodelsComp472.pth.tar'):
    print("Saving Checkpoint")
    torch.save(state_dict,filename)
    

model = SecondCNN()
#loss function 
criterion = nn.CrossEntropyLoss()
#updates the weight of the model
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

total_step = len(trainLoader)
loss_list = []
acc_list = []

epochLoss=500
counter = 0
stopping_threshold =3

for epoch in range(num_epochs):
    for i, (image, label) in enumerate(trainLoader):
        
        #Forward pass
        #images is extracted and ran through cnn
        outputs= model(image)
        #loss function is given the CNN output and the actual label to calculated the loss
        loss = criterion(outputs, label)
        #loss differene is appended to list
        loss_list.append(loss.item())

        # Backprop and optimisation
        #claculate loss gradients nad optimizer updates model parameters base don gradients 
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()

        # Train accuracy
        total = label.size(0)
        _, predicted = torch.max(outputs.data, 1)
        #compare the predicted label with the real label, correct hold # of correct label for the current batch 
        correct = (predicted == label).sum().item()
        #compute batch accuracy
        acc_list.append(correct / total)
        
        print(i)

        if (i + 1) % 50 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
                (correct / total) * 100))
            
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        val_loss = 0.0
        #gets total #for images proccesed and the number of correct procced 
        for images, labels in ValidationLoader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss = criterion(outputs, labels)
            val_loss +=loss.item()*images.size(0)
            
    val_accuracy = correct / total
    avg_val_loss = val_loss / len(ValidationLoader.sampler)
    
    if avg_val_loss <= epochLoss:
        save_checkpoint(model.state_dict())
        epochLoss = avg_val_loss
        counter = 0
        
    else:
        counter += 1
        if counter >= stopping_threshold:
            print(f'Validation loss hasn\'t improved for {stopping_threshold} epochs. Stopping training.')
            break 
        
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy * 100:.2f}%')
    
        
matrixAcutal = []
matrixPredictions = []
  
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    #gets total #for images proccesed and the number of correct procced 
    for images, labels in testLoader:
        matrixAcutal.append(labels)
        
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        matrixPredictions.append(predicted)
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    print('Test Accuracy of the model on the test images: {} %'
            .format((correct / total) * 100))
    
cm = confusion_matrix(matrixAcutal, matrixPredictions)
ConfusionMatrixDisplay(cm).plot()
    
#directory = "/Users/tarakuruvila/documents/testmodelsComp472.pth"
    
#torch.save(model.state_dict(), directory)
    #look into early stopping techniques
#to save
#torch.save(modelA.state_dict(), PATH)
#torestore
#modelB = TheModelBClass(*args, **kwargs)
#smodelB.load_state_dict(torch.load(PATH), strict=False)