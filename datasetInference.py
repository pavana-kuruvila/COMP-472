#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 21:09:41 2024

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
from cnn import SecondCNN as CNN

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

classes = ["Focused", 
           "Happy", 
           "Surpised", 
           "Neutral",
           ]


models = CNN()
models.load_state_dict(torch.load('/Users/tarakuruvila/documents/testmodelsComp472.pth.tar'))
models.eval()


def classifySingleImage(data_transforms,imagePath, classes):
    image = Image.open(imagePath)
    image = data_transforms(image)
    image = image.unsqueeze(0)
    with torch.no_grad():
        output = models(image)
    _, predicted = torch.max(output.data, 1)
        
    print(classes[predicted.item()])
    
def classifyDataset(data_transforms,datasetPath):
    dataset = datasets.ImageFolder(datasetPath, transform=data_transforms)
    inferenceLoader = DataLoader(dataset, shuffle=True, batch_size=1000)
    with torch.no_grad():
        correct = 0
        total = 0
        #gets total #for images proccesed and the number of correct procced 
        for images, labels in inferenceLoader:
            outputs = models(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        print('Test Accuracy of the model on the images: {} %'
                .format((correct / total) * 100))
    
classifySingleImage(data_transforms,"/Users/tarakuruvila/.spyder-py3/happy.jpg", classes)
classifyDataset(data_transforms,"/Users/tarakuruvila/.spyder-py3/dataset2" )
        