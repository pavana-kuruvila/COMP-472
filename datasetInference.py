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
from cnnVarient1 import SecondCNN as CNNV1
from cnnVarient2 import SecondCNN as CNNV2


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

#----------------------------------------------------------------#

#Defining the classes
classes = ["Focused", 
           "Happy", 
           "Neutral", 
           "Surprised",
           ]



#to test variants 1 or 2 replace the line with either of these
#models = CNNV1()
#models = CNNV2()

models = CNN()

#Loading the saved model
#change the path to the model you would like to test. make sure the model above matches the varient 
#options for models to load are testmodelsMain.pth.tar, variant2.pth.tar, variant1.pth.tar
models.load_state_dict(torch.load('./testmodelsMain.pth.tar'))
models.eval()


def classifySingleImage(data_transforms,imagePath, classes):
    #opening the image
    image = Image.open(imagePath)
    #transforming it
    image = data_transforms(image)
    image = image.unsqueeze(0)
    with torch.no_grad():
        #getting the prediction
        output = models(image)
    _, predicted = torch.max(output.data, 1)
        
    print(classes[predicted.item()])
    
def classifyDataset(data_transforms,datasetPath):
    #transforming the dataset
    dataset = datasets.ImageFolder(datasetPath, transform=data_transforms)
    inferenceLoader = DataLoader(dataset, shuffle=True, batch_size=32)
    with torch.no_grad():
        correct = 0
        total = 0
        #gets total #for images proccesed and the number of correct procced 
        for images, labels in inferenceLoader:
            #getting the predictions
            outputs = models(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        print('Test Accuracy of the model on the images: {} %'
                .format((correct / total) * 100))

#replace the path with the name of the image you would like to test. Options included are testHappyImage.jpeg, testSurpriseImage.jpg, testFocusedImage.jpg
classifySingleImage(data_transforms,"./testHappyImage.jpeg", classes)
#replace the path with the data set you would like to test on.
classifyDataset(data_transforms,"./dataset2/" )
