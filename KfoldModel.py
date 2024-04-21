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
from torch.utils.data import DataLoader , random_split, SubsetRandomSampler
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import torch.nn as nn
from cnn import SecondCNN
from cnnVarient1 import SecondCNN as CNNV1
from cnnVarient2 import SecondCNN as CNNV2
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import KFold

#dataset path
datasetPath = "./dataset2/"


#if you would like to load a model and continue working it change the path 
loadModel = False

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

kFoldsNum = 10
kFold = KFold(n_splits =kFoldsNum, shuffle =True)
results = {}

num_epochs = 10
num_classes = 4
classes = dataset.classes

#loss function 
criterion = nn.CrossEntropyLoss()

for fold, (train, test) in enumerate(kFold.split(dataset)):

    #replace with path wanted 
    checkpoint_path = 'fold_{}_model.pth'.format(fold)
    if loadModel:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # Optionally, resume training from the saved fold
        fold = checkpoint['fold']


     # Print
    print(f'FOLD {fold}')
    print('--------------------------------')

    train_subsampler = torch.utils.data.SubsetRandomSampler(train)
    test_subsampler = torch.utils.data.SubsetRandomSampler(test)

    trainloader = torch.utils.data.DataLoader(
                      dataset, 
                      batch_size=32, sampler=train_subsampler)
    testloader = torch.utils.data.DataLoader(
                      dataset,
                      batch_size=32, sampler=test_subsampler)
    

    #chnage the path to save the models to a different file
    def save_checkpoint(state_dict, filename='./testmodel.pth.tar'):
        print("Saving Checkpoint")
        torch.save(state_dict,filename)
        

    model = SecondCNN()
    #updates the weight of the model
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    total_step = len(trainloader)
    loss_list = []
    acc_list = []

    for epoch in range(num_epochs):
        # Print epoch
        print(f'Starting epoch {epoch+1}')
        for i, (image, label) in enumerate(trainloader):
            
            #Forward pass
            #images is extracted and ran through cnn
            outputs= model(image)
            #loss function is given the CNN output and the actual label to calculated the loss
            loss = criterion(outputs, label)
            #loss differene is appended to list
            loss_list.append(loss.item())

            # Backprop and optimisation
            #claculate loss gradients nad optimizer updates model parameters based on gradients 
            optimizer.zero_grad()
            loss.backward()
            #gradient clipping
            nn.utils.clip_grad_norm_(model.parameters(), 0.75)
            optimizer.step()

            # Train accuracy
            total = label.size(0)
            _, predicted = torch.max(outputs.data, 1)
            #compare the predicted label with the real label, correct hold # of correct label for the current batch 
            correct = (predicted == label).sum().item()
            #compute batch accuracy
            acc_list.append(correct / total)
            
            print(i)

            if (i + 1) % 30 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                    .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
                    (correct / total) * 100))
            
            if (i + 1) % 50 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                    .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
                    (correct / total) * 100))
                

    #final eval        
    matrixAcutal = []
    matrixPredictions = []
    
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        #gets total #for images proccesed and the number of correct procced 
        for images, labels in testloader:
            matrixAcutal.append(labels)
            
            #gets the predictions
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            matrixPredictions.append(predicted)
            
            #calculates the accuracy
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        print('Test Accuracy of the fold on the test images: {} %'
                .format((correct / total) * 100))
        print('--------------------------------')
        results[fold] = 100.0 * (correct / total)

     # Flatten the lists of tensors
    matrix_actual_flat = np.concatenate([labels.numpy().flatten() for labels in matrixAcutal])
    matrix_predictions_flat = np.concatenate([predictions.numpy().flatten() for predictions in matrixPredictions])

    # Compute confusion matrix
    cm = confusion_matrix(matrix_actual_flat, matrix_predictions_flat)

    # Define class names
    classes = ['Focused', 'Happy', 'Neutral', 'Surprised']

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    ConfusionMatrixDisplay(cm, display_labels=classes).plot(cmap='Blues')
    plt.title(f'Confusion Matrix FOLD {fold}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()    

    torch.save({
    'fold': fold,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    # Add any other relevant information
    }, 'fold_{}_model.pth'.format(fold))

  # Print fold results
print(f'K-FOLD CROSS VALIDATION RESULTS FOR {kFoldsNum} FOLDS')
print('--------------------------------')
sum = 0.0
for key, value in results.items():
    print(f'Fold {key}: {value} %')
    sum += value
print(f'Average: {sum/len(results.items())} %')


