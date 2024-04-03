#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 19:11:02 2024

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

class SecondCNN(nn.Module):

    def __init__(self):
        super(SecondCNN, self).__init__()
        self.conv_layer = nn.Sequential(

            #we currently have 4 convulutional layers, number of filters increase throught the layers so the CNN can learn higher level features
            #we could add another layer with more channels to deepen the feature recognitions, but too many layer can lead to over fitting

            #out channels is number of filters and kernel size is the filter size so 3x3 and padding is 1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            #normalization
            nn.BatchNorm2d(32),
            #allows a small, non-zero gradient when the unit is not active helps with vanishing gradient problem.
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            #pooling window is 2x2 and stride means it moves by 2 pixels at a time
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
        )

        self.fc_layer = nn.Sequential(
            #randomly sets input units to zero to prevent overfitting
            nn.Dropout(p=0.1),
            nn.Linear( 54 * 54 * 64, 1000),
            nn.LeakyReLU(inplace=True),
            nn.Linear(1000, 512),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512, 10)
        )
    
    def forward(self, x):
            # performs the convolutional layers
            x = self.conv_layer(x)
            # flatten to 2d 
            x = x.view(x.size(0), -1)
            # fully connected layer to performs classification base don the feautres that model extracted
            x = self.fc_layer(x)
            return x