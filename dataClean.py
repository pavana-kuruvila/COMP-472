import torch
import os
from torchvision import transforms 
from PIL import Image
import torchvision.datasets as datasets
from torch.utils.data import DataLoader , random_split
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF

datasetPath = "/Users/vanakuruvila/Documents/School/COMP 472/DataSet copy"
dataset = datasets.ImageFolder(datasetPath)

"""
def getMeanStd(loader):
    # Compute the mean and standard deviation of all pixels in the dataset
    num_pixels = 0
    mean = 0.0
    std = 0.0
    for images, _ in loader:
        #images = transforms.ToTensor()(images)
        batch_size, num_channels, height, width = images.shape
        num_pixels += batch_size * height * width
        mean += images.mean(axis=(0, 2, 3)).sum()
        std += images.std(axis=(0, 2, 3)).sum()

    mean /= num_pixels
    std /= num_pixels

    return mean, std


batch_size = 32

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

dataset = datasets.ImageFolder(datasetPath, transform=transform)

loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
mean, std = getMeanStd(loader)

print(mean , std)

"""
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    #transforms.Normalize(mean=mean, std=std)
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Standard from ImageNet
])

dataset = datasets.ImageFolder(datasetPath, transform=data_transforms)
print("Dataset classes: ", dataset.classes)

trainSet, testset = random_split(dataset, [0.8,0.2])

trainLoader = DataLoader(trainSet, shuffle=True, batch_size=32)
testLoadeer = DataLoader(testset, shuffle=True, batch_size=32)

samples = [trainSet[i] for i in range(10)]

fig, axes = plt.subplots(2, 5, figsize=(12,6))
for i, (image, label) in enumerate(samples):
    ax = axes[i // 5, i % 5]
    ax.imshow(TF.to_pil_image(image))
    ax.set_title(f"Class: {dataset.classes[label]}")
    ax.axis("off")

plt.show()

