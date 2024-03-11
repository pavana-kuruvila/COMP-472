import torch
import numpy as np
from torchvision import transforms 
from PIL import Image
import torchvision.datasets as datasets
from torch.utils.data import DataLoader , random_split
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF

#dataset path
datasetPath = "./dataset2/"

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
trainSet, testSet = random_split(dataset, [0.8,0.2])

trainLoader = DataLoader(trainSet, shuffle=True, batch_size=32)
testLoadeer = DataLoader(testSet, shuffle=True, batch_size=32)



# ------------------ Random images display After Cleaning-------------------- #
for i in range(4):
    samples = [(image, label) for image, label in trainSet if dataset.classes[label] == dataset.classes[i]][:25]

    #displaying the transformed images in a 5x5 grid
    fig, axes = plt.subplots(5, 5, figsize=(12,8))
    for j, (image, label) in enumerate(samples):
        
        #transform images to PIL and display them in their grid spots
        pilImage = TF.to_pil_image(image)
        ax = axes[j // 5, j % 5]
        ax.imshow(pilImage)
        ax.set_title(f"Image #{j+1}")
        ax.axis("off")

    #formatting and title
    plt.subplots_adjust(wspace=-0.7, hspace=0.5)
    plt.suptitle(f"Class: {dataset.classes[i]}, After Data Cleaning", fontsize=22, fontweight="bold")
    plt.show()

# ------------------ Data Visuilization: Random images display -------------------- #
        
