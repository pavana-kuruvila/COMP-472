import torch
import numpy as np
from torchvision import transforms 
from PIL import Image
import torchvision.datasets as datasets
from torch.utils.data import DataLoader , random_split
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF

datasetPath = "/Users/vanakuruvila/Documents/School/COMP 472/DataSet2"

#Data Cleaning 
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomVerticalFlip(0.2),
    transforms.RandomRotation(degrees=(-10, 10)),
    transforms.RandomResizedCrop(size=(224, 224), scale=(0.9, 1.1), ratio=(0.9, 1.1)),
    transforms.ColorJitter(brightness=(0.5,1.5),contrast=(0.5,1.5),saturation=(0.5,1.5),hue=(-0.1,0.1)),
    transforms.GaussianBlur(kernel_size=3, sigma=2),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Standard from ImageNet
])

dataset = datasets.ImageFolder(datasetPath, transform=data_transforms)
print("Dataset classes: ", dataset.classes)


#Data Visualization: Bar Graph
distrubutionList = []

classHappy = [(image, label) for image, label in dataset if dataset.classes[label] == dataset.classes[0]]
distrubutionList.append(len(classHappy))

classFocused = [(image, label) for image, label in dataset if dataset.classes[label] == dataset.classes[1]]
distrubutionList.append(len(classFocused))

classNeutral = [(image, label) for image, label in dataset if dataset.classes[label] == dataset.classes[2]]
distrubutionList.append(len(classNeutral))

classSurprised = [(image, label) for image, label in dataset if dataset.classes[label] == dataset.classes[3]]
distrubutionList.append(len(classSurprised))

print(distrubutionList)

plt.bar(dataset.classes, distrubutionList)      
plt.ylabel('Number of Images in each Class')  # labling y-axis
plt.xlabel('Class Name')           # labling x-axis
plt.title('Distribution of images per class')  
plt.show()   


#Split into traing and test sets
trainSet, testSet = random_split(dataset, [0.8,0.2])

trainLoader = DataLoader(trainSet, shuffle=True, batch_size=32)
testLoadeer = DataLoader(testSet, shuffle=True, batch_size=32)


#Data Visuilization: Random images display 
for i in range(4):
    samples = [(image, label) for image, label in trainSet if dataset.classes[label] == dataset.classes[i]][:25]

    fig, axes = plt.subplots(5, 5, figsize=(12,8))
    for j, (image, label) in enumerate(samples):
        pilImage = TF.to_pil_image(image)
        ax = axes[j // 5, j % 5]
        ax.imshow(pilImage)
        ax.set_title(f"Image #{j+1}")
        ax.axis("off")

    plt.subplots_adjust(wspace=-0.7, hspace=0.5)
    plt.suptitle(f"Class: {dataset.classes[i]}", fontsize=22, fontweight="bold")
    plt.show()

    fig, axes = plt.subplots(5, 5, figsize=(12,10))
    for k, (image, label) in enumerate(samples):
        pilImage = TF.to_pil_image(image)

        normalized_array = np.array(pilImage)

        channel1 = normalized_array[:, :, 0].flatten()
        channel2 =normalized_array[:, :, 1].flatten()
        channel3 =normalized_array[:, :, 2].flatten()

        ax = axes[k // 5, k % 5]
        ax.hist(channel1, bins=100, range=[0, 256], density=True, color='red', alpha=0.5, label='Red')
        ax.hist(channel2, bins=100, range=[0, 256], density=True, color='green', alpha=0.5, label='Green')
        ax.hist(channel3, bins=100, range=[0, 256], density=True, color='blue', alpha=0.5, label='Blue')
        
        ax.set_title(f"Image #{k+1}", fontsize=8)
        ax.axis("on")
        ax.tick_params(axis='both', labelsize=8) 

    plt.subplots_adjust(wspace=0.5, hspace=0.6)
    plt.suptitle(f"Class: {dataset.classes[i]}", fontsize=22, fontweight="bold")
    plt.show()

   


"""

https://www.geeksforgeeks.org/how-to-normalize-images-in-pytorch/
https://medium.com/@sehjadkhoja0/title-exploring-and-analyzing-image-data-with-python-79a7f72f4d2b
https://www.kaggle.com/code/sanikamal/data-visualization-using-matplotlib
https://www.geeksforgeeks.org/how-to-rotate-an-image-by-an-angle-using-pytorch-in-python/
https://openreview.net/pdf?id=HXz7Vcm3VgM"""