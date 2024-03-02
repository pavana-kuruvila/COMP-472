import torch
import os
from torchvision import transforms 
from PIL import Image
import torchvision.datasets as datasets
from torch.utils.data import DataLoader , random_split
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF

datasetPath = "/Users/vanakuruvila/Documents/School/COMP 472/DataSet copy"

#Data Cleaning 
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    #transforms.Normalize(mean=mean, std=std)
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Standard from ImageNet
])

dataset = datasets.ImageFolder(datasetPath, transform=data_transforms)
print("Dataset classes: ", dataset.classes)

"""
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
"""

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


#Data Visuilization: Histograph display 
"""for i in range(4):
    samples = [(image, label) for image, label in trainSet if dataset.classes[label] == dataset.classes[i]][:25]



    fig, axes = plt.subplots(5, 5, figsize=(12,8))
    for k, (image, label) in enumerate(samples):
        pilImage = TF.to_pil_image(image)
        pixelIntesities = list(pilImage.getdata())
        ax = axes[k // 5, k % 5]
        ax.hist(pixelIntesities, bins=256, range=(0, 256), density=True, alpha=0.7)
        ax.set_title(f"Image #{k+1}")
        ax.axis("off")

    plt.subplots_adjust(wspace=-0.7, hspace=0.5)
    plt.suptitle(f"Class: {dataset.classes[i]}", fontsize=22, fontweight="bold")
    plt.show()"""

   

pilImage = TF.to_pil_image(samples[0][0])
pixelIntesities = list(pilImage.getdata())
print(len(pixelIntesities))


plt.hist(pixelIntesities)
plt.show()

