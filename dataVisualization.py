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

#---------  Resizing the dataset ----------#

data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

#transforming the data
dataset = datasets.ImageFolder(datasetPath, transform=data_transforms)
print("Dataset classes: ", dataset.classes)
#----------------------------------------------------------------#


#data visualization before the cleaning, to get a better understanding of the photos in the dataset
# ----------------- Data Visualization: Bar Graph -------------------- #
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
# ---------------------------------------------------------- #



#Split into traing and test sets
trainSet, testSet = random_split(dataset, [0.8,0.2])

trainLoader = DataLoader(trainSet, shuffle=True, batch_size=32)
testLoadeer = DataLoader(testSet, shuffle=True, batch_size=32)



# ------------------ Data Visuilization: Random images display -------------------- #
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
    plt.suptitle(f"Class: {dataset.classes[i]}", fontsize=22, fontweight="bold")
    plt.show()

    #displaying the image pixel densities 
    fig, axes = plt.subplots(5, 5, figsize=(12,10))
    for k, (image, label) in enumerate(samples):
        pilImage = TF.to_pil_image(image)

        normalized_array = np.array(pilImage)

        #exatracting each RGB channel to overlay on the histogram
        channel1 = normalized_array[:, :, 0].flatten()
        channel2 =normalized_array[:, :, 1].flatten()
        channel3 =normalized_array[:, :, 2].flatten()

        #overlaying the density channels
        ax = axes[k // 5, k % 5]
        ax.hist(channel1, bins=100, range=[0, 256], density=True, color='red', alpha=0.5, label='Red')
        ax.hist(channel2, bins=100, range=[0, 256], density=True, color='green', alpha=0.5, label='Green')
        ax.hist(channel3, bins=100, range=[0, 256], density=True, color='blue', alpha=0.5, label='Blue')
        
        #titles and grid line for graphs
        ax.set_title(f"Image #{k+1}", fontsize=8)
        ax.axis("on")
        ax.tick_params(axis='both', labelsize=8) 

    plt.subplots_adjust(wspace=0.5, hspace=0.6)
    plt.suptitle(f"Class: {dataset.classes[i]}", fontsize=22, fontweight="bold")
    plt.show()
# ------------------------------------------------------------------------ #
        
