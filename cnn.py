import torch
import numpy as np
from torchvision import transforms 
from PIL import Image
import torchvision.datasets as datasets
from torch.utils.data import DataLoader , random_split
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import torch.nn as nn

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
testLoader = DataLoader(testSet, shuffle=True, batch_size=32)

num_epochs = 10
num_classes = 4
learning_rate = 0.001

classes = dataset.classes



class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init()
        self.conv_layer = nn.Sequential(

            #we currently have 4 convulutional layers, number of filters increase throught the layers so the CNN can learn higher level features
            #we could add another layer with more channels to deepen the feature recognitions, but too many layer can lead to over fitting

            #out channels is number of filters and kernel size is the filter size so 3x3 and padding is 1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            #normalization
            nn.BatchNorm2d(32),
            #allows a small, non-zero gradient when the unit is not active helps with vanishing gradient problem.
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            #pooling window is 2x2 and stride means it moves by 2 pixels at a time
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
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
            nn.Linear(8 * 8 * 64, 1000),
            nn.ReLU(inplace=True),
            nn.Linear(1000, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512, 4)
        )
    
def forward(self, x):
    # performs the convolutional layers
    x = self.conv_layer(x)
    # flatten to 2d 
    x = x.view(x.size(0), -1)
    # fully connected layer to performs classification base don the feautres that model extracted
    x = self.fc_layer(x)
    return x

model = CNN()
#loss function 
criterion = nn.CrossEntropyLoss()
#updates the weight of the model
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

total_step = len(trainLoader)
loss_list = []
acc_list = []

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
        optimizer.step()

        # Train accuracy
        total = label.size(0)
        _, predicted = torch.max(outputs.data, 1)
        #compare the predicted label with the real label, correct hold # of correct label for the current batch 
        correct = (predicted == label).sum().item()
        #compute batch accuracy
        acc_list.append(correct / total)

        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
                (correct / total) * 100))
            
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    #gets total #for images proccesed and the number of correct procced 
    for images, labels in testLoader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the test images: {} %'
        .format((correct / total) * 100))
    

    #look into early stopping techniques
#to save
#torch.save(modelA.state_dict(), PATH)
#torestore
#modelB = TheModelBClass(*args, **kwargs)
#smodelB.load_state_dict(torch.load(PATH), strict=False)