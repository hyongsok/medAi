
# Import
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper Parameters
num_epochs = 5
num_classses = 4

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])

#Image Data load
trainset = torchvision.datasets.ImageFolder(root='D:\DM', train=True, download=True, transform=transform)
testset = torchvision.datasets.ImageFolder(root='D:\DM', train=False, download=True, transform=transform)


trainloader = DataLoader(trainset, batch_size=8, shuffle=True, num_workers=2)
testloader = DataLoader(testset, batch_size=8, shuffle=False, num_workers=2)

def imshow(img):
    img = img / 2 + 0.5 #unnormalize
    np_img = img.numpy()
    #plt.imshow(np_img)
    plt.imshow(np.transpose(np_img, (1,2,0)))

    print(np_img.shape)
    print((np.transpose(np_img, (1,2,0))).shape)

