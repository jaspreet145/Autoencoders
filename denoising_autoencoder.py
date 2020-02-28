# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 12:42:05 2020

@author: Jaspreet Singh
"""

import torch
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

transform = transforms.ToTensor()

train_data = datasets.MNIST(root='data' , train=True ,download=True ,transform=transform)
test_data = datasets.MNIST(root='data' , train=False ,download=True ,transform=transform)

batchsize = 20

train_loader = torch.utils.data.DataLoader(train_data,batch_size=batchsize)
test_loader = torch.utils.data.DataLoader(test_data,batch_size = batchsize)

images , labels = iter(train_loader).next()
images = images.numpy()

img  = np.squeeze(images[0]) #makes (28,28,1) -> (28,28)

#plt.imshow(img)
#plt.show()

import torch.nn as nn
import torch.nn.functional as F

class ConvDenoiser(nn.Module):
    def __init__(self):
        super(ConvDenoiser, self).__init__()
        #encoder
        self.conv1 = nn.Conv2d(1,32,3,padding =1)
        self.conv2 = nn.Conv2d(32,16 ,3,padding = 1)
        self.conv3 = nn.Conv2d(16,8, 3, padding = 1)
        self.pool = nn.MaxPool2d(2,2) # kernal and stride
        #decoder
        self.t_conv1 = nn.ConvTranspose2d(8,8, 3,stride=2)
        self.t_conv2 = nn.ConvTranspose2d(8,16,2,stride=2)
        self.t_conv3 = nn.ConvTranspose2d(16 , 32 , 2, stride=2)
        self.conv_out = nn.Conv2d(32,1, 3, padding=1)

    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)

        x = F.relu(self.t_conv1(x))
        x = F.relu(self.t_conv2(x))
        x = F.relu(self.t_conv3(x))
        x = F.sigmoid(self.conv_out(x))

        return x

model = ConvDenoiser()

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.01)

n_epochs = 20

noise_factor = 0.5

for epoch in range(n_epochs):
    train_loss = 0.0
    for data in train_loader:
        images , _ = data
        noisy_imgs = images + noise_factor*torch.randn(*images.shape)
        noisy_imgs = np.clip(noisy_imgs,0. , 1.)
        optimizer.zero_grad()
        outputs = model(noisy_imgs)
        loss = criterion(outputs,images)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss = train_loss/len(train_loader)
    print('epoch: {epoch}\t training loss : {trainloss}'.format(**{'epoch':epoch,'trainloss':train_loss}))

images,_ = iter(train_loader).next()
noisy_imgs = images + noise_factor*torch.randn(*images.shape)
noisy_imgs = np.clip(noisy_imgs,0. , 1.)
plt.imshow(np.squeeze(noisy_imgs[0]))
plt.show()
plt.imshow(np.squeeze(images[0]))
plt.show()
        
    
