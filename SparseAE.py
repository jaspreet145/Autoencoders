# -*- coding: utf-8 -*-
"""
Created on Wed Mar 2 21:46:00 2020
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

import torch.nn as nn
import torch.nn.functional as F

class SparseAE(nn.Module):
    def __init__(self):
        super(SparseAE,self).__init__()
        self.input = nn.Linear(28*28,128)
        self.hidden = nn.Linear(128,28*28)


    def forward(self,x):
        x = x.view(-1,28*28)
        x = F.relu(self.input(x))
        x= F.sigmoid(self.hidden(x))
        return x

model = SparseAE()

criterion = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)


n_epochs = 20

noise_factor = 0.5

for epoch in range(n_epochs):
    train_loss = 0.0
    for data in train_loader:
        images , _ = data
        noisy_imgs = images + noise_factor*torch.randn(*images.shape)
        noisy_imgs = np.clip(noisy_imgs,0. , 1.)
        noisy_imgs = noisy_imgs.view(-1,28*28)
        optimizer.zero_grad()
        outputs = model(noisy_imgs)
        loss = criterion(outputs,images.view(-1,28*28))
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss = train_loss/len(train_loader)
    print('epoch: {epoch}\t training loss : {trainloss}'.format(**{'epoch':epoch,'trainloss':train_loss}))

images,_ = iter(train_loader).next()
noisy_imgs = images + noise_factor*torch.randn(*images.shape)
noisy_imgs = np.clip(noisy_imgs,0. , 1.)
noisy_imgs = noisy_imgs.view(-1,28*28)
outputs = model(noisy_imgs)
output = outputs[0]
plt.imshow(output.view(28,28).detach())
plt.show()
