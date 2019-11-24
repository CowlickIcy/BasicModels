#%%
import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision
from torch.autograd import Variable
from torch.utils.data import DataLoader
import cv2

batch_size = 100
#%%
#  train dataset
train_dataset = datasets.MNIST(root='./num/',
                               train=True,
                               transform=transforms.ToTensor(),
                               download=True)
# test dataset
test_dataset = datasets.MNIST(root='./num/',
                              train=False,
                              transform=transforms.ToTensor(),
                              download=True)

#%%
# Dataset to load dataset name
# Batch_size to set image number
# In the loading the dataset will be shuffle and be packed

# Load the train_dataset
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
# Load the test_dataset
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=True)

# Build a dataloader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=True)


#%%   Make the single image visual
images, labels = next(iter(train_loader))
img = torchvision.utils.make_grid(images)

img = img.numpy().transpose(1, 2, 0)
std = [0.5, 0.5, 0.5]
mean = [0.5, 0.5, 0.5]
img = img * std + mean
print(labels)
cv2.imshow('win', img)
key_pressed = cv2.waitKey(0)

#%%
# Convolution layer use torch.nn.Conv2d
# Activating layer use torch.nn.ReLU
# Pooling layer use torch.nn.MaxPool2d
# Max_connection layer use  torch.nn.Linear

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1, 6, 3, 1, 2), nn.ReLU(),
                                   nn.MaxPool2d(2, 2))

        self.conv2 = nn.Sequential(nn.Conv2d(6, 16, 5), nn.ReLU(),
                                   nn.MaxPool2d(2, 2))

        self.fc1 = nn.Sequential(nn.Linear(16 * 5 * 5, 120),
                                 nn.BatchNorm1d(120), nn.ReLU())

        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.BatchNorm1d(84),
            nn.ReLU(),
            nn.Linear(84, 10)) # the 10 is because of the label between 0-9

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

#%%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 64
LR = 0.001

net = LeNet().to(device)
# Loss function use the cross entropy loss
criterion = nn.CrossEntropyLoss()
# optimizer use the adam adaptive optimization algorithm
optimizer = optim.Adam(net.parameters(), lr=LR,)

epoch = 1
if __name__ == '__main__':
    for epoch in range(epoch):
        sum_loss = 0.0
        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()
            optimizer.zero_grad()  # Make the gradient to zero
            outputs = net(inputs)  # Make the data into the net and forward
            loss = criterion(outputs, labels)  # Get the loss function
            loss.backward()  # backward broadcast
            optimizer.step()  # update the para by the gradient

            # print(loss)
            sum_loss += loss.item()
            if i % 100 == 99:
                print('[%d,%d] loss:%.03f' %
                      (epoch + 1, i + 1, sum_loss / 100))
                sum_loss = 0.0

#%% Test model
net.eval()
correct = 0
total = 0
for data_test in test_loader:
    images, labels = data_test
    images, labels = Variable(images).cuda(), Variable(labels).cuda()
    output_test = net(images)
    _, predicted = torch.max(output_test, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()
print("correct1: ", correct)
print("Test acc: {0}".format(correct.item() / len(test_dataset)))

