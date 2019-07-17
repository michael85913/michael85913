#!/usr/bin/env python
# coding: utf-8

# In[325]:


from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torch.utils.data as Data
from torchvision import datasets, transforms
from torch.autograd import Variable


# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=150, metavar='N',
                    help='number of epochs to train (default: 150)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

args = parser.parse_args(args=[])
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Dataloader
train_loader_np =  np.load("C:/Users/michael85913/X11b_train.npz")
test_loader_np =  np.load("C:/Users/michael85913/X11b_test.npz")


train_loader = train_loader_np['signal'].reshape((540,1,750,2))
train_x = torch.from_numpy(train_loader)
train_y = torch.from_numpy(train_loader_np['label'])
test_x = torch.from_numpy(test_loader_np['signal'])
test_y = torch.from_numpy(test_loader_np['label'])

torch_dataset = Data.TensorDataset(train_x, train_y)
train_loader = Data.DataLoader(dataset=torch_dataset, batch_size = args.batch_size, shuffle=True, num_workers=2)

#Define Network, we implement LeNet here
class EEGNet(nn.Module):
    def __init__(self):
        super(EEGNet, self).__init__()
        #self.T = 120
        
        #layer1
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(1,51),stride=(1, 1), padding=(0, 25), bias=False)
        self.batchnorm1 = nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #layer2
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(2,1),stride=(1, 1), groups=16, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #ELU(alpha=1.0)
        self.pooling2 = nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4), padding=0)
        #Dropout(p=0.25)
        #layer3
        self.conv3 = nn.Conv2d(32, 32, kernel_size=(1,15),stride=(1, 1), padding =(0, 7), bias=False)
        self.batchnorm3 = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #ELU(alpha=1.0)
        self.pooling3 = nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8), padding=0)
        #Dropout(p=0.25)
        
        #FC layer
        self.fc1 = nn.Linear(in_features = 736, out_features = 2, bias=True)
    def forward(self, out):
        out = self.conv1(out)
        out = self.batchnorm1(out)
        out = self.conv2(out)
        out = F.relu(self.batchnorm2(out))
        out = self.pooling2(out)
        out = F.dropout(out, 0.25)
        out = self.conv3(out)
        out = F.relu(self.batchnorm3(out))
        out = self.pooling3(out)
        out = F.dropout(out, 0.25)
        
        out = out.view(-1,out.size(0))
        out = self.fc1(out)
        return out

model = EEGNet()
if args.cuda:
	device = torch.device('cuda')
	model.to(device)

#define optimizer/loss function
Loss = nn.CrossEntropyLoss()
#optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
optimizer = optim.Adam(model.parameters(), lr=args.lr)

#learning rate scheduling
def adjust_learning_rate(optimizer, epoch):
    if epoch < 10:
       lr = 0.01
    elif epoch < 15:
       lr = 0.001
    else: 
       lr = 0.0001

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

#training function
def train(epoch):
    model.train()
    adjust_learning_rate(optimizer, epoch)

    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.to(device=device, dtype=torch.float), target.to(device=device, dtype=torch.float)

        optimizer.zero_grad()
        output = model(data)
        loss = Loss(output, target)
        loss.backward()
        optimizer.step()
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))

#Testing function
def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(test_loader):
        if args.cuda:
            data, target = data.to(device), target.to(device)
        with torch.no_grad():
            output = model(data)
        test_loss += Loss(output, target).data[0]
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    test_loss = test_loss
    test_loss /= len(test_loader) # loss function already averages over batch size
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

#run and save model
for epoch in range(1, args.epochs + 1):
    train(epoch)
    test(epoch)


# In[277]:


import numpy as np
import torch

# Dataloader
train_loader_np =  np.load("C:/Users/michael85913/X11b_train.npz")
test_loader_np =  np.load("C:/Users/michael85913/X11b_test.npz")

train_loader_np_c1, train_loader_np_c2 = np.split(train_loader_np['signal'][0], [1], axis=1)
train_loader = np.concatenate([train_loader_np_c1, train_loader_np_c2], axis=0)
train_loader_1 = train_loader.reshape((1,2,750))


for i in range (539):
  train_loader_np_c1, train_loader_np_c2 = np.split(train_loader_np['signal'][i+1], [1], axis=1)
  train_loader = np.concatenate([train_loader_np_c1, train_loader_np_c2], axis=0)
  train_loader = train_loader.reshape((1,2,750))
  train_loader_1 = np.concatenate([train_loader_1, train_loader])
train_loader_1 = train_loader_1.reshape((540,1,2,750))
train_x = torch.from_numpy(train_loader)
train_y = torch.from_numpy(train_loader_np['label'])
test_x = torch.from_numpy(test_loader_np['signal'])
test_y = torch.from_numpy(test_loader_np['label'])

print(train_loader_1.size())


# In[312]:


import numpy as np
import torch
train_loader_np =  np.load("C:/Users/michael85913/X11b_train.npz")
test_loader_np =  np.load("C:/Users/michael85913/X11b_test.npz")
 
print(train_loader_np['signal'][0][349])


# In[305]:


import numpy as np
import torch

# Dataloader
train_loader_np =  np.load("C:/Users/michael85913/X11b_train.npz")
test_loader_np =  np.load("C:/Users/michael85913/X11b_test.npz")
train_loader_np_c1, train_loader_np_c2 = np.split(train_loader_np['signal'][0]
train_loader = [train_loader_np_c1,train_loader_np_c2]                                                 
#train_loader = np.concatenate([train_loader_np_c1, train_loader_np_c2], axis=0)



print(train_loader)


# In[310]:


import numpy as np
import torch

# Dataloader
train_loader_np =  np.load("C:/Users/michael85913/X11b_train.npz")
test_loader_np =  np.load("C:/Users/michael85913/X11b_test.npz")

train_loader = train_loader_np['signal'].reshape((540,1,750,2))
print(train_loader[0][0][349][0])


# In[ ]:




