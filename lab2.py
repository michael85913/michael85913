#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torch.utils.data as Data
from torchvision import datasets, transforms
from torch.autograd import Variable

def read_bci_data():
    S4b_train = np.load('C:/Users/michael85913/S4b_train.npz')
    X11b_train = np.load('C:/Users/michael85913/X11b_train.npz')
    S4b_test = np.load('C:/Users/michael85913/S4b_test.npz')
    X11b_test = np.load('C:/Users/michael85913/X11b_test.npz')

    train_data = np.concatenate((S4b_train['signal'], X11b_train['signal']), axis=0)
    train_label = np.concatenate((S4b_train['label'], X11b_train['label']), axis=0)
    test_data = np.concatenate((S4b_test['signal'], X11b_test['signal']), axis=0)
    test_label = np.concatenate((S4b_test['label'], X11b_test['label']), axis=0)

    train_label = train_label - 1
    test_label = test_label -1
    train_data = np.transpose(np.expand_dims(train_data, axis=1), (0, 1, 3, 2))
    test_data = np.transpose(np.expand_dims(test_data, axis=1), (0, 1, 3, 2))

    mask = np.where(np.isnan(train_data))
    train_data[mask] = np.nanmean(train_data)

    mask = np.where(np.isnan(test_data))
    test_data[mask] = np.nanmean(test_data)

    print(train_data.shape, train_label.shape, test_data.shape, test_label.shape)
    print(args.cuda)

    return train_data, train_label, test_data, test_label


# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=54, metavar='N',
                    help='input batch size for training (default: 54)')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train (default: 200)')
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

train_data, train_label, test_data, test_label = read_bci_data()

train_data = torch.from_numpy(train_data)
train_label = torch.from_numpy(train_label)
test_data = torch.from_numpy(test_data)
test_label = torch.from_numpy(test_label)

torch_dataset_tr = Data.TensorDataset(train_data, train_label)
torch_dataset_ts = Data.TensorDataset(test_data, test_label)
train_loader = Data.DataLoader(dataset=torch_dataset_tr, batch_size = args.batch_size, shuffle=True, num_workers=2)
test_loader = Data.DataLoader(dataset=torch_dataset_ts, batch_size = args.batch_size, shuffle=True, num_workers=2)


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
        out = out.view(out.size(0),-1)
        out = self.fc1(out)
        return out

model = EEGNet().double()
if args.cuda:
	device = torch.device('cuda')
	model.to(device)

#define optimizer/loss function
Loss = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
#optimizer = optim.Adam(model.parameters(), lr=args.lr)

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
            data, target = data.to(device=device), target.to(device=device, dtype=torch.long)
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
            data, target = data.to(device), target.to(device, dtype=torch.long)
        with torch.no_grad():
            output = model(data)
        test_loss += Loss(output, target.data)[0]
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


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




