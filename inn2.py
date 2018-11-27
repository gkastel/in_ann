from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import random
import numpy
#from torchvision import datasets, transforms
import matplotlib.pyplot as plt


from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

NBATCH   = 20 # batch size

X =  numpy.loadtxt(open("x_data.csv", "rb"), delimiter=",", skiprows=0)
Y =  numpy.loadtxt(open("y_data.csv", "rb"), delimiter=",", skiprows=0)

NSAMPLES = X.shape[0]
NDENDS   = X.shape[1]
NSYNS    = int(numpy.max(X))


all_x = torch.tensor(X).float()
all_y = torch.tensor(numpy.reshape(Y, (NSAMPLES,1))).float()

#shuffle
p = numpy.random.permutation(NSAMPLES)


#Trim some data to account for batch size

x_train = all_x[p[0:2500]]
y_train = all_y[p[0:2500]]

print(x_train.shape)

x_test = all_x[p[2500:3300]]
y_test = all_y[p[2500:3300]]



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #self.dlayers = [nn.Linear(NSYNS, 2, bias=False)  for n in range(NDENDS) ]
        self.dlayer =  nn.Linear(NDENDS, 2, bias=False)
        self.dlayer1 =  nn.Linear(NDENDS, 1, bias=False)
        self.soma_layer = nn.Linear(2, 1)

    def forward(self, xinput):
        #print (len(xinput))
        #dlayer_outs[bi, di] = torch.relu(self.dlayers[di]( xinput[bi, di] ) )
        if (args.act == 'sub'):
            out = torch.sqrt(torch.relu(self.dlayer( xinput ) ))
        elif (args.act == 'supra'):
            out  = torch.sigmoid(torch.relu(self.dlayer( xinput ) ))
        elif (args.act == 'mixed'):
            o = torch.relu(self.dlayer1( xinput ) )
            out  = torch.tensor([ torch.sigmoid(o1) , torch.sqrt(o1) ])
        else:
            out  = torch.relu(self.dlayer( xinput ) )
                #dlayer_outs[bi, di]  = torch.pow(self.dlayers[di]( xinput[bi, di] ), 0.8 )
                #dlayer_outs[bi, di]  = torch.sigmoid(self.dlayers[di]( xinput[bi, di] ) )
        #torch.clamp(dlayer_outs, min=0)
        
        
        soma_out = self.soma_layer( out )
        return soma_out
        #return F.log_softmax(x, dim=1)



class MyDataset(Dataset):
    def __init__(self, X, y):
        self.data = X
        self.target = y
    
    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        return x, y

    def __len__(self):
        return len(self.data)




def train(args, model, device, train_loader, optimizer, epoch):
    model.train()

    tot_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)

        loss = F.mse_loss(output, target, reduce=True)

        loss.backward()
        optimizer.step()

        for p in model.parameters():
                p.data.clamp_(0)
        
        tot_loss += loss.item();

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

    tot_loss /= len(train_loader.dataset)
    print('Epoch loss: %f'%(tot_loss))
    return tot_loss



def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            output = model(data)
            #test_loss += F.mse_loss(output, target, reduction='sum').item() # sum up batch loss
            test_loss += F.mse_loss(output, target, reduce=True)
            
    test_loss /= len(test_loader.dataset)

    print("\nTest set: Average loss: %4f\n"%( test_loss) ) 
    return test_loss




# Training settings
parser = argparse.ArgumentParser(description='PyTorch Example')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--act', default='linear',
                    help='activation: linear, sub, supra')
args = parser.parse_args()
use_cuda = True #not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if use_cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

train_dataset = MyDataset(x_train, y_train)
test_dataset = MyDataset(x_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=NBATCH, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=NBATCH, shuffle=True)

model = Net().to(device)
#optimizer = optim.SGD(model.parameters(), lr=args.lr)
optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))

print("Act=%s"%(args.act))

error_log = numpy.zeros((args.epochs, 2))
for epoch in range(0, args.epochs):
    error_log[epoch, 0 ] = train(args, model, device, train_loader, optimizer, epoch)
    error_log[epoch, 1 ] = test(args, model, device, test_loader)

numpy.savetxt('errors2-%s.txt'%(args.act), error_log, fmt="%f")

plt.figure()

plt.plot(error_log[:, 0])
plt.plot(error_log[:, 1])
plt.xlabel("Epoch")
plt.ylabel("MSE Error")
plt.legend(["Train error","Test error"] )
plt.show()




