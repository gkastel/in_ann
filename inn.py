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

if (True):    # Load Hipp data
    X =  numpy.loadtxt(open("x_data_hipp.csv", "rb"), delimiter=",", skiprows=0)
    Y =  numpy.loadtxt(open("y_data_hipp.csv", "rb"), delimiter=",", skiprows=0)
else:   # load PFC data
    X =  numpy.loadtxt(open("x_data_pfc.csv", "rb"), delimiter=",", skiprows=0)
    Y =  numpy.loadtxt(open("y_data_pfc.csv", "rb"), delimiter=",", skiprows=0)


# number of samples
NSAMPLES = X.shape[0]
NSAMPLES -= NSAMPLES%NBATCH #trim data set to fit batch size

print("NSAMPLES = %d"%(NSAMPLES))
X = X[0:NSAMPLES]
Y = Y[0:NSAMPLES]

# number of dendrites
NDENDS   = X.shape[1]

#NSYNS    = int(numpy.max(X))

# number of symapses set to 60, even though at most 40 of them are active
NSYNS    = 60


# Put all input data in one array
xdata = numpy.zeros( (NSAMPLES, NDENDS, NSYNS));
for i in range(NSAMPLES):
	for j in range(NDENDS):
		syns = int(X[i,j])
		xdata[i,j, 0:syns] = 1. # 1 for active synapses, 0 for the rest


# ensure they are 32bit floats  or we ll get an error
all_x = torch.tensor(xdata).float()    
all_y = torch.tensor(numpy.reshape(Y, (NSAMPLES,1))).float()

# shuffle rows
p = numpy.random.permutation(NSAMPLES)


# Get approx 75% percent of data for the training set
# The size of input data must be a multiple of the batch size, so we trim some data:

NSPLIT = int(float(NSAMPLES)*0.75);
NSPLIT -= NSPLIT%NBATCH

x_train = all_x[p[0:NSPLIT]]
y_train = all_y[p[0:NSPLIT]]

print("Shape of train set data:")
print(x_train.shape)


# Test set data
x_test = all_x[p[NSPLIT:NSAMPLES]]
y_test = all_y[p[NSPLIT:NSAMPLES]]

print("Shape of test set data:")
print(x_test.shape)



# The network

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # each dendrite is a layer
        self.dlayers = [nn.Linear(NSYNS, 1, bias=False)  for n in range(NDENDS) ]

        # soma/output layer is linear
        self.soma_layer = nn.Linear(NDENDS, 1)

    def forward(self, xinput):

        # stores the intermediate outputs of dendritic layers
        dlayer_outs = torch.zeros( (NBATCH, NDENDS ) )

        for bi in range(NBATCH): # we need to loop over each minibatch entry
            for di in range(NDENDS):

                if (args.act == 'mixed'):
                    if (di < NDENDS*0.4): # 40% of dendrites assumed sigmoid
                        dlayer_outs[bi, di]  = torch.sigmoid(torch.relu(self.dlayers[di]( xinput[bi, di] )) )
                    else:
                        dlayer_outs[bi, di]  = torch.sqrt(torch.relu(self.dlayers[di]( xinput[bi, di] ) ) )
                elif (args.act == 'sub'):
                    dlayer_outs[bi, di]  = torch.sqrt(torch.relu(self.dlayers[di]( xinput[bi, di] ) ))
                elif (args.act == 'supra'):
                    dlayer_outs[bi, di]  = torch.sigmoid(torch.relu(self.dlayers[di]( xinput[bi, di] ) ))
                else:
                    dlayer_outs[bi, di]  = torch.relu(self.dlayers[di]( xinput[bi, di] ) )


        # torch.clamp(dlayer_outs, min=0) # clamp output to be always above zero
        # soma layer receives the output of dendritic layers as input
        soma_out = self.soma_layer( dlayer_outs )
        return soma_out



# wraps our input data
class MyDataset(Dataset):
    def __init__(self, X, y):
        self.data   = X
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
        output = model(data) # pass data through the model

        loss = F.mse_loss(output, target, reduce=True) # mean squared error loss

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



def test(args, model, device, loader):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)

            output = model(data)
            #test_loss += F.mse_loss(output, target, reduction='sum').item() # sum up batch loss
            test_loss += F.mse_loss(output, target, reduce=True)
            
    test_loss /= len(loader.dataset)

    print("\nTest set: Average loss: %4f\n"%( test_loss) ) 

    return test_loss




# Predict using trained model. Almost the same as pred()
def pred(args, model, device, loader):
    model.eval()
    test_loss = 0

    # fill in with predicted values
    out_data = torch.zeros((len(loader.dataset),1))
    idx=0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            out_data[idx:idx+len(data)] = model(data)
            #test_loss += F.mse_loss(output, target, reduction='sum').item() # sum up batch loss
            #test_loss += F.mse_loss(output, target, reduce=True)
            idx += len(data)
            
    #test_loss /= len(loader.dataset)
    # print("\nTest set: Average loss: %4f\n"%( test_loss) ) 
    print("Out shape = ")
    print(out_data.shape)

    return out_data



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
use_cuda = False #not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if use_cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

train_dataset = MyDataset(x_train, y_train)
test_dataset = MyDataset(x_test, y_test)
pred_dataset = MyDataset(all_x[0:NSAMPLES], all_y)

train_loader = DataLoader(train_dataset, batch_size=NBATCH, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=NBATCH, shuffle=True)

pred_loader = DataLoader(pred_dataset, batch_size=NBATCH, shuffle=False)



model = Net().to(device)
#optimizer = optim.SGD(model.parameters(), lr=args.lr)
optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))

print("Act=%s"%(args.act))

error_log = numpy.zeros((args.epochs, 2))
for epoch in range(0, args.epochs):
    error_log[epoch, 0 ] = train(args, model, device, train_loader, optimizer, epoch)
    error_log[epoch, 1 ] = test(args, model, device, test_loader)

numpy.savetxt('errors-%s.txt'%(args.act), error_log, fmt="%f")

predictions =  pred(args, model, device, pred_loader)
print("predictions size=")
print(predictions.shape)

numpy.savetxt('predictions-%s.txt'%(args.act), predictions, fmt="%f")
numpy.savetxt('actual-%s.txt'%(args.act), all_y, fmt="%f")


plt.figure()
plt.plot(error_log[:, 0])
plt.plot(error_log[:, 1])
plt.xlabel("Epoch")
plt.ylabel("MSE Error")
plt.legend(["Train error","Test error"] )


plt.figure()
plt.scatter(predictions, all_y)



# uncomment to show  plots
#plt.show()

