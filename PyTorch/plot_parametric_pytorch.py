'''
Reproduces the parametric plot experiment from the paper
for a network like C3.

Plots a parametric plot between SB and LB
minimizers demonstrating the relative sharpness
of the two minima.

Requirements:
- Keras (only for CIFAR-10 dataset; easy to avoid)
- Matplotlib
- Numpy

TODO:
- Enable the code to run on CUDA as well.
  (As of now, it only runs on CPU)

Run Command:
        python plot_parametric_pytorch.py

The plot is saved as C3ish.pdf
'''

import numpy as np
np.random.seed(1337)
import torch
torch.manual_seed(1337)
from torch.autograd import Variable
import torch.nn.functional as F
#import keras #This dependency is only for loading the CIFAR-10 data set
from copy import deepcopy
import vgg 

import torchvision
import torch
from torchvision import transforms

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# This is where you can load any model of your choice.
# I stole PyTorch Vision's VGG network and modified it to work on CIFAR-10.
# You can take this line out and add any other network and the code
# should run just fine. 
model = vgg.vgg11_bn()


# Forward pass
opfun = lambda X: model.forward(Variable(torch.from_numpy(X)))

# Forward pass through the network given the input
predsfun = lambda op: np.argmax(op.data.numpy(), 1)

# Do the forward pass, then compute the accuracy
accfun   = lambda op, y: np.mean(np.equal(predsfun(op), y.squeeze()))*100

# Initial point
x0 = deepcopy(model.state_dict())

# Number of epochs to train for
# Choose a large value since LB training needs higher values
nb_epochs = 100


# If SB.pth and LB.pth are available
# set hotstart = True and run only the
# parametric plot (i.e., don't train the network)
hotstart = True

criterion = torch.nn.CrossEntropyLoss()

if not hotstart:
    for fractions_of_dataset in [50, 200]: #Run with 1/10th the data set and 1/200th the dataset
        optimizer = torch.optim.Adam(model.parameters())
        model.load_state_dict(x0)
        model.cuda()
        average_loss_over_epoch = '-'
        batch_size = int(len(trainset)/fractions_of_dataset)
        print('Optimizing the network with batch size {0}'.format(batch_size))
        
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
        
        np.random.seed(1337) #So that both networks see same sequence of batches
        for e in range(nb_epochs):
            model.eval()
            print('Epoch:', e, ' of ', nb_epochs, 'Average loss:', average_loss_over_epoch)
            average_loss_over_epoch = 0.
            # Checkpoint the model every epoch
            torch.save(model.state_dict(), ('SB' if fractions_of_dataset==200 else 'LB')+'.pth')

            # Training loop!
            for images, labels in trainloader:
                model.train()
                optimizer.zero_grad()
                ops = model(images.cuda())
                loss = criterion(ops, labels.cuda())
                average_loss_over_epoch += loss / float(fractions_of_dataset)
                loss.backward()
                optimizer.step()

# Load stored values
# If hotstarted, loop is ignored and SB/LB files must be provided
mbatch = torch.load('LB.pth')
mstoch = torch.load('SB.pth')
print('Loaded stored solutions') 


grid_size = 25 #How many points of interpolation between [-1, 2]
data_for_plotting = np.zeros((grid_size, 4))
alpha_range = np.linspace(-1, 2, grid_size)
i = 0

# Fill in the train and test, loss and accuracy values
# for `grid_size' points in the interpolation
for i, alpha in enumerate(alpha_range):
    mydict = {}
    for key in mbatch:
        if 'var' in key:
            x = mbatch[key].sqrt() * alpha + (1 - alpha) * mstoch[key].sqrt()
            mydict[key] = x * x
        else:
            mydict[key] = mbatch[key] * alpha + (1 - alpha) * mstoch[key]
    model.load_state_dict(mydict)
    model.cuda()
    testloss = trainloss = testacc = trainacc = 0.
    batch_size = 5000
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    model.eval()
    with torch.no_grad():
        total = 0
        correct = 0
        for images, labels in trainloader:
            outputs = model(images.cuda())
            loss = criterion(outputs, labels.cuda())
            trainloss += loss / float(len(trainset) / batch_size)
            _, predicted = torch.max(outputs.cpu(), 1)
            total += labels.size(0)
            correct += (predicted.cpu() == labels).sum().item()
        trainacc = correct / float(total)

        total = 0
        correct = 0
        for images, labels in testloader:
            outputs = model(images.cuda())
            loss = criterion(outputs, labels.cuda())
            testloss += loss / float(len(testset) / batch_size)
            _, predicted = torch.max(outputs.cpu(), 1)
            total += labels.size(0)
            correct += (predicted.cpu() == labels).sum().item()
        testacc = correct / float(total)
    
    data_for_plotting[i, 0] = trainloss
    data_for_plotting[i, 1] = testloss
    data_for_plotting[i, 2] = trainacc
    data_for_plotting[i, 3] = testacc
    
    print('alpha : {0} done'.format(alpha))

np.save('intermediate-values', data_for_plotting)

# Actual plotting;
# if matplotlib is not available, use any tool of your choice by
# loading intermediate-values.npy
import matplotlib.pyplot as plt
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.semilogy(alpha_range, data_for_plotting[:, 0], 'b-')
ax1.semilogy(alpha_range, data_for_plotting[:, 1], 'b--')

ax2.plot(alpha_range, data_for_plotting[:, 2], 'r-')
ax2.plot(alpha_range, data_for_plotting[:, 3], 'r--')

ax1.set_xlabel('alpha')
ax1.set_ylabel('Cross Entropy', color='b')
ax2.set_ylabel('Accuracy', color='r')
ax1.legend(('Train', 'Test'), loc=0)

ax1.grid(b=True, which='both')
plt.savefig('C3ish.pdf')
print('Saved figure; Task complete')
