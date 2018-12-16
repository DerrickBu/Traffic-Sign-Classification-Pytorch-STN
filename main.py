from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transformm
from torchvision import datasets, transforms
import torchvision
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt


class Plotter:
    def plot_curve(self, sub_plot, params, train_column, valid_column, linewidth = 2, train_linestyle = "b-", valid_linestyle = "g-"):
        train_values = params[train_column]
        valid_values = params[valid_column]
        epochs = train_values.shape[0]
        x_axis = np.arange(epochs)
        #sub_plot.plot(x_axis[train_values > 0], train_values[train_values > 0], train_linestyle, linewidth=linewidth, label="train")
        sub_plot.plot(x_axis[valid_values > 0], valid_values[valid_values > 0], valid_linestyle, linewidth=linewidth, label="train")
        return epochs

    # Plot history curves
    def plot_learning_curves(self, params):
        curves_figure = plt.figure(figsize = (10, 4))
        sub_plot = curves_figure.add_subplot(1, 2, 1)
        epochs_plotted = self.plot_curve(sub_plot, params, train_column = "train_acc", valid_column = "val_acc")

        plt.grid()
        plt.legend()
        plt.xlabel("epoch")
        plt.ylabel("accuracy")
        plt.xlim(0, epochs_plotted)
'''
        sub_plot = curves_figure.add_subplot(1, 2, 2)
        epochs_plotted = self.plot_curve(sub_plot, params, train_column = "train_loss", valid_column = "val_loss")

        plt.grid()
        plt.legend()
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.xlim(0, epochs_plotted)
        plt.yscale("log")
'''
plotter = Plotter()

# Training settings
parser = argparse.ArgumentParser(description='PyTorch GTSRB example')
parser.add_argument('--data', type=str, default='data', metavar='D',
                    help="folder where data is located. train_data.zip and test_data.zip need to be found in the folder")
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()

torch.manual_seed(args.seed)

### Data Initialization and Loading
from data import initialize_data, data_transforms # data.py in the same folder
initialize_data(args.data) # extracts the zip files, makes a validation set

train_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(args.data + '/train_images',
                         transform=data_transforms),
    batch_size=args.batch_size, shuffle=True, num_workers=4)
val_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(args.data + '/val_images',
                         transform=data_transforms),
    batch_size=args.batch_size, shuffle=False, num_workers=4)

### Neural Network and Optimizer
# We define neural net in model.py so that it can be reused by the evaluate.py script

from model import Net
model = Net()
#from model import model


optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
histories = {
            "train_loss": np.empty(0, dtype=np.float32),
            "train_acc": np.empty(0, dtype=np.float32),
            "val_loss": np.empty(0, dtype=np.float32),
            "val_acc": np.empty(0, dtype=np.float32)
    }
#optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99))
#criterion = nn.CrossEntropyLoss()


def validate_model(loader, model, index):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0

    for images, labels in loader:
        images_batch = Variable(images, volatile=True)
        labels_batch = Variable(labels.long())

       

        output = model(images_batch)
        loss = nn.functional.cross_entropy(
            output, labels_batch.long(), size_average=False)
        total_loss += loss.data[0]
        total += len(labels_batch)        
        #correct += (labels_batch == output.max(1)[1]).data.sum()        
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(labels.data.view_as(pred)).cpu().sum()
    if index == 2:   
        print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            total_loss / len(val_loader.dataset), correct, len(val_loader.dataset),
            100. * correct / len(val_loader.dataset)))
    
    model.train()

    average_loss = total_loss / total
    return 100. * correct / len(val_loader.dataset), average_loss

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))
    
    train_acc, train_loss = validate_model(train_loader, model, 1)
    val_acc, val_loss = validate_model(val_loader, model, 2)

    histories['train_loss'] = np.append(histories['train_loss'], [train_loss])
    histories['val_loss'] = np.append(histories['val_loss'], [val_loss])
    histories['val_acc'] = np.append(histories['val_acc'], [val_acc])
    histories['train_acc'] = np.append(histories['train_acc'], [train_acc])
  



    

for epoch in range(1, args.epochs + 1):
    train(epoch)
    #validation()
    model_file = 'model_' + str(epoch) + '.pth'
    torch.save(model.state_dict(), model_file)
    print('\nSaved model to ' + model_file + '. You can run `python evaluate.py ' + model_file + '` to generate the Kaggle formatted csv file')
plotter.plot_learning_curves(histories)
plt.show()
