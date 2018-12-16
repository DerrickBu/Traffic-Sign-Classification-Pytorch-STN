import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
nclasses = 43 # GTSRB as 43 classes

from torch import nn

'''
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.dropout = nn.Dropout2d()
        self.conv1 = nn.Conv2d(3, 100, 5, padding=2)
        self.max_pool2d1 = nn.MaxPool2d(2, stride=2)
        self.batch_norm_2d1 = nn.BatchNorm2d(100)

        self.conv2 = nn.Conv2d(100, 150, 3, padding=1)
        self.max_pool2d2 = nn.MaxPool2d(2, stride=2)
        self.batch_norm_2d2 = nn.BatchNorm2d(150)

        self.conv3 = nn.Conv2d(150, 250, 1, padding=0)
        self.max_pool2d3 = nn.MaxPool2d(2, stride=2)
        self.batch_norm_2d3 = nn.BatchNorm2d(250)
        self.fc1 = nn.Linear(250 * 4 * 4, 350)
        self.fc2 = nn.Linear(350, nclasses)


        # Spatial transformer localization-network
        self.localization1 = nn.Sequential(
            nn.Conv2d(3, 200, kernel_size=7, padding=3),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(200, 300, kernel_size=5, padding =2),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        self.localization2 = nn.Sequential(
            nn.Conv2d(150, 150, kernel_size=5, padding=2),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(150, 150, kernel_size=3, padding =1),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )


        # Regressor for the 3 * 2 affine matrix
        self.fc_loc1 = nn.Sequential(
            nn.Linear(300 * 8 * 8, 200),
            nn.ReLU(True),
            nn.Linear(200, 3 * 2)
        )

        self.fc_loc2 = nn.Sequential(
            nn.Linear(150 * 2 * 2, 150),
            nn.ReLU(True),
            nn.Linear(150, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc1[2].weight.data.fill_(0)
        self.fc_loc1[2].bias.data = torch.FloatTensor([1, 0, 0, 0, 1, 0])

         # Initialize the weights/bias with identity transformation
        self.fc_loc2[2].weight.data.fill_(0)
        self.fc_loc2[2].bias.data = torch.FloatTensor([1, 0, 0, 0, 1, 0])


    # Spatial transformer network forward function
    def stn1(self, x):
        xs1 = self.localization1(x)
        xs1 = xs1.view(-1, 300 * 8 * 8)
        theta1 = self.fc_loc1(xs1)
        theta1 = theta1.view(-1, 2, 3)

        grid1 = F.affine_grid(theta1, x.size())
        x = F.grid_sample(x, grid1)

        return x
    
    def stn2(self, x):
        xs2 = self.localization2(x)
        xs2 = xs2.view(-1, 150 * 2 * 2)
        theta2 = self.fc_loc2(xs2)
        theta2 = theta2.view(-1, 2, 3)

        grid2 = F.affine_grid(theta2, x.size())
        x = F.grid_sample(x, grid2)

        return x
    

    def forward(self, x):
        # transform the input
        x = self.stn1(x)
        x = self.batch_norm_2d1(self.max_pool2d1(nn.functional.leaky_relu(self.conv1(x))))
        x = self.dropout(x)
        x = self.batch_norm_2d2(self.max_pool2d2(nn.functional.leaky_relu(self.conv2(x))))
        x = self.dropout(x)
        x = self.stn2(x)
        x = self.batch_norm_2d3(self.max_pool2d3(nn.functional.leaky_relu(self.conv3(x))))
        x = self.dropout(x)
        x = x.view(-1, 250 * 4 * 4)
        x = self.fc1(x)
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
'''  
'''
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.max_pool2d1 = nn.MaxPool2d(2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.conv3 = nn.Conv2d(16, 400, kernel_size=5)
        self.fc1 = nn.Linear(800, nclasses)

    def forward(self, x):
        # Perform the usual forward pass
        x = F.relu(self.conv1(x))
        x = self.max_pool2d1(x)
        x = F.relu(self.conv2(x))
        x = self.max_pool2d1(x)
        x2 = x
        x = F.relu(self.conv3(x))
        x = x.view(-1, 400)
        x2 = x2.view(-1, 400)
        x3 = torch.cat([x, x2],1)
        x3 = F.dropout(x3, training=self.training)
        x3 = self.fc1(x3)
        return F.log_softmax(x3, dim=1)
'''
'''
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(20 * 5 * 5, 50)
        self.fc2 = nn.Linear(50, 43)

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 4 * 4, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.fill_(0)
        self.fc_loc[2].bias.data = torch.FloatTensor([1, 0, 0, 0, 1, 0])

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 4 * 4)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

    def forward(self, x):
        # transform the input
        x = self.stn(x)

        # Perform the usual forward pass
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 20 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
'''

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 100, kernel_size=3)
        self.batch_norm_2d1 = nn.BatchNorm2d(100)
        self.conv2 = nn.Conv2d(100, 150, kernel_size=4)
        self.batch_norm_2d2 = nn.BatchNorm2d(150)
        self.conv3 = nn.Conv2d(150, 250, kernel_size=3)
        self.batch_norm_2d3 = nn.BatchNorm2d(250)
        self.max_pool2d1 = nn.MaxPool2d(2, stride=2)
        self.fc1 = nn.Linear(2 * 2 * 250, 200)
        self.fc2 = nn.Linear(200, nclasses)
        self.dropout = nn.Dropout2d()
        
        self.localization = nn.Sequential(
            nn.Conv2d(1, 200, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(200, 300, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )
        
        '''
        self.localization2 = nn.Sequential(
            nn.Conv2d(150, 150, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout(0.5),
            nn.Conv2d(150, 150, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout(0.5)
        )
        '''
        '''
        self.localization = nn.Sequential(
            nn.Conv2d(3, 200, kernel_size=7),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout(0.5),
            nn.Conv2d(200, 300, kernel_size=5),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout(0.5)
        )
        '''
        # Regressor for the 3 * 2 affine matrix
        
        self.fc_loc = nn.Sequential(
            nn.Linear(300 * 4 * 4, 200),
            nn.ReLU(True),
            nn.Linear(200, 3 * 2)
        )
        '''
        self.fc_loc2 = nn.Sequential(
            nn.Linear(150, 150),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(150, 3 * 2)
        )
        '''
        '''
        self.fc_loc = nn.Sequential(
            nn.Linear(300 * 4 * 4, 200),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(200, 3 * 2)
        )
        '''
        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.fill_(0)
        self.fc_loc[2].bias.data = torch.FloatTensor([1, 0, 0, 0, 1, 0])
        '''
        self.fc_loc2[3].weight.data.fill_(0)
        self.fc_loc2[3].bias.data = torch.FloatTensor([1, 0, 0, 0, 1, 0])
        '''
    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 300 * 4 * 4)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x
    '''
    def stn2(self, x):
        xs = self.localization2(x)
        xs = xs.view(-1, 150)
        theta = self.fc_loc2(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x
    '''
    def forward(self, x):
        x = self.stn(x)
        # Perform the usual forward pass
        x = self.batch_norm_2d1(self.max_pool2d1(nn.functional.leaky_relu(self.conv1(x))))
        #x = self.max_pool2d1(nn.functional.leaky_relu(self.conv1(x)))
        #x = self.dropout(x)
        x = self.batch_norm_2d2(self.max_pool2d1(nn.functional.leaky_relu(self.conv2(x))))
        #x = self.max_pool2d1(nn.functional.leaky_relu(self.conv2(x)))
        #x = self.dropout(x)
        #x = self.stn2(x)
        x = self.batch_norm_2d3(self.max_pool2d1(nn.functional.leaky_relu(self.conv3(x))))
        #x = self.max_pool2d1(nn.functional.leaky_relu(self.conv3(x)))
        #x = self.dropout(x)
        #x = x.view(-1, 2 * 2 * 250)
        x = x.view(-1, 2 * 2 * 250)
        x = self.dropout(F.relu((self.fc1(x))))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

'''
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(500, 50)
        self.fc2 = nn.Linear(50, nclasses)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 500)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)
'''
#model = inception_v3_google(num_classes=43, pretrained=False)