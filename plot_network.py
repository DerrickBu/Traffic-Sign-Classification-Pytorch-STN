import torch
from torch.autograd import Variable

from model import Net
from visualize import make_dot

x = Variable(torch.randn(1,3,224,224))#change 12 to the channel number of network input
model = Net()
y = model(x)
g = make_dot(y)
g.view()