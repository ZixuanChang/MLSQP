import numpy as np
import torch
import torch.nn as nn
from MLSQP import MLsqp
from MLSQP import Net

class Act_op(nn.Module):
    def __init__(self):
        super(Act_op, self).__init__()
    def forward(self, x):
        x = x * torch.sigmoid(x)
        return x
swish = Act_op()
class Net(Net):
    def __init__(self, input_size, output_size):
        super(Net, self).__init__()

        # 第一个隐藏层
        self.fc1 = nn.Linear(input_size, 256)
        self.hidden_layers_1 = nn.ModuleList([nn.Linear(256, 256) for _ in range(4)])

        # 第二个隐藏层
        self.fc2 = nn.Linear(256, 128)
        self.hidden_layers_2 = nn.ModuleList([nn.Linear(128, 128) for _ in range(4)])

        # 输出层
        self.fc_out = nn.Linear(128, output_size)
    def forward(self, x):
        x = swish(self.fc1(x))
        for layer in self.hidden_layers_1:
            x = swish(layer(x))
        x = swish(self.fc2(x))
        for layer in self.hidden_layers_2:
            x = swish(layer(x))
        output = self.fc_out(x)

        return output



class OPT_Griewank(MLsqp):
    def Objective(self,x,y):
        return y

x_mean = [0 for _ in range(5)]
x_std = [1 for _ in range(5)]
y_mean = [0]
y_std = [1]
mean = [x_mean, y_mean]
std = [np.array(x_std), np.array(y_std)]
opt = OPT_Griewank()
ip = torch.tensor([2 for _ in range(5)])
opt.setInitialPoint(ip)
lb = np.array([-2 for _ in range(5)] + [-np.inf])
ub = np.array([2 for _ in range(5)] + [np.inf])
A = np.array([[0 for _ in range(5)]])
G = np.array([[0 for _ in range(5)]])
h = np.array([0])
b = np.array([0])
opt.setConstraints(G=G, h=h, A=A, b=b, lb=lb, ub=ub)
opt.setMLmodel(mean=mean, std=std, model_path='./241210_griewank.pth')
opt.sqp_solve()
