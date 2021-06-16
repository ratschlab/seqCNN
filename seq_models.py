import torch
from torch import nn


class SeqCNN(nn.Module):
    def __init__(self,in_channels, groups, kernel, num_layers, stride, channels):
        super(SeqCNN, self).__init__()
        g = groups 
        padding=int(kernel/2)
        layers = []
        width = channels * in_channels  # use (x channels) number of channels
        for l in range(num_layers):
            l1, l2 = width * 2**(l), width * 2**(l+1)
            if l==0:
                l1 = in_channels
            if groups == 0: # every input channel => connect to 2 output channels
                g = l1
            layers.append(nn.BatchNorm1d(l1))
            layers.append(nn.Conv1d(l1,l2, kernel, stride=stride, padding=padding, groups=g))
            layers.append(nn.ReLU())
        self.layers = layers
        self.seq = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.seq(x)