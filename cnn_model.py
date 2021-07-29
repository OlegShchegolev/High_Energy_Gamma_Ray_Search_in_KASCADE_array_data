import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import one_hot


def convrelu(in_channels, out_channels, kernel):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(out_channels)
    )
    
class CNN_B(nn.Module):
    def __init__(self, n_features, n_mat, n_class):
        super().__init__()
        
        self.conv1 = convrelu(n_mat, 32, 3)
        self.conv2 = convrelu(32, 64, 3)
        self.conv3 = convrelu(64, 32, 3)
        self.fc1 = nn.Linear(32*10*10+n_features, 100)
        self.fc2 = nn.Linear(100, 30)
        self.fc3 = nn.Linear(30, n_class)
        
        self.weights_init()
        
    def forward(self, x_features, x_mat):
        s0 = x_mat.shape[0]
        x_mat = self.conv1(x_mat)
        x_mat = self.conv2(x_mat)
        x_mat = self.conv3(x_mat)
        x_mat = x_mat.view(s0,-1)
        x = torch.cat((x_features, x_mat), dim=1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training, p=0.3)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, training=self.training, p=0.3)
        x = self.fc3(x)
        return x
    
    def weights_init(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
