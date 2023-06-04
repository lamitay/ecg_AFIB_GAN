import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, signal_length=7500):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(signal_length, 2500)
        self.fc2 = nn.Linear(2500, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 2500)
        self.fc5 = nn.Linear(2500, signal_length)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.dropout(x, p=0.2)
        x = F.leaky_relu(self.fc3(x))
        x = F.leaky_relu(self.fc4(x))
        x = F.dropout(x, p=0.2)
        x = self.fc5(x)
        return x.unsqueeze(1)


class Discriminator(nn.Module):
    def __init__(self, signal_length=7500):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(signal_length, 2500)
        self.fc2 = nn.Linear(2500, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 256)
        self.fc5 = nn.Linear(256, 1)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.dropout(x, p=0.2)
        x = F.leaky_relu(self.fc3(x))
        x = F.leaky_relu(self.fc4(x))
        x = F.dropout(x, p=0.2)
        x = torch.sigmoid(self.fc5(x))
        return x
