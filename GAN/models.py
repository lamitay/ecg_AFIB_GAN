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
import torch.nn as nn
import torchvision.datasets as dataset

###### taken from https://github.com/LixiangHan/GANs-for-1D-Signal/blob/main/dcgan.py
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class DCDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            # input 1824
            nn.Conv1d(1, 64, kernel_size=5, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size 912
            nn.Conv1d(64, 128, kernel_size=15, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # state size 456
            nn.Conv1d(128, 256, kernel_size=20,
                      stride=2, padding=1, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # state size 228
            nn.Conv1d(256, 512, kernel_size=40,
                      stride=2, padding=1, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # state size 114
            nn.Conv1d(512, 1, kernel_size=70, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x, y=None):
        x = self.main(x)
        return x.squeeze(-1)


class DCGenerator(nn.Module):
    def __init__(self, nz):
        super().__init__()
        self.nz=nz
        self.main = nn.Sequential(
            nn.ConvTranspose1d(nz, 512, 70, 1, 0, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(True),

            nn.ConvTranspose1d(512, 256, 40, 2, 1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(True),

            nn.ConvTranspose1d(256, 128, 20, 2, 1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(True),

            nn.ConvTranspose1d(128, 64, 15, 2, 1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(True),

            nn.ConvTranspose1d(64, 1, 10, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        x = x.view(-1, self.nz, 1)
        x = self.main(x)
        return x


if __name__ == '__main__':
    from torchsummary import summary
    noise = torch.rand(256,1,100)
    # Gen = DCGenerator(signal_length=1500, noise_length=100)
    # Dec = DCDiscriminator(signal_length=1500)
    Gen = DCGenerator(nz=100)
    Dec = DCDiscriminator()
    print(summary(Gen, (1,100), device='cpu'))
    print(Gen)
    print(Dec)
    gen_out = Gen(noise)
    dec_output = Dec(gen_out)
