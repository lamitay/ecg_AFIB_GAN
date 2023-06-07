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


class DCGenerator(nn.Module):
    def __init__(self, signal_length, noise_length):
        super(DCGenerator, self).__init__()
        ngf = 64
        self.noise_length = noise_length
        self.main = nn.Sequential(
            # shape in = [N, 50, 1]
            nn.ConvTranspose1d(noise_length, ngf * 2, 4, 1, 0, bias=False),
            nn.BatchNorm1d(ngf * 2),
            nn.ReLU(True),
            # shape in = [N, 64*4, 4]
            nn.ConvTranspose1d(ngf * 2, ngf * 4, 4, 1, 0, bias=False),
            nn.BatchNorm1d(ngf * 4),
            nn.ReLU(True),
            # shape in = [N, 64*2, 7]
            nn.ConvTranspose1d(ngf * 4, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm1d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose1d(ngf * 8, ngf * 16, 3, 2, 1, bias=False),
            nn.BatchNorm1d(ngf * 16),
            nn.ReLU(True),
            nn.ConvTranspose1d(ngf * 16, ngf * 8, 4, 2, 2, bias=False),
            nn.BatchNorm1d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose1d(ngf * 8, ngf * 4, 4, 2, 2, bias=False),
            nn.BatchNorm1d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose1d(ngf * 4, ngf, 4, 2, 2, bias=False),
        )
        self.fc = nn.Linear(12928, signal_length)

    def forward(self, x):
        x = x.view(-1, self.noise_length, 1)
        x = self.main(x)
        x = x.view(-1,12928)
        x = self.fc(x)
        return x


class DCDiscriminator(nn.Module):
    def __init__(self, signal_length):
        super(DCDiscriminator, self).__init__()
        ndf = 32
        self.input_size = signal_length
        self.main = nn.Sequential(
        # input is (nc) x 64 x 64
        nn.Conv1d(in_channels=1, out_channels=ndf, kernel_size=4, stride=2, padding=1, bias=False),
        nn.LeakyReLU(0.2, inplace=True),
        # state size. (ndf) x 32 x 32
        nn.Conv1d(ndf, ndf * 2, 4, 2, 1, bias=False),
        nn.BatchNorm1d(ndf * 2),
        nn.LeakyReLU(0.2, inplace=True),
        # state size. (ndf*2) x 16 x 16
        nn.Conv1d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
        nn.BatchNorm1d(ndf * 4),
        nn.LeakyReLU(0.2, inplace=True),
        # state size. (ndf*4) x 8 x 8
        nn.Conv1d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
        nn.BatchNorm1d(ndf * 8),
        nn.LeakyReLU(0.2, inplace=True),

        nn.Conv1d(ndf * 8, ndf , 4, 2, 1, bias=False),
        nn.BatchNorm1d(ndf ),
        nn.LeakyReLU(0.2, inplace=True),

        # state size. (ndf*8) x 4 x 4
        nn.Conv1d(ndf , 1, 5, 2, 0, bias=False),
        nn.Linear(115,1),
        nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(-1, 1, self.input_size)
        return self.main(x)
        
if __name__ == '__main__':
    noise = torch.rand(1,1000)
    Gen = DCGenerator(signal_length=7500, noise_length=1000)
    Dec = DCDiscriminator(signal_length=7500)
    print(Gen)
    print(Dec)
    gen_out = Gen(noise)
    dec_output = Dec(gen_out)
