import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, input_noise_size=100, out_signal_length=1500):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(input_noise_size, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc4 = nn.Linear(1024, out_signal_length)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.dropout(x, p=0.2)
        x = F.leaky_relu(self.fc3(x))
        x = F.dropout(x, p=0.2)
        x = self.fc4(x)
        return x.unsqueeze(1)


class Discriminator(nn.Module):
    def __init__(self, signal_length=1500):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(signal_length, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 1)

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
            # input 1824 - not 1500?
            nn.Conv1d(1, 64, kernel_size=10, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size 912 - 750?
            nn.Conv1d(64, 128, kernel_size=10, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # state size 456 - 375?
            nn.Conv1d(128, 256, kernel_size=10, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # state size 228 - 187.5?
            nn.Conv1d(256, 512, kernel_size=10, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # state size 114 - 93.75??
            nn.Conv1d(512, 1, kernel_size=10, stride=1, padding=0, bias=False),
            nn.AvgPool1d(kernel_size=79),
            nn.Sigmoid()
        )

    def forward(self, x, y=None):
        x = self.main(x)
        return x.squeeze(-1)


class DCGenerator(nn.Module):
    def __init__(self, nz):
        super().__init__()
        self.nz=nz
        self.layer1 = nn.Sequential(
            nn.ConvTranspose1d(nz, 512, 10, 1, 0, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(True))

        self.layer2 = nn.Sequential(
            nn.ConvTranspose1d(512, 256, 10, 5, 8, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(True))

        self.layer3 = nn.Sequential(
            nn.ConvTranspose1d(256, 128, 10, 4, 5, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(True))

        self.layer4 = nn.Sequential(
            nn.ConvTranspose1d(128, 64, 9, 5, 5, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(True))

        self.layer5 = nn.Sequential(
            nn.ConvTranspose1d(64, 1, 2, 2, 4, bias=False),
            nn.Tanh())

    def forward(self, x):
        x = x.view(-1, self.nz, 1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x


class DC_LSTM_Generator(nn.Module):
    def __init__(self, n_features = 1, hidden_dim = 50, seq_length = 1500, num_layers = 2, tanh_output = False):
        super(Generator,self).__init__()
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.seq_length = seq_length
        self.tanh_output = tanh_output
        
        self.conv1 = nn.Sequential(
                nn.ConvTranspose1d(n_features, 256, 4, 2, 1, bias=False),
                nn.BatchNorm1d(256),
                nn.ReLU(True))

        self.conv2 = nn.Sequential(
                nn.ConvTranspose1d(256, 512, 4, 2, 1, bias=False),
                nn.BatchNorm1d(512),
                nn.ReLU(True))

        self.conv3 = nn.Sequential(
                nn.ConvTranspose1d(512, 1024, 4, 2, 1, bias=False),
                nn.BatchNorm1d(1024),
                nn.ReLU(True))
        
        self.lstm = nn.LSTM(input_size = 1024, hidden_size = self.hidden_dim, 
                                    num_layers = self.num_layers,batch_first = True)
        
        if self.tanh_output == True:
            self.out = nn.Sequential(nn.Linear(self.hidden_dim,self.seq_length),nn.Tanh()) # to make sure the output is between 0 and 1 - removed ,nn.Sigmoid()
        else:
            self.out = nn.Linear(self.hidden_dim,self.seq_length) 
      
      
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.num_layers, batch_size, self.hidden_dim).zero_(),
                    weight.new(self.num_layers, batch_size, self.hidden_dim).zero_())
        return hidden
    
    def forward(self,x,hidden):
        x = self.conv1(x.view(-1, self.n_features, 1))
        x = self.conv2(x)
        x = self.conv3(x)
        x,hidden = self.lstm(x.transpose(1,2), hidden)  # transpose is used to swap the seq_len and feature_dim
        x = self.out(x.squeeze(1))

        return x.unsqueeze(1)


class DC_LSTM_Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=10, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(64, 128, kernel_size=10, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(128, 256, kernel_size=10, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(256, 512, kernel_size=10, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.lstm = nn.LSTM(512, 256, batch_first=True)

        self.linear = nn.Sequential(
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x, y=None):
        x = self.conv(x)
        x, _ = self.lstm(x.permute(0, 2, 1))  # LSTM expects input of shape (batch_size, seq_len, input_size)
        x = self.linear(x[:, -1, :])  # Use the output from the last time step
        return x



if __name__ == '__main__':
    from torchsummary import summary
    noise = torch.rand(256,1,100)
    # Gen = DCGenerator(signal_length=1500, noise_length=100)
    # Dec = DCDiscriminator(signal_length=1500)
    Gen = DCGenerator(nz=100)
    Dec = DCDiscriminator()
    # print(summary(Gen, (1,100), device='cpu'))
    print(Gen)
    print(Dec)
    gen_out = Gen(noise)
    dec_output = Dec(gen_out)
