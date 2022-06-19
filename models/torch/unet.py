import torch.nn as nn
import torch

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        num_neurons = [64, 128, 256, 512, 1024]

        self.e1 = self.make_elayer(in_channels, num_neurons[0])
        self.e2 = self.make_elayer(num_neurons[0], num_neurons[1])
        self.e3 = self.make_elayer(num_neurons[1], num_neurons[2])
        self.e4 = self.make_elayer(num_neurons[2], num_neurons[3])

        latent = []
        latent.append(nn.Conv2d(num_neurons[3], num_neurons[4], kernel_size=3, padding=1))
        latent.append(nn.ReLU())
        latent.append(nn.Conv2d(num_neurons[4], num_neurons[4], kernel_size=3, padding=1))
        latent.append(nn.ReLU())
        latent.append(nn.Upsample(scale_factor=2))
        self.latent = nn.Sequential(*latent)

        self.d1 = self.make_dlayer(num_neurons[4], num_neurons[3])
        self.d2 = self.make_dlayer(num_neurons[3], num_neurons[2])
        self.d3 = self.make_dlayer(num_neurons[2], num_neurons[1])
        self.d4 = self.make_dlayer(num_neurons[1], num_neurons[0])

        out = []
        out.append(nn.Conv2d(num_neurons[0], out_channels, kernel_size=1, padding=1))
        out.append(nn.Sigmoid())
        self.out = nn.Sequential(*out)

    
    def make_elayer(self, in_channels, out_channels):
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        layers.append(nn.MaxPool2d(kernel_size=2))
        layers.append(nn.Dropout(0.5))
        return nn.Sequential(*layers)


    def make_dlayer(self, in_channels, out_channels):
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        layers.append(nn.Upsample(scale_factor=2))
        layers.append(nn.Dropout(0.5))
        return nn.Sequential(*layers)


    def forward(self, x):
        y0 = self.e1(x)
        y1 = self.e2(y0)
        y2 = self.e3(y1)
        y3 = self.e4(y2)
        y = self.latent(y3)
        y = self.d1(torch.cat((y, y3), 1))
        y = self.d2(torch.cat((y, y2)))
        y = self.d3(torch.cat((y, y1)))
        y = self.d4(torch.cat((y, y0)))
        y = self.out(y)
        return y