import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):

    def __init__(self, latent_size):
        super(Generator, self).__init__()
        self.latent_size = latent_size
        self.leaky_value = 0.1

        self.output_bias = nn.Parameter(torch.zeros(3, 64, 64), requires_grad=True)

        self.main = nn.Sequential(
            nn.ConvTranspose2d(self.latent_size, 256, 4, stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(self.leaky_value, inplace=True),

            nn.ConvTranspose2d(256, 128, 4, stride=2, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(self.leaky_value, inplace=True),

            nn.ConvTranspose2d(128, 64, 4, stride=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(self.leaky_value, inplace=True),

            nn.ConvTranspose2d(64, 32, 4, stride=2, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(self.leaky_value, inplace=True),

            nn.ConvTranspose2d(32, 32, 5, stride=2, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(self.leaky_value, inplace=True),

            nn.ConvTranspose2d(32, 32, 4, stride=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(self.leaky_value, inplace=True),

            nn.ConvTranspose2d(32, 3, 3, stride=1, bias=False)
        )

    def forward(self, input):
        output = self.main(input)
        output = torch.sigmoid(output + self.output_bias)
        return output


class Encoder(nn.Module):

    def __init__(self, latent_size, noise=False):
        super(Encoder, self).__init__()
        self.latent_size = latent_size
        self.leaky_value = 0.1

        self.main1 = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(self.leaky_value, inplace=True),

            nn.Conv2d(32, 64, 4, stride=2, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(self.leaky_value, inplace=True),

            nn.Conv2d(64, 128, 4, stride=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(self.leaky_value, inplace=True),

            nn.Conv2d(128, 256, 4, stride=2, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(self.leaky_value, inplace=True)
        )

        self.main2 = nn.Sequential(
            nn.Conv2d(256, 512, 4, stride=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(self.leaky_value, inplace=True)
        )

        self.main3 = nn.Sequential(
            nn.Conv2d(512, 512, 5, stride=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(self.leaky_value, inplace=True)
        )

        self.main4 = nn.Sequential(
            nn.Conv2d(512, self.latent_size, 5, stride=1, bias=True)
        )

    def forward(self, input):
        batch_size = input.size()[0]
        x1 = self.main1(input)
        x2 = self.main2(x1)
        x3 = self.main3(x2)
        output = self.main4(x3)
        return output


class Discriminator(nn.Module):

    def __init__(self, latent_size, dropout, output_size=1):
        super(Discriminator, self).__init__()
        self.latent_size = latent_size
        self.dropout = dropout
        self.output_size = output_size
        self.leaky_value = 0.1

        self.infer_x = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, bias=True),
            nn.LeakyReLU(self.leaky_value, inplace=True),
            nn.Dropout2d(p=self.dropout),

            nn.Conv2d(32, 64, 4, stride=2, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(self.leaky_value, inplace=True),
            nn.Dropout2d(p=self.dropout),

            nn.Conv2d(64, 128, 4, stride=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(self.leaky_value, inplace=True),
            nn.Dropout2d(p=self.dropout),

            nn.Conv2d(128, 256, 4, stride=2, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(self.leaky_value, inplace=True),
            nn.Dropout2d(p=self.dropout),

            nn.Conv2d(256, 512, 4, stride=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(self.leaky_value, inplace=True),
            nn.Dropout2d(p=self.dropout)
        )

        self.infer_z = nn.Sequential(
            nn.Conv2d(self.latent_size, 512, 1, stride=1, bias=False),
            nn.LeakyReLU(self.leaky_value, inplace=True),
            nn.Dropout2d(p=self.dropout),

            nn.Conv2d(512, 512, 1, stride=1, bias=False),
            nn.LeakyReLU(self.leaky_value, inplace=True),
            nn.Dropout2d(p=self.dropout)
        )

        self.infer_joint = nn.Sequential(
            nn.Conv2d(1024, 1024, 1, stride=1, bias=True),
            nn.LeakyReLU(self.leaky_value, inplace=True),
            nn.Dropout2d(p=self.dropout),

            nn.Conv2d(1024, 1024, 1, stride=1, bias=True),
            nn.LeakyReLU(self.leaky_value, inplace=True),
            nn.Dropout2d(p=self.dropout)
        )

        self.final = nn.Conv2d(1024, self.output_size, 1, stride=1, bias=True)

    def forward(self, x, z):
        output_x = self.infer_x(x)
        output_z = self.infer_z(z)
        output_features = self.infer_joint(torch.cat([output_x, output_z], dim=1))
        output = self.final(output_features)
        if self.output_size == 1:
            output = torch.sigmoid(output)
        return output.squeeze()