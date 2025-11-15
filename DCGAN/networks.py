import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, channels_img, feature_d):
        super(Discriminator, self).__init__()
        # Input: N x channels_img x 64 x 64, 
        self.net = nn.Sequential(
            nn.Conv2d(
                channels_img, feature_d, kernel_size=4, stride=2, padding=1
            ), # 32 x 32
            nn.LeakyReLU(0.2),
            self._conv2d_block(feature_d, feature_d * 2, 4, 2, 1), # 16 x 16
            self._conv2d_block(feature_d * 2, feature_d * 4, 4, 2, 1), # 8 x 8
            self._conv2d_block(feature_d * 4, feature_d * 8, 4, 2, 1), # 4 x 4
            nn.Conv2d(feature_d * 8, 1, kernel_size=4, stride=2, padding=0), # 1 x 1
            nn.Sigmoid(),
        )
        
    def _conv2d_block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )
    
    def forward(self, x):
        return self.net(x)
    
class Generator(nn.Module):
    def __init__(self, z_dim, channels_img, feature_g):
        super(Generator, self).__init__()
        # Input: N x z_dim x 1 x 1
        self.net = nn.Sequential(
            self._deconv2d_block(z_dim, feature_g * 16, 4, 1, 0), # 4 x 4
            self._deconv2d_block(feature_g * 16, feature_g * 8, 4, 2, 1), # 8 x 8
            self._deconv2d_block(feature_g * 8, feature_g * 4, 4, 2, 1), # 16 x 16
            self._deconv2d_block(feature_g * 4, feature_g * 2, 4, 2, 1), # 32 x 32
            nn.ConvTranspose2d(
                feature_g * 2, channels_img, kernel_size=4, stride=2, padding=1
            ), # 64 x 64
            nn.Tanh()
        )
        
    def _deconv2d_block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.05),
        )
        
    def forward(self, x):
        return self.net(x)
    
def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

def test():
    N, in_channels, H, W = 8, 3, 64, 64
    z_dim = 100
    feature_d = 16
    feature_g = 16
    
    x = torch.randn((N, in_channels, H, W))
    
    disc = Discriminator(in_channels, feature_d)
    initialize_weights(disc)
    assert disc(x).shape == (N, 1, 1, 1)
    
    noise = torch.randn((N, z_dim, 1, 1))
    gen = Generator(z_dim, in_channels, feature_g)
    assert gen(noise).shape == (N, in_channels, H, W)
    
    print("All tests passed!")