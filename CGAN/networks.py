import torch
import torch.nn as nn

class Critic(nn.Module):
    def __init__(self, channels_img, feature_d, num_classes, image_size):
        super(Critic, self).__init__()
        
        self.num_classes = num_classes
        self.image_size = image_size
        
        # Input: N x channels_img x 64 x 64,
        self.net = nn.Sequential(
            nn.Conv2d(
                channels_img + 1, feature_d, kernel_size=4, stride=2, padding=1
            ), # 32 x 32
            nn.LeakyReLU(0.2),
            self._conv2d_block(feature_d, feature_d * 2, 4, 2, 1), # 16 x 16
            self._conv2d_block(feature_d * 2, feature_d * 4, 4, 2, 1), # 8 x 8
            self._conv2d_block(feature_d * 4, feature_d * 8, 4, 2, 1), # 4 x 4
            nn.Conv2d(feature_d * 8, 1, kernel_size=4, stride=2, padding=0), # 1 x 1
            nn.Flatten(start_dim=1),
        )
        
        self.embed = nn.Embedding(num_classes, image_size * image_size)
        
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
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2),
        )
    
    def forward(self, x, labels):
        embedding = self.embed(labels).view(labels.shape[0], 1, self.image_size, self.image_size)
        x = torch.cat([x, embedding], dim=1)
        
        return self.net(x)
    
class Generator(nn.Module):
    def __init__(
            self, 
            channels_noise, 
            channels_img, 
            feature_g,
            num_classes,
            embed_size,
        ):
        super(Generator, self).__init__()
        self.num_classes = num_classes
        self.embed_size = embed_size
        
        # Input: N x channels_noise x 1 x 1
        self.net = nn.Sequential(
            self._deconv2d_block(channels_noise + embed_size, feature_g * 16, 4, 1, 0), # 4 x 4
            self._deconv2d_block(feature_g * 16, feature_g * 8, 4, 2, 1), # 8 x 8
            self._deconv2d_block(feature_g * 8, feature_g * 4, 4, 2, 1), # 16 x 16
            self._deconv2d_block(feature_g * 4, feature_g * 2, 4, 2, 1), # 32 x 32
            nn.ConvTranspose2d(
                feature_g * 2, channels_img, kernel_size=4, stride=2, padding=1
            ), # 64 x 64
            nn.Tanh()
        )
        
        self.embed = nn.Embedding(num_classes, embed_size)
        
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
        embedding = self.embed(x).view(x.shape[0], self.embed_size, 1, 1)
        x = torch.cat([x, embedding], dim=1)
        
        return self.net(x)
    
def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

def test():
    N, in_channels, H, W = 8, 3, 64, 64
    channels_noise = 100
    feature_d = 16
    feature_g = 16
    
    x = torch.randn((N, in_channels, H, W))
    
    critic = Critic(in_channels, feature_d)
    initialize_weights(critic)
    assert critic(x).shape == (N, 1, 1, 1)
    
    noise = torch.randn((N, channels_noise, 1, 1))
    gen = Generator(channels_noise, in_channels, feature_g)
    assert gen(noise).shape == (N, in_channels, H, W)
    
    print("All tests passed!")