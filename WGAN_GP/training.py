import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from GAN.WGAN_GP.model import WGAN_GP


# dataset = datasets.MNIST(root="lessons/data", train=True, transform=transforms, download=True)

model = WGAN_GP("config_1", load_model=True)

model.load_data("datasets/celeb_dataset")

model.train()