import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.utils.tensorboard import SummaryWriter
from GAN.CGAN.model import CGAN



model = CGAN("config_1", load_model=True)


transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize((model.IMAGE_SIZE, model.IMAGE_SIZE)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            [0.5 for _ in range(model.CHANNELS_IMG)], [0.5 for _ in range(model.CHANNELS_IMG)]
        ),
    ]
)

dataset = datasets.MNIST(root="./datasets", train=True, transform=transforms, download=True)

model.load_data(dataset)

model.train()