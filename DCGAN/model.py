import torch
import yaml
import os
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from GAN.DCGAN.networks import Discriminator, Generator, initialize_weights
from GAN.DCGAN.utils import save_graph, save_checkpoint, load_checkpoint

class DCGAN():
    def __init__(self, hyperparameter_set, load_model=True):
        with open("GAN/DCGAN/hyperparameters.yaml", "r") as file:
            all_hyperparameter_sets = yaml.safe_load(file)
            hyperparameters = all_hyperparameter_sets[hyperparameter_set]
        
        # Load hyperparameters
        self.hyperparameter_set = hyperparameter_set
        self.D_LEARNING_RATE = hyperparameters["D_LEARNING_RATE"]
        self.G_LEARNING_RATE = hyperparameters["G_LEARNING_RATE"]
        self.BATCH_SIZE = hyperparameters["BATCH_SIZE"]
        self.IMAGE_SIZE = hyperparameters["IMAGE_SIZE"]
        self.CHANNELS_IMG = hyperparameters["CHANNELS_IMG"]
        self.NOISE_DIM = hyperparameters["NOISE_DIM"]
        self.NUM_EPOCHS = hyperparameters["NUM_EPOCHS"]
        self.FEATURES_DISC = hyperparameters["FEATURES_DISC"]
        self.FEATURES_GEN = hyperparameters["FEATURES_GEN"]
        self.LOG_DIR = hyperparameters["LOG_DIR"]
        self.BETAS = hyperparameters["BETAS"]
        self.epoch = 1
        
        # Initialize device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize networks
        self.generator = Generator(self.NOISE_DIM, self.CHANNELS_IMG, self.FEATURES_GEN).to(self.device)
        self.discriminator = Discriminator(self.CHANNELS_IMG, self.FEATURES_DISC).to(self.device)
        
        # Initialize weights
        initialize_weights(self.generator)
        initialize_weights(self.discriminator)
        
        # Initialize optimizer
        self.opt_gen = torch.optim.Adam(self.generator.parameters(), lr=self.G_LEARNING_RATE, betas=self.BETAS)
        self.opt_disc = torch.optim.Adam(self.discriminator.parameters(), lr=self.D_LEARNING_RATE, betas=self.BETAS)
        
        # Initialize criterion
        self.criterion = torch.nn.BCELoss()
        
        # Initialize fixed noise for visualization
        self.fixed_noise = torch.randn(32, self.NOISE_DIM, 1, 1).to(self.device)
        
        # Initialize loss lists
        self.G_losses = []
        self.D_losses = []
        
        # Initialize data loader
        self.loader = None
        
        # Initialize run directory
        self.runs_dir = "GAN/DCGAN/" + self.LOG_DIR
        os.makedirs(self.runs_dir, exist_ok=True)
        
        # Load checkpoint if available
        self.gen_checkpoint_file = os.path.join(self.runs_dir, "gen_checkpoint.pth")
        self.disc_checkpoint_file = os.path.join(self.runs_dir, "disc_checkpoint.pth")
        
        if os.path.exists(self.gen_checkpoint_file) and os.path.exists(self.disc_checkpoint_file) and load_model:
            gen_checkpoint = load_checkpoint(self.gen_checkpoint_file)
            disc_checkpoint = load_checkpoint(self.disc_checkpoint_file)
            
            gen_state_dict, gen_opt_state_dict, self.epoch, self.G_losses = gen_checkpoint
            disc_state_dict, disc_opt_state_dict, _, self.D_losses = disc_checkpoint
            
            self.generator.load_state_dict(gen_state_dict)
            self.opt_gen.load_state_dict(gen_opt_state_dict)
            
            self.discriminator.load_state_dict(disc_state_dict)
            self.opt_disc.load_state_dict(disc_opt_state_dict)
            
            print("Checkpoint loaded")
            
    def load_data(self, root_dir):
        transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((self.IMAGE_SIZE, self.IMAGE_SIZE)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    [0.5 for _ in range(self.CHANNELS_IMG)], [0.5 for _ in range(self.CHANNELS_IMG)]
                ),
            ]
        )
        
        dataset = datasets.ImageFolder(root=root_dir, transform=transforms)
        
        self.loader = DataLoader(dataset, batch_size=self.BATCH_SIZE, shuffle=True)
            
    def train(self):
        self.generator.train()
        self.discriminator.train()
        
        for epoch in range(self.epoch, self.NUM_EPOCHS + 1):
            for batch_idx, (real, _) in enumerate(self.loader):
                real = real.to(self.device)
                noise = torch.randn(self.BATCH_SIZE, self.NOISE_DIM, 1, 1).to(self.device)
                
                # Train discriminator
                self.opt_disc.zero_grad()
                fake = self.generator(noise)
                disc_real = self.discriminator(real).view(-1)
                lossD_real = self.criterion(disc_real, torch.ones_like(disc_real))
                disc_fake = self.discriminator(fake.detach()).view(-1)
                lossD_fake = self.criterion(disc_fake, torch.zeros_like(disc_fake))
                lossD = (lossD_real + lossD_fake) / 2
                lossD.backward()
                self.opt_disc.step()
                
                # Train generator
                self.opt_gen.zero_grad()
                output = self.discriminator(fake).view(-1)
                lossG = self.criterion(output, torch.ones_like(output))
                lossG.backward()
                self.opt_gen.step()
                
                # Save losses
                self.D_losses.append(lossD.item())
                self.G_losses.append(lossG.item())
                
                if batch_idx % 2 == 0:
                    print(f"Epoch [{epoch}/{self.NUM_EPOCHS}] - Iter {batch_idx}/{len(self.loader)}")
                    print(f"Loss D: {lossD:.4f} - Loss G: {lossG:.4f}")
                    
                    # Save generated images and checkpoint
                    save_graph(self.G_losses, self.D_losses, (self.generator(self.fixed_noise) + 1) / 2, (real + 1) / 2, self.runs_dir)
                    self.save_checkpoint()
            
            self.epoch = epoch
            
        print("Training complete")
        
    def random_sample(self, num_samples):
        samples = None
        with torch.no_grad():
            noise = torch.randn(num_samples, self.NOISE_DIM, 1, 1).to(self.device)
            samples = self.generator(noise)
        return samples / 2 + 1
    
    def sample_from_noise(self, noise):
        samples = None
        with torch.no_grad():
            samples = self.generator(noise)
        return samples / 2 + 1
            
    def save_checkpoint(self):
        save_checkpoint(self.generator, self.opt_gen, self.epoch, self.G_losses, self.gen_checkpoint_file)
        save_checkpoint(self.discriminator, self.opt_disc, self.epoch, self.D_losses, self.disc_checkpoint_file)
            
            