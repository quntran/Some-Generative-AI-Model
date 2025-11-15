import torch
import yaml
import os
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import tqdm

from GAN.CGAN.networks import Critic, Generator, initialize_weights
from GAN.CGAN.utils import save_graph, save_checkpoint, load_checkpoint

class CGAN():
    def __init__(self, hyperparameter_set, load_model=True):
        with open("GAN/CGAN/hyperparameters.yaml", "r") as file:
            all_hyperparameter_sets = yaml.safe_load(file)
            hyperparameters = all_hyperparameter_sets[hyperparameter_set]
        
        # Load hyperparameters
        self.load_model = load_model
        self.hyperparameter_set = hyperparameter_set
        self.D_LEARNING_RATE = hyperparameters["D_LEARNING_RATE"]
        self.G_LEARNING_RATE = hyperparameters["G_LEARNING_RATE"]
        self.BATCH_SIZE = hyperparameters["BATCH_SIZE"]
        self.IMAGE_SIZE = hyperparameters["IMAGE_SIZE"]
        self.NUM_CLASSES = hyperparameters["NUM_CLASSES"]
        self.GEN_EMBEDDING = hyperparameters["GEN_EMBEDDING"]
        self.CHANNELS_IMG = hyperparameters["CHANNELS_IMG"]
        self.NOISE_DIM = hyperparameters["NOISE_DIM"]
        self.NUM_EPOCHS = hyperparameters["NUM_EPOCHS"]
        self.FEATURES_CRITIC = hyperparameters["FEATURES_CRITIC"]
        self.FEATURES_GEN = hyperparameters["FEATURES_GEN"]
        self.CRITIC_ITERATIONS = hyperparameters["CRITIC_ITERATIONS"]
        self.LOG_DIR = hyperparameters["LOG_DIR"]
        self.BETAS = hyperparameters["BETAS"]
        self.LAMBDA_GP = hyperparameters["LAMBDA_GP"]
        self.epoch = 1
        
        # Initialize device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize networks
        self.generator = Generator(self.NOISE_DIM, self.CHANNELS_IMG, self.FEATURES_GEN, self.NUM_CLASSES, self.GEN_EMBEDDING).to(self.device)
        self.critic = Critic(self.CHANNELS_IMG, self.FEATURES_CRITIC, self.NUM_CLASSES, self.IMAGE_SIZE).to(self.device)
        
        # Initialize weights
        initialize_weights(self.generator)
        initialize_weights(self.critic)
        
        # Initialize optimizer
        self.opt_gen = torch.optim.Adam(self.generator.parameters(), lr=self.G_LEARNING_RATE, betas=self.BETAS)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=self.D_LEARNING_RATE, betas=self.BETAS)
        
        # Initialize fixed noise for visualization
        self.fixed_noise = torch.randn(32, self.NOISE_DIM, 1, 1).to(self.device)
        
        # Initialize loss lists
        self.G_losses = []
        self.C_losses = []
        
        # Initialize data loader
        self.loader = None
        
        # Initialize run directory
        self.runs_dir = "GAN/CGAN/" + self.LOG_DIR
        os.makedirs(self.runs_dir, exist_ok=True)
        
        # Load checkpoint if available
        self.gen_checkpoint_file = os.path.join(self.runs_dir, "gen_checkpoint.pth")
        self.critic_checkpoint_file = os.path.join(self.runs_dir, "critic_checkpoint.pth")
        
        if os.path.exists(self.gen_checkpoint_file) and os.path.exists(self.critic_checkpoint_file) and load_model:
            gen_checkpoint = load_checkpoint(self.gen_checkpoint_file)
            critic_checkpoint = load_checkpoint(self.critic_checkpoint_file)
            
            gen_state_dict, gen_opt_state_dict, self.epoch, self.G_losses = gen_checkpoint
            critic_state_dict, critic_opt_state_dict, _, self.C_losses = critic_checkpoint
            
            self.generator.load_state_dict(gen_state_dict)
            self.opt_gen.load_state_dict(gen_opt_state_dict)
            
            self.critic.load_state_dict(critic_state_dict)
            self.opt_critic.load_state_dict(critic_opt_state_dict)
            
            print("Checkpoint loaded")
            
    def load_data(self, dataset):
        
        self.loader = DataLoader(dataset, batch_size=self.BATCH_SIZE, shuffle=True)
        
    def reload_dataset(self):
        if self.loader is None:
            raise ValueError("Data loader not initialized")
            
        self.loader = DataLoader(self.loader.dataset, batch_size=self.BATCH_SIZE, shuffle=True)
            
    def train(self):
        self.generator.train()
        self.critic.train()
        
        for epoch in range(self.epoch, self.NUM_EPOCHS + 1):
            for batch_idx, (real, labels) in enumerate(self.loader):
                real = real.to(self.device)
                noise = torch.randn(self.BATCH_SIZE, self.NOISE_DIM, 1, 1).to(self.device)
                labels = labels.to(self.device)
                fake = self.generator(noise, labels)

                self.opt_critic.zero_grad()
                critic_real = self.critic(real, labels)
                critic_fake = self.critic(fake.detach(), labels)
                gp = self.gradient_penalty(real, fake, labels)
                lossC = torch.mean(critic_fake) - torch.mean(critic_real) + self.LAMBDA_GP * gp
                lossC.backward(retain_graph=True)
                self.opt_critic.step()
                
                lossC = lossC.item()

                if batch_idx % self.CRITIC_ITERATIONS == 0:
                    # Train generator
                    self.opt_gen.zero_grad()
                    output = self.critic(fake, labels)
                    lossG = -torch.mean(output)
                    lossG.backward()
                    self.opt_gen.step()
                    
                    lossG = lossG.item()
                else:
                    lossG = self.G_losses[-1] if len(self.G_losses) > 0 else 0
                
                # Save losses
                if batch_idx % 5 == 0 and (batch_idx > 0 or self.load_model):
                    self.C_losses.append(lossC)
                    self.G_losses.append(lossG)
                    print(f"Epoch [{epoch}/{self.NUM_EPOCHS}] - Iter {batch_idx}/{len(self.loader)}")
                    print(f"Critic Loss: {lossC:.4f} - Generator Loss: {lossG:.4f}")
                    
                    # Save generated images and checkpoint
                    save_graph(self.G_losses, self.C_losses, (self.generator(self.fixed_noise) + 1) / 2, (real + 1) / 2, self.runs_dir)
                    self.save_checkpoint()
            
            self.epoch = epoch
            self.reload_dataset()
            
        print("Training complete")
        
    def gradient_penalty(self, real, fake, labels):
        BATCH_SIZE, C, H, W = real.shape
        
        epsilon = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(self.device)
        interpolated = epsilon * real + (1 - epsilon) * fake
        
        mixed_scores = self.critic(interpolated, labels)
        
        gradient = torch.autograd.grad(
            inputs=interpolated,
            outputs=mixed_scores,
            grad_outputs=torch.ones_like(mixed_scores),
            create_graph=True,
            retain_graph=True,
        )
        
        gradient = gradient[0].view(gradient[0].shape[0], -1)
        gradient_norm = gradient.norm(2, dim=1)
        return torch.mean((gradient_norm - 1) ** 2)
        
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
        save_checkpoint(self.critic, self.opt_critic, self.epoch, self.C_losses, self.critic_checkpoint_file)
            
            