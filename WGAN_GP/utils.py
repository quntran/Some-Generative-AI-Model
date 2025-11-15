import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

# 'Agg': used to generate plots as images and save them to a file instead of rendering to screen
matplotlib.use('Agg')

def save_graph(G_losses, D_losses, generated_img, real_img, runs_dir):
    GRAPH_FILE = os.path.join(runs_dir, "graph.png")
    REAL_FILE = os.path.join(runs_dir, "real.png")
    FAKE_FILE = os.path.join(runs_dir, "fake.png")
    
    # Plot losses
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="Generator Loss")
    plt.plot(D_losses, label="Discriminator Loss")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(GRAPH_FILE)
    
    # Save generated and real images
    # 8 columns and 4 rows for 32 images
    plt.figure(figsize=(10, 5))
    plt.axis("off")
    plt.title("Generated Images")
    for i in range(32):
        plt.subplot(4, 8, i + 1)
        # rgb image
        plt.imshow(generated_img[i].detach().cpu().numpy().transpose(1, 2, 0), cmap="gray")
        plt.axis("off")
    plt.savefig(FAKE_FILE)
    
    plt.figure(figsize=(10, 5))
    plt.title("Real Images")
    for i in range(32):
        plt.subplot(4, 8, i + 1)
        plt.imshow(real_img[i].detach().cpu().numpy().transpose(1, 2, 0), cmap="gray")
        plt.axis("off")
    plt.savefig(REAL_FILE)
    
    # Close the plots to free up resources
    plt.close("all")
    
def save_checkpoint(model, optimizer, epoch, losses, filename):
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "losses": losses,
    }
    torch.save(checkpoint, filename)
    
def load_checkpoint(filename):
    checkpoint = torch.load(filename)
    return checkpoint["model"], checkpoint["optimizer"], checkpoint["epoch"], checkpoint["losses"]