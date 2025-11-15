# Generative AI Models - GAN Implementations

A collection of Generative Adversarial Network (GAN) implementations in PyTorch. This repository contains three popular GAN architectures for generating synthetic images.

## 📋 Overview

This project implements three different GAN variants:

- **DCGAN** (Deep Convolutional GAN) - A stable architecture using deep convolutional networks
- **CGAN** (Conditional GAN) - Allows conditional generation based on class labels
- **WGAN-GP** (Wasserstein GAN with Gradient Penalty) - Improved training stability using Wasserstein distance

## 🚀 Features

- Modular architecture with separate implementations for each GAN variant
- Configurable hyperparameters via YAML files
- Checkpoint support for resuming training
- TensorBoard integration for monitoring training progress
- Image generation and visualization utilities
- Support for custom datasets

## 📦 Installation

### Prerequisites

- Python 3.7+
- CUDA-capable GPU (recommended for training)

### Dependencies

Install the required packages:

```bash
pip install torch torchvision pyyaml matplotlib numpy tqdm tensorboard
```

Or if you have a `requirements.txt`:

```bash
pip install -r requirements.txt
```

## 📂 Project Structure

```
.
├── DCGAN/              # Deep Convolutional GAN implementation
│   ├── model.py        # DCGAN model class
│   ├── networks.py     # Generator and Discriminator networks
│   ├── training.py     # Training script
│   ├── utils.py        # Utility functions
│   └── hyperparameters.yaml  # Configuration file
├── CGAN/               # Conditional GAN implementation
│   ├── model.py        # CGAN model class
│   ├── networks.py     # Generator and Critic networks
│   ├── training.py     # Training script
│   ├── utils.py        # Utility functions
│   └── hyperparameters.yaml  # Configuration file
├── WGAN_GP/            # Wasserstein GAN with Gradient Penalty
│   ├── model.py        # WGAN-GP model class
│   ├── networks.py     # Generator and Critic networks
│   ├── training.py     # Training script
│   ├── utils.py        # Utility functions
│   └── hyperparameters.yaml  # Configuration file
└── README.md
```

## 🎯 Usage

### Training a Model

Each model can be trained by running its respective training script. First, configure the hyperparameters in the `hyperparameters.yaml` file for the desired model.

#### DCGAN Example

```python
from GAN.DCGAN.model import DCGAN

# Initialize the model
dcgan = DCGAN(hyperparameter_set="config_1", load_model=False)

# Load your dataset
dcgan.load_data(root_dir="path/to/your/dataset")

# Train the model
dcgan.train()
```

#### CGAN Example

```python
from GAN.CGAN.model import CGAN

# Initialize the model
cgan = CGAN(hyperparameter_set="config_1", load_model=False)

# Load your dataset
cgan.load_data(root_dir="path/to/your/dataset")

# Train the model
cgan.train()
```

#### WGAN-GP Example

```python
from GAN.WGAN_GP.model import WGAN_GP

# Initialize the model
wgan_gp = WGAN_GP(hyperparameter_set="config_1", load_model=False)

# Load your dataset
wgan_gp.load_data(root_dir="path/to/your/dataset")

# Train the model
wgan_gp.train()
```

### Generating Samples

After training, you can generate samples using:

```python
# Generate random samples
samples = model.random_sample(num_samples=16)

# Or generate from specific noise
import torch
noise = torch.randn(16, noise_dim, 1, 1).to(device)
samples = model.sample_from_noise(noise)
```

### Resuming Training

To resume training from a checkpoint, set `load_model=True`:

```python
model = DCGAN(hyperparameter_set="config_1", load_model=True)
model.load_data(root_dir="path/to/your/dataset")
model.train()
```

## ⚙️ Configuration

Each model has a `hyperparameters.yaml` file with configurable parameters:

- `D_LEARNING_RATE` - Discriminator/Critic learning rate
- `G_LEARNING_RATE` - Generator learning rate
- `BATCH_SIZE` - Training batch size
- `IMAGE_SIZE` - Input image size
- `CHANNELS_IMG` - Number of image channels (3 for RGB, 1 for grayscale)
- `NOISE_DIM` - Dimension of the input noise vector
- `NUM_EPOCHS` - Number of training epochs
- `FEATURES_DISC/FEATURES_CRITIC` - Feature map size in discriminator/critic
- `FEATURES_GEN` - Feature map size in generator
- `LOG_DIR` - Directory for saving logs and checkpoints

## 📊 Monitoring Training

Training progress is saved in the log directory specified in the configuration. You can visualize the training using the saved graphs and checkpoints.

## 🤝 Contributing

Contributions are welcome! Feel free to:

- Report bugs
- Suggest new features
- Submit pull requests

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Quang Tran**

## 🙏 Acknowledgments

- Original DCGAN paper: [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434)
- Original WGAN-GP paper: [Improved Training of Wasserstein GANs](https://arxiv.org/abs/1704.00028)
- Conditional GAN paper: [Conditional Generative Adversarial Nets](https://arxiv.org/abs/1411.1784)