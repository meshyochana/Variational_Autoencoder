# Configuration settings for the model and training process
import torch

# Device configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Training parameters
BATCH_SIZE = 128
LEARNING_RATE = 0.001
EPOCHS = 10

# Model parameters
latent_dim = 3
