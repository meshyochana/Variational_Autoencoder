import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.optim import Adam

import matplotlib.pyplot as plt
import numpy as np
from config_mnist import device, BATCH_SIZE, LEARNING_RATE, latent_dim


class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dim=20):
        super(VariationalAutoencoder, self).__init__()

        # Encoder part
        self.encoder = nn.Sequential(
            nn.Linear(784, 400),
            nn.ReLU(),
            nn.Linear(400, 40)  # Outputs both mu and log_var (20 each)
        )

        # Mean and log variance layers
        self.fc_mu = nn.Linear(40, latent_dim)
        self.fc_log_var = nn.Linear(40, latent_dim)

        # Decoder part
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 784),
            nn.Sigmoid()  # Ensuring output is between 0 and 1
        )

    def encode(self, x):
        h1 = self.encoder(x.view(-1, 784))
        return self.fc_mu(h1), self.fc_log_var(h1)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var


def loss_function(recon_x, x, mu, log_var, flag_bce=True, flag_kld=True):
    BCE = flag_bce * F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = flag_kld * -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD

def train(model, data_loader, flag_bce=True, flag_kld=True):
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model.train()  
    train_loss = 0  

    for batch_idx, (data, _) in enumerate(data_loader):  
        data = data.to(device)  
        optimizer.zero_grad()  # Zero out gradients before backpropagation
        recon_batch, mu, log_var = model(data)  # Forward pass
        loss = loss_function(recon_batch, data, mu, log_var, flag_bce=flag_bce, flag_kld=flag_kld)  
        loss.backward()  # Backpropagation
        train_loss += loss.item()  # Sum up batch loss
        optimizer.step()  # Update model parameters
    return train_loss / len(data_loader.dataset)  # Average loss over the dataset

def visualize_reconstructions(model, data_loader, n=10):
    model.eval()
    with torch.no_grad():
        for i, (data, _) in enumerate(data_loader):
            data = data.to(device)
            reconstruction, _, _ = model(data)
            if i == 0:
                n = min(data.size(0), n)
                comparison = torch.cat([data[:n],
                                      reconstruction.view(BATCH_SIZE, 1, 28, 28)[:n]])
                break

    # Plotting
    fig, axes = plt.subplots(nrows=2, ncols=n, figsize=(10, 2))
    for i in range(n):
        axes[0, i].imshow(data[i].cpu().numpy().reshape(28, 28), cmap='gray')
        axes[0, i].axis('off')
        axes[1, i].imshow(reconstruction[i].cpu().numpy().reshape(28, 28), cmap='gray')
        axes[1, i].axis('off')
    plt.show()

def generate_new_digits(model, num_samples):
    model.eval()  # Put the model in inference mode
    with torch.no_grad():  # No need to track gradients
        # Sample random points in the latent space
        z = torch.randn(num_samples, latent_dim).to(device)
        # Generate digits from the latent vectors
        generated_digits = model.decoder(z)
    return z, generated_digits

def plot_loss(losses):
    fig, ax = plt.subplots(1,1, figsize=(10, 5))
    ax.plot(losses, "o-")
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    plt.show()

def plot_digits_and_latents(latents, digits, plot_samples=5):
    fig, axes = plt.subplots(nrows=2, ncols=plot_samples, figsize=(10, 4))

    for i in range(plot_samples):
        ax = axes[0, i]
        ax.imshow(digits[i].view(28, 28).cpu().numpy(), cmap='gray')  # Assuming digits are 28x28
        ax.axis('off')

        ax = axes[1, i]
        ax.bar(range(20), latents[i].cpu().numpy()) 
        ax.set_ylim([latents.min(), latents.max()])
    axes[0, 2].set_title("Digits")
    axes[1, 2].set_title("Latent representations")
    plt.tight_layout()
    plt.show()