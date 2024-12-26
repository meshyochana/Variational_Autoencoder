import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.optim import Adam

import matplotlib.pyplot as plt
import numpy as np
from config_mnist import device, BATCH_SIZE, LEARNING_RATE, latent_dim


import torch
import torch.nn as nn
import torch.nn.functional as F

class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dim=20):
        super(VariationalAutoencoder, self).__init__()
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),  # Output: (32, 32, 32)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), # Output: (64, 16, 16)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # Output: (128, 8, 8)
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), # Output: (256, 4, 4)
            nn.ReLU()
        )

        self.fc_mu = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_log_var = nn.Linear(256 * 4 * 4, latent_dim)

        # Decoder
        self.fc_decoder = nn.Linear(latent_dim, 256 * 4 * 4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), # Output: (128, 8, 8)
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # Output: (64, 16, 16)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),   # Output: (32, 32, 32)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),    # Output: (3, 64, 64)
            nn.Sigmoid()  # Ensure output is in range [0, 1]
        )

    def encode(self, x):
        h = self.encoder(x)
        h = torch.flatten(h, start_dim=1)  # Flatten for fully connected layers
        mu = self.fc_mu(h)
        log_var = self.fc_log_var(h)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc_decoder(z)
        h = h.view(-1, 256, 4, 4)  # Reshape for transposed convolutions
        x_reconstructed = self.decoder(h)
        return x_reconstructed

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_reconstructed = self.decode(z)
        return x_reconstructed, mu, log_var


def loss_function(recon_x, x, mu, log_var, flag_bce=True, flag_kld=True):
    BCE = flag_bce * F.mse_loss(recon_x, x, reduction='sum')
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
            if i == 0: # Only process the first batch
                n = min(data.size(0), n) # Ensure we don't exceed batch size
                comparison = torch.cat([data[:n], reconstruction[:n]])  # Concatenate originals and reconstructions
                break

    # Normalize data and reconstructions to [0, 1] if needed
    data = data.clamp(0, 1)  # Ensure data is in [0, 1]
    reconstruction = reconstruction.clamp(0, 1)  # Ensure reconstruction is in [0, 1]

    # Plotting
    plt.close('all')
    fig, axes = plt.subplots(nrows=2, ncols=n, figsize=(10, 2))
    for i in range(n):
        # Original images
        axes[0, i].imshow(data[i].cpu().permute(1, 2, 0).numpy())  # Convert tensor to HWC format for RGB
        axes[0, i].axis('off')

        # Reconstructed images
        axes[1, i].imshow(reconstruction[i].cpu().permute(1, 2, 0).numpy())  # Convert tensor to HWC format for RGB
        axes[1, i].axis('off')
    plt.show()


def generate_new_cats(model, num_samples):
    model.eval()  # Put the model in inference mode
    with torch.no_grad():  # No need to track gradients
        # Sample random points in the latent space
        z = torch.randn(num_samples, latent_dim).to(device)
        # Generate digits from the latent vectors
        generated_digits = model.decode(z)
        print(generated_digits.shape)
    return z, generated_digits


def plot_loss(losses):
    plt.close('all')
    fig, ax = plt.subplots(1,1, figsize=(10, 5))
    ax.plot(losses, "o-")
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    plt.show()


def plot_images_and_latents(latents, images, plot_samples=5, latent_dim=20):
    plt.close('all')
    fig, axes = plt.subplots(nrows=2, ncols=plot_samples, figsize=(15, 6))

    # Normalize latent range for consistent bar plot scaling
    latent_min, latent_max = latents.min().cpu().numpy(), latents.max().cpu().numpy()

    for i in range(plot_samples):
        # Plot the images
        ax_img = axes[0, i]
        ax_img.imshow(images[i].cpu().permute(1, 2, 0).clamp(0, 1).numpy())  # Convert to HWC format for RGB
        ax_img.axis('off')

        # Plot the latent vectors
        ax_latent = axes[1, i]
        ax_latent.bar(range(latent_dim), latents[i].cpu().numpy())
        ax_latent.set_ylim([latent_min, latent_max])

    axes[0, plot_samples // 2].set_title("Generated Images", fontsize=12)
    axes[1, plot_samples // 2].set_title("Latent Representations", fontsize=12)
    plt.tight_layout()
    plt.show()