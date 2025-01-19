import torch
from vae_mnist import VariationalAutoencoder

# Load the full model
vae_model = VariationalAutoencoder()
vae_model.load_state_dict(torch.load("vae_model.pth", weights_only=True))
vae_model.eval()

# Extract the decoder part
decoder = vae_model.decoder

# Save the decoder
torch.save(decoder.state_dict(), "vae_decoder.pth")
