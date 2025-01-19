import torch
from vae_mnist import VariationalAutoencoder

# Load the trained model
model = VariationalAutoencoder()
model.load_state_dict(torch.load('vae_model.pth', weights_only=True))
model.eval()

# Export to TorchScript
scripted_model = torch.jit.script(model)
scripted_model.save("vae_model.pt")
