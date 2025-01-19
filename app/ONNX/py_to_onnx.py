import torch
from vae_mnist import VariationalAutoencoder

# Load the trained model
model = VariationalAutoencoder()
model.load_state_dict(torch.load("vae_model.pth"))
model.eval()  # Set the model to evaluation mode

# Create a dummy input matching the model's input size
latent_dim = 20  # Adjust to match your latent space size
dummy_input = torch.randn(1, latent_dim)  # Batch size = 1, latent_dim

# Export the model to ONNX format
torch.onnx.export(
    model,
    dummy_input,
    "vae_model.onnx",  # Output file
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=["latent_vector"],
    output_names=["generated_image"],
)
