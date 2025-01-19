import torch
import torch.nn as nn

# Define the decoder
decoder = nn.Sequential(
    nn.Linear(20, 400),
    nn.ReLU(),
    nn.Linear(400, 784),
    nn.Sigmoid()  # Ensures pixel values are between 0 and 1
)

# Save the model
#torch.save(decoder.state_dict(), "vae_decoder.pth")

# Load the saved weights
decoder.load_state_dict(torch.load("vae_decoder.pth"))
decoder.eval()

# Create a dummy latent vector
latent_dim = 20
dummy_input = torch.randn(1, latent_dim)

# Export to ONNX
torch.onnx.export(
    decoder,
    dummy_input,
    "vae_decoder.onnx",
    verbose=True
)

#    export_params=True,
#    opset_version=11,
#    do_constant_folding=True,
#    input_names=["latent_vector"],
#    output_names=["generated_image"],