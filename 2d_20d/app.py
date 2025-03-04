from flask import Flask, request, jsonify, send_from_directory
import torch
from torchvision.transforms import ToPILImage
from flask_cors import CORS
from vae_mnist import VariationalAutoencoder  # Import your VAE class
from io import BytesIO
import base64

app = Flask(__name__)
application = app
CORS(app)  # Allow all origins  # Enable CORS for all routes

# Load the two models
device = torch.device('cpu')  # Use CPU for simplicity

# First model (from app.py)
LATENT_DIM_20 = 20
model_vae_20 = VariationalAutoencoder(latent_dim=LATENT_DIM_20)
model_vae_20.load_state_dict(torch.load('vae_model_ldim_20.pth', map_location=device, weights_only=True))
model_vae_20.eval()

# Second model (from app_22.py) with a different latent dimension
LATENT_DIM_2 = 2
model_vae_2 = VariationalAutoencoder(latent_dim=LATENT_DIM_2)
model_vae_2.load_state_dict(torch.load('vae_model_ldim_2.pth', map_location=device, weights_only=True))
model_vae_2.eval()


def generate_image(model, z, latent_dim):
    """Helper function to generate an image from a given model and latent vector."""
    z = torch.tensor(z, dtype=torch.float32).unsqueeze(0)  # Shape (1, latent_dim)

    if z.shape[1] != latent_dim:
        return jsonify({'error': f'Expected latent dimension of {latent_dim}, but got {z.shape[1]}'}), 400

    # Decode z to generate the digit
    generated_digits = model.decoder(z).detach().squeeze(0)  # Shape: (784,)

    # Reshape to 2D for MNIST (28x28)
    generated_digits = generated_digits.view(28, 28)

    # Normalize pixel values
    generated_digits = (generated_digits - generated_digits.min()) / (generated_digits.max() - generated_digits.min())

    # Convert to PIL image
    pil_image = ToPILImage()(generated_digits)

    # Convert image to base64
    buffer = BytesIO()
    pil_image.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return jsonify({'image': img_str, 'latent_dim': latent_dim})


@app.route('/generate_vae', methods=['POST'])
def generate_vae():
    """Generate image using the first model (vae_model.pth)."""
    data = request.json
    return generate_image(model_vae_20, data['z'], LATENT_DIM_20)


@app.route('/generate_vae_22', methods=['POST'])
def generate_vae_22():
    """Generate image using the second model (vae_model_ldim_22.pth)."""
    data = request.json
    return generate_image(model_vae_2, data['z'], LATENT_DIM_2)


@app.route('/latent/<filename>')
def serve_latent_file(filename):
    """Serve latent files (only needed for vae_22 model)."""
    return send_from_directory('./latent_data', filename)


if __name__ == '__main__':
    app.run(debug=True)
