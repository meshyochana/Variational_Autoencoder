from flask import Flask, request, jsonify
import torch
from torchvision.transforms import ToPILImage
from flask_cors import CORS  # Import CORS

# Load the pre-trained VAE model
from vae_mnist import VariationalAutoencoder  # Import your VAE class

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the saved model
device = torch.device('cpu')  # Use CPU for simplicity
model = VariationalAutoencoder()  # Ensure this matches your model definition
model.load_state_dict(torch.load('vae_model.pth', map_location=device, weights_only=True))
model.eval()

# Route to generate digits based on latent space z
@app.route('/generate', methods=['POST'])
def generate():
    try:
        data = request.json
        z = torch.tensor(data['z'], dtype=torch.float32).unsqueeze(0)  # Shape (1, latent_dim)
        
        # Decode z to generate the digit
        generated_digits = model.decoder(z).detach().squeeze(0)  # Shape: (784,)
        
        # Reshape to 2D for MNIST (28x28)
        generated_digits = generated_digits.view(28, 28)  # Ensure it's 2D
        
        # Normalize pixel values (optional, depends on your decoder output)
        generated_digits = (generated_digits - generated_digits.min()) / (generated_digits.max() - generated_digits.min())
        
        # Convert to PIL image
        pil_image = ToPILImage()(generated_digits)
        
        # Convert image to base64
        from io import BytesIO
        import base64
        buffer = BytesIO()
        pil_image.save(buffer, format="PNG")
        img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        
        return jsonify({'image': img_str})
    except Exception as e:
        return jsonify({'error': str(e)}), 400
    
if __name__ == '__main__':
    app.run(debug=True)