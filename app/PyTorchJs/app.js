let model;
const latentDim = 20; // The size of the latent space vector
let z = Array(latentDim).fill(0); // Initial latent vector (all zeros)

// Create sliders for latent vector input
const slidersContainer = document.getElementById("sliders-container");
for (let i = 0; i < latentDim; i++) {
  const container = document.createElement("div");
  container.className = "slider-container";

  const slider = document.createElement("input");
  slider.type = "range";
  slider.min = -1;
  slider.max = 1;
  slider.step = 0.01;
  slider.value = 0;
  slider.className = "slider";
  slider.addEventListener("input", (e) => {
    z[i] = parseFloat(e.target.value); // Update the latent vector
  });

  container.appendChild(slider);
  slidersContainer.appendChild(container);
}

// Load the TorchScript model
async function loadModel() {
  try {
    console.log("Loading model...");
    model = await torch.jit.load("./vae_model.pt"); // Adjust the path if needed
    console.log("Model loaded successfully.");
  } catch (error) {
    console.error("Error loading model:", error);
  }
}

// Generate digit from the latent vector
async function generateDigit() {
  if (!model) {
    alert("Model not loaded yet. Please wait.");
    return;
  }

  const latentVector = torch.tensor(z).unsqueeze(0); // Create a tensor from the latent vector
  const output = await model.forward(latentVector); // Run the model
  const outputTensor = output.squeeze().detach(); // Remove batch dimension

  renderImage(outputTensor); // Render the image
}

// Render tensor output to canvas
function renderImage(tensor) {
  const canvas = document.getElementById("outputCanvas");
  const ctx = canvas.getContext("2d");
  const imageData = ctx.createImageData(28, 28); // MNIST images are 28x28

  const normalizedTensor = tensor.sub(tensor.min()).div(tensor.max().sub(tensor.min())); // Normalize tensor
  const data = normalizedTensor.dataSync();

  for (let i = 0; i < data.length; i++) {
    const value = Math.floor(data[i] * 255); // Convert to grayscale value
    imageData.data[i * 4] = value; // Red
    imageData.data[i * 4 + 1] = value; // Green
    imageData.data[i * 4 + 2] = value; // Blue
    imageData.data[i * 4 + 3] = 255; // Alpha
  }

  ctx.putImageData(imageData, 0, 0); // Draw the image on the canvas
}

// Load the model when the page loads
loadModel();
