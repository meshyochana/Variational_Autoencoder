let session; // ONNX.js session
const latentDim = 20; // Size of the latent vector
let z = Array(latentDim).fill(0); // Initialize latent vector

// Create sliders for user input
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
    z[i] = parseFloat(e.target.value); // Update latent vector
  });

  container.appendChild(slider);
  slidersContainer.appendChild(container);
}

// Load the ONNX model
async function loadModel() {
  console.log("Loading model...");
  session = await ort.InferenceSession.create("vae_decoder.onnx"); // Ensure correct path to the ONNX file
  console.log("Model loaded successfully.");
}

// Generate a digit from the latent vector
async function generateDigit() {
  if (!session) {
    alert("Model not loaded yet. Please wait.");
    return;
  }

  // Convert latent vector to ONNX tensor
  const tensor = new ort.Tensor("float32", Float32Array.from(z), [1, latentDim]);

  // Run inference
  const output = await session.run({ latent_vector: tensor });
  const outputData = output.generated_image.data;

  renderImage(outputData);
}

// Render the output image to the canvas
function renderImage(data) {
  const canvas = document.getElementById("outputCanvas");
  const ctx = canvas.getContext("2d");
  const imageData = ctx.createImageData(28, 28); // Adjust dimensions if needed

  // Normalize data and convert to grayscale
  const min = Math.min(...data);
  const max = Math.max(...data);
  const normalizedData = data.map((x) => ((x - min) / (max - min)) * 255);

  for (let i = 0; i < normalizedData.length; i++) {
    const value = normalizedData[i];
    imageData.data[i * 4] = value; // Red
    imageData.data[i * 4 + 1] = value; // Green
    imageData.data[i * 4 + 2] = value; // Blue
    imageData.data[i * 4 + 3] = 255; // Alpha
  }

  ctx.putImageData(imageData, 0, 0); // Draw image on canvas
}

// Load the model when the page loads
loadModel();
