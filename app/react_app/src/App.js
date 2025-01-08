import React, { useState } from "react";
import "./App.css"; // For additional styles

// Predefined latent space vectors for digits 0-9
const digitPresets = {
  0: [-1.2524,  0.0755, -0.0106, -0.6898, -0.2629, -0.1628,  0.3950,  0.5596, -1.3591, -1.0935,  0.1366,  1.3057,  0.8853, -0.7318,  1.9713, -0.4037, 0.1359,  1.5798,  0.7698, -0.4532],
  1: [ 0.2889,  0.7243,  0.1015,  2.1720,  0.5653,  1.3859, -0.7426,  0.3278,-0.1835,  0.3353,  0.3486, -0.7963, -0.2437, -0.0955, -0.8270, -0.6165,-0.6155,  0.1667,  0.5399, -0.6885],
  2: [ 0.0589, -0.8529,  0.8941, -1.2263,  0.2421, -1.3546, -0.2346,  2.5476,0.4506,  1.4276, -0.9362, -0.7672,  1.4461,  1.5935, -0.5549, -0.2404, -0.3451,  1.5197, -0.7947, -0.8239], 
  3: [-0.7725, -0.1289, -0.9178, -1.3700, -0.2018, -0.5672, -1.7603,  0.2503,0.0063,  1.1344,  1.4230,  0.9938,  1.7464, -0.5343, -1.1024,  1.2552,-0.5002, -1.0286, -1.7176,  2.1579], 
  4: [ 1.2268,  0.1663, -0.8555,  0.0089, -0.6548, -0.9456, -0.9971,  0.7299, 0.2486, -0.8077, -1.8104, -0.6614,  0.4173, -0.8855, -0.8769, -0.0720,1.5613,  0.9759,  1.0364, -0.7370],
  5: [-2.7489, -0.1045, -1.2293,  0.0048, -0.5565, -0.0250,  0.6223, -0.2694, 0.6734,  0.5011, -0.8099,  0.6100, -1.3594,  0.2587, -0.8404, -0.1785,-0.7335, -1.4432, -0.4499, -0.6451],
  6: [-0.8916,  0.1975, -0.8116,  0.4024,  1.0810,  0.5630, -0.7907, -0.5423, 0.3144,  0.1675, -0.8055, -0.2056,  0.2756, -0.0517,  0.8227, -0.3951,-0.6636, -0.1979, -0.4699,  1.1600],
  7: [-0.4210, -0.8965,  0.5823, -0.2998,  0.3512, -1.7472,  0.6045,  1.1194,0.4498,  0.2767,  0.7401, -0.9582, -1.4372, -0.8044, -1.1384,  0.7434,0.0311,  1.5520, -1.1587, -0.2964],
  8: [-2.2752, -0.6756, -0.3134,  0.6081, -1.2022,  1.1326,  0.6693, -0.2049, 0.7858,  2.0446, -0.1717,  1.3134,  0.0734, -0.8289, -0.5228, -1.0525, 0.5811,  0.7046, -0.4138,  0.4686],
  9: [-0.7980,  0.4672, -0.2171, -0.4204, -0.0185, -0.4072, -0.2959,  1.6062,0.0608, -1.2366,  0.1038,  0.8815,  1.2771, -1.0541, -0.1874, -0.0928,-0.7434, -0.2191,  1.1101, -0.7981],
};

function App() {
  const [z, setZ] = useState(Array(20).fill(0));
  const [image, setImage] = useState(null);

  const setPreset = (digit) => {
    const preset = digitPresets[digit];
    if (preset) {
      setZ(preset);
    }
  };

  const setRandom = () => {
    const randomZ = Array.from({ length: 20 }, () => (Math.random() * 2 - 1).toFixed(2));
    setZ(randomZ.map(Number));
  };

  const updateZ = (index, value) => {
    const newZ = [...z];
    newZ[index] = parseFloat(value);
    setZ(newZ);
  };

  const generateDigit = async () => {
    try {
      const response = await fetch("http://localhost:5000/generate", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ z }),
      });
      const data = await response.json();
      if (data.error) {
        alert(`Error: ${data.error}`);
        return;
      }
      setImage(`data:image/png;base64,${data.image}`);
    } catch (error) {
      console.error("Error generating digit:", error);
    }
  };

  return (
    <div className="app">
      <h1 className="title">VAE Digit Generator</h1>

      <div className="buttons">
        {Object.keys(digitPresets).map((digit) => (
          <button
            key={digit}
            onClick={() => setPreset(Number(digit))}
            className="digit-button"
          >
            Set {digit}
          </button>
        ))}
        <button onClick={setRandom} className="random-button">
          Ser Random Digit
        </button>
      </div>

      <div className="sliders">
        {z.map((value, index) => (
          <div key={index} className="slider-container">
            <input
              type="range"
              min="-1"
              max="1"
              step="0.01"
              value={value}
              onChange={(e) => updateZ(index, e.target.value)}
              className="slider"
            />
            <div className="slider-value">{value.toFixed(2)}</div>
          </div>
        ))}
      </div>

      <button onClick={generateDigit} className="generate-button">
        Generate Digit
      </button>

      {image && (
        <div className="output">
          <h2>Generated Digit</h2>
          <img src={image} alt="Generated Digit" className="output-image" />
        </div>
      )}
    </div>
  );
}

export default App;
