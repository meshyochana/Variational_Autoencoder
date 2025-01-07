import React, { useState } from "react";

function App() {
  const [z, setZ] = useState(Array(20).fill(0)); // Latent space vector
  const [image, setImage] = useState(null);

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
    <div style={{ textAlign: "center", padding: "20px" }}>
      <h1>VAE Digit Generator</h1>
      <div style={{ display: "flex", justifyContent: "center", flexWrap: "wrap" }}>
        {z.map((value, index) => (
          <div key={index} style={{ margin: "10px" }}>
            <input
              type="range"
              min="-1"
              max="1"
              step="0.01"
              value={value}
              onChange={(e) => updateZ(index, e.target.value)}
              style={{ width: "100px" }}
            />
            <div style={{ fontSize: "14px", marginTop: "5px" }}>{value.toFixed(2)}</div>
          </div>
        ))}
      </div>
      <button onClick={generateDigit} style={{ marginTop: "20px", padding: "10px 20px" }}>
        Generate Digit
      </button>
      {image && (
        <div style={{ marginTop: "20px" }}>
          <h2>Generated Digit</h2>
          <img src={image} alt="Generated Digit" style={{ width: "500px", height: "500px" }} />
        </div>
      )}
    </div>
  );
}

export default App;
