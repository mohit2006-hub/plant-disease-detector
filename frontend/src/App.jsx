// frontend/src/App.jsx
import React, { useState } from 'react';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  // Handle local file selection and generate view preview URL
  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (!file) return;
    setSelectedFile(file);
    setPreviewUrl(URL.createObjectURL(file));
    setResult(null); // Reset previous results
  };

  // Dispatch image payload multi-part form packet to FastAPI server
  const handleUpload = async () => {
    if (!selectedFile) return;
    setLoading(true);
    
    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await fetch('http://localhost:8000/api/predict', {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      if (data.success) {
        setResult(data);
      } else {
        console.error("Inference Error:", data.message);
      }
    } catch (err) {
      console.error("Network Error connecting to FastAPI:", err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ fontFamily: 'sans-serif', padding: '30px', maxWidth: '900px', margin: '0 auto', backgroundColor: '#0F172A', color: '#E2E8F0', minHeight: '95vh', borderRadius: '16px' }}>
      <header style={{ textAlign: 'center', marginBottom: '40px' }}>
        <h1 style={{ color: '#10B981', margin: '0 0 10px 0' }}>AgroVision AI: Leaf Disease Detector</h1>
        <p style={{ color: '#94A3B8', margin: 0 }}>Upload a high-resolution crop leaf photo for real-time neural network diagnostic assessment.</p>
      </header>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '30px' }}>
        {/* Left Side: Upload Control Panel */}
        <div style={{ backgroundColor: '#1E293B', padding: '25px', borderRadius: '12px', border: '1px solid #334155', textAlign: 'center' }}>
          <h2 style={{ fontSize: '1.2rem', color: '#38BDF8', marginTop: 0 }}>Capture Input Image</h2>
          
          <label style={{ display: 'block', padding: '30px 20px', border: '2px dashed #475569', borderRadius: '8px', cursor: 'pointer', backgroundColor: '#0F172A', marginBottom: '20px' }}>
            <span style={{ color: '#94A3B8' }}>{selectedFile ? selectedFile.name : 'Click to select image file'}</span>
            <input type="file" accept="image/*" onChange={handleFileChange} style={{ display: 'none' }} />
          </label>

          {previewUrl && (
            <div style={{ marginBottom: '20px' }}>
              <img src={previewUrl} alt="Preview" style={{ maxWidth: '100%', maxHeight: '220px', borderRadius: '6px', border: '1px solid #475569' }} />
            </div>
          )}

          <button onClick={handleUpload} disabled={!selectedFile || loading} style={{ width: '100%', padding: '12px', fontWeight: 'bold', border: 'none', borderRadius: '6px', cursor: 'pointer', backgroundColor: selectedFile && !loading ? '#10B981' : '#475569', color: '#FFF', transition: '0.2s' }}>
            {loading ? 'Processing Tensor Inference...' : 'Analyze Leaf Health'}
          </button>
        </div>

        {/* Right Side: ML Assessment Diagnostic Outputs */}
        <div style={{ backgroundColor: '#1E293B', padding: '25px', borderRadius: '12px', border: '1px solid #334155', display: 'flex', flexDirection: 'column', justifyContent: result ? 'flex-start' : 'center', alignItems: result ? 'stretch' : 'center' }}>
          {!result && !loading && (
            <p style={{ color: '#64748B', textAlign: 'center', margin: 0 }}>Awaiting input matrix image payload stream...</p>
          )}

          {loading && (
            <div style={{ textAlign: 'center' }}>
              <div style={{ width: '40px', height: '40px', border: '4px solid #334155', borderTopColor: '#10B981', borderRadius: '50%', animation: 'spin 1s linear infinite', margin: '0 auto 15px auto' }} />
              <p style={{ color: '#94A3B8' }}>Feeding image through Convolutional Layers...</p>
            </div>
          )}

          {result && (
            <div>
              <h2 style={{ fontSize: '1.4rem', color: '#4ADE80', marginTop: 0, borderBottom: '1px solid #334155', paddingBottom: '10px' }}>Diagnostic Analysis</h2>
              
              <div style={{ display: 'flex', justifyContent: 'space-between', margin: '15px 0', padding: '10px', backgroundColor: '#0F172A', borderRadius: '6px' }}>
                <span><strong>Detected Condition:</strong></span>
                <span style={{ color: '#F43F5E', fontWeight: 'bold' }}>{result.prediction}</span>
              </div>

              <div style={{ display: 'flex', justifyContent: 'space-between', margin: '15px 0', padding: '10px', backgroundColor: '#0F172A', borderRadius: '6px' }}>
                <span><strong>Model Confidence Evaluation:</strong></span>
                <span style={{ color: '#38BDF8', fontWeight: 'bold' }}>{result.confidence}%</span>
              </div>

              <div style={{ marginTop: '20px', padding: '15px', backgroundColor: '#0284C7', color: '#FFF', borderRadius: '8px' }}>
                <h3 style={{ margin: '0 0 8px 0', fontSize: '1rem' }}>Recommended Field Action Strategy:</h3>
                <p style={{ margin: 0, fontSize: '0.95rem', lineHeight: '1.4' }}>{result.remedy}</p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;