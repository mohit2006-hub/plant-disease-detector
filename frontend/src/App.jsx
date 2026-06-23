// frontend/src/App.jsx
import React, { useState } from 'react';
import './App.css';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [predictionData, setPredictionData] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState('');

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (!file) return;

    setSelectedFile(file);
    setPreviewUrl(URL.createObjectURL(file));
    setPredictionData(null);
    setErrorMessage('');
  };

  const handleInferenceSubmit = async (event) => {
    event.preventDefault();
    if (!selectedFile) return;

    setIsLoading(true);
    setErrorMessage('');
    setPredictionData(null);

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await fetch('http://127.0.0.1:8000/api/predict', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (data.success) {
        setPredictionData(data);
      } else {
        setErrorMessage(data.message || 'Inference engine failed to compute prediction matrices.');
      }
    } catch (err) {
      setErrorMessage('Network error: Unable to connect to backend inference gateway.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="app-container">
      {/* Academic/Institutional Header */}
      <header className="app-header">
        <div className="meta-badge">DEPARTMENT OF AGRICULTURAL & FOOD ENGINEERING</div>
        <h1>Agronomic Deep Learning Classifier</h1>
        <p>Real-Time Convolutional Neural Network (CNN) Pipeline for Crop Foliage Pathology Identification</p>
      </header>

      <main className="dashboard-content">
        {/* Left Column: Input / Controls */}
        <section className="upload-card">
          <h2>
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4M17 8l-5-5-5 5M12 3v12"/></svg>
            Specimen Ingestion Channel
          </h2>
          
          <form onSubmit={handleInferenceSubmit} className="form-layout">
            <div className={`dropzone ${selectedFile ? 'has-file' : ''}`}>
              <input type="file" accept="image/*" onChange={handleFileChange} id="file-input" />
              <div className="dropzone-inner">
                <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"><path d="M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h7M16 5l3 3m0-3l-3 3M21 3l-3 3"/><circle cx="9" cy="9" r="2"/><path d="m21 15-3.086-3.086a2 2 0 0 0-2.828 0L6 21"/></svg>
                <label htmlFor="file-input" className="file-label">
                  {selectedFile ? 'Modify Target Asset Selection' : 'Upload Foliage Specimen Tensor'}
                </label>
                <span className="file-subtext">Supports high-res PNG, JPG, or JPEG imagery</span>
              </div>
            </div>

            {previewUrl && (
              <div className="preview-container">
                <div className="preview-badge">INPUT SPECIMEN VIEW</div>
                <img src={previewUrl} alt="Crop Leaf Specimen Target View" className="specimen-image" />
              </div>
            )}

            <button type="submit" className="analyze-button" disabled={!selectedFile || isLoading}>
              {isLoading ? (
                <span className="btn-flex">
                  <span className="mini-spinner"></span> Computing Model Transformations...
                </span>
              ) : (
                'Execute Diagnostic Analysis'
              )}
            </button>
          </form>
        </section>

        {/* Right Column: Telemetry Matrix Output */}
        <section className="results-card">
          <h2>
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg>
            Diagnostic Engine Telemetry
          </h2>
          
          {errorMessage && (
            <div className="alert error-alert">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
              {errorMessage}
            </div>
          )}
          
          {isLoading && (
            <div className="loading-state">
              <div className="spinner"></div>
              <p className="loading-text">Parsing mathematical tensor transformations across deep hidden feature layers...</p>
            </div>
          )}

          {!predictionData && !isLoading && !errorMessage && (
            <div className="placeholder-wrapper">
              <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="#374151" strokeWidth="1.5"><path d="M22 12h-4l-3 9L9 3l-3 9H2"/></svg>
              <p className="placeholder-text">System idle. Awaiting structural image ingestion sequence to populate diagnostic profiles.</p>
            </div>
          )}

          {predictionData && (
            <div className="telemetry-report">
              <div className="metric-row">
                <span className="label">Detected Biological Classification:</span>
                <span className="value keyword">{predictionData.prediction.replace(/___/g, ' — ').replace(/_/g, ' ')}</span>
              </div>
              <div className="metric-row">
                <span className="label">Softmax Confidence Probability:</span>
                <span className="value confidence">{predictionData.confidence}%</span>
              </div>
              
              <div className="remedy-box">
                <div className="remedy-header">
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z"/></svg>
                  <h3>Recommended Agronomic Intervention Protocol</h3>
                </div>
                <p>{predictionData.remedy}</p>
              </div>
            </div>
          )}
        </section>
      </main>
    </div>
  );
}

export default App;