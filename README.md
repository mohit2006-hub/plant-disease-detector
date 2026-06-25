# 🌿 Agronomic Deep Learning Classifier Pipeline

An end-to-end, high-throughput full-stack application leveraging an optimized **Custom Convolutional Neural Network (CNN)** architecture built in PyTorch to detect, isolate, and categorize 38 distinct variations of plant foliage pathologies.

---

## 🏗️ System & Pipeline Architecture

The application isolates computational concerns into three independent runtime layers:

1. **Reactive Frontend Client (React + Vite):** A single-page dashboard designed for synchronous image ingestion.
2. **Asynchronous Inference Gateway (FastAPI + Uvicorn):** A high-performance web routing layer utilizing `python-multipart` to pipe tensors directly through model checkpoints.
3. **Deep Learning Engine (PyTorch):** A Custom CNN that processes structural activations and resolves predictions via a 38-way Softmax layer.

---

## 📈 Neural Network Specifications & Training Analytics

* **Input Image Target Dimensions:** 224 × 224 Pixels (RGB)
* **Optimization Parameters:** Adam Optimizer ($Learning Rate = 0.001$)
* **Terminal Training Loss:** `0.1314`
* **Peak Training Classification Accuracy:** `95.72%`
* **Production Live Inference Confidence:** `98.61%`

---

## 🚀 Local Installation & Deployment

### 1. Backend API Server Setup
```bash
cd plant-disease-detector
source env/bin/activate
pip install fastapi uvicorn python-multipart torch torchvision pillow
python3 -m uvicorn backend.src.main:app --reload --port 8000