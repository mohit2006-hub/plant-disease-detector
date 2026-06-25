# backend/src/main.py
import os
import io
import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from backend.src.model import DiseaseClassifier

app = FastAPI(title="Plant Disease Detection API Gateway")

# Enable Cross-Origin Resource Sharing (CORS) for Frontend Communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows connections from Vite port (localhost:5173)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static mapping for agronomic treatment advice/remedies based on class predictions
REMEDIES = {
    'Apple___Apple_scab': "Apply protective fungicides early in the season. Rake and destroy fallen leaves to reduce overwintering spores.",
    'Apple___Black_rot': "Prune out dead wood and cankers during winter. Apply labeled copper-based fungicides from green tip stage onward.",
    'Apple___Cedar_apple_rust': "Remove nearby alternative hosts (such as Eastern Red Cedar). Apply preventative rust-specific fungicides.",
    'Potato___Early_blight': "Maintain adequate soil fertility, practice crop rotation, and apply preventative chlorothalonil or copper fungicides.",
    'Potato___Late_blight': "Extremely destructive. Destroy infected crop debris immediately. Apply systematic fungicides like metalaxyl.",
    'Tomato___Bacterial_spot': "Avoid overhead watering to restrict water splash propagation. Apply copper-mancozeb tank mixes early.",
    'Tomato___Early_blight': "Prune lower leaves to enhance air circulation. Apply organic copper sprays at 7-14 day intervals.",
    'Tomato___Late_blight': "Immediately remove and quarantine infected specimens. Treat surrounding plants with defensive protective fungicides.",
    'Tomato___healthy': "Plant exhibits high vegetative vigor. Maintain current irrigation, soil aeration, and nutrition cycles."
}

# Global structural placeholder for our model object
classifier = None

@app.on_event("startup")
def load_ml_weights():
    global classifier
    print("⏳ Initializing Deep Learning Inference Engine...")
    classifier = DiseaseClassifier(num_classes=38)
    
   # Automatically calculates the absolute path to your weights folder
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    weights_path = os.path.join(BASE_DIR, "backend", "weights", "plant_model.pth")
    if os.path.exists(weights_path):
        print(f"💾 Found trained checkpoint. Serializing weights from {weights_path}...")
        # Safely map tensors back to CPU architecture for light-weight server execution
        classifier.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
    else:
        print("⚠️ Warning: No pre-trained weight checkpoint found! Executing backend with random initialization.")
    
    classifier.eval() # Freeze Batch Normalization/Dropout layers

@app.get("/api/health")
def health_check():
    return {"status": "online", "model_loaded": classifier is not None}

@app.post("/api/predict")
async def predict_crop_health(file: UploadFile = File(...)):
    if classifier is None:
        return {"success": False, "message": "ML Engine state remains uninitialized."}
        
    try:
        # Asynchronously ingest binary network packet streams into memory buffer
        image_bytes = await file.read()
        
        # Invoke core PyTorch inference pipeline wrapper
        result = classifier.predict(image_bytes)
        
        if result["status"] == "error":
            return {"success": False, "message": result["message"]}
            
        detected_disease = result["class_name"]
        
        # Dynamically pull the remedy recommendation, fallback to generic advice if class isn't explicitly listed
        remedy_advice = REMEDIES.get(
            detected_disease, 
            "Isolate the plant, avoid moisture accumulation on leaf surfaces, and consult a local agricultural extension office."
        )
        
        return {
            "success": True,
            "filename": file.filename,
            "prediction": detected_disease,
            "confidence": round(result["confidence"] * 100, 2), # Render nicely as percentage (e.g., 94.25%)
            "remedy": remedy_advice
        }
        
    except Exception as e:
        return {"success": False, "message": f"Server Runtime Exception: {str(e)}"}