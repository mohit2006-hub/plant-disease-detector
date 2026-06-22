# backend/src/main.py
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from .model import DiseaseClassifier

app = FastAPI(title="Plant Disease Detection API Gateway")

# Enable CORS middle-tier routing headers so our React client can consume the data packet payloads securely
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows requests from your local Vite port (5173)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instantiate our Machine Learning inference engine class instance at startup cache
classifier = DiseaseClassifier()

@app.post("/api/predict")
async def predict_crop_health(file: UploadFile = File(...)):
    try:
        # Read the raw incoming binary image format stream into an execution buffer
        image_bytes = await file.read()
        
        # Execute the prediction logic through our Torch model pipeline wrapper
        result = classifier.predict(image_bytes)
        
        if result["status"] == "error":
            return {"success": False, "message": result["message"]}
            
        return {
            "success": True,
            "filename": file.filename,
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "remedy": result["remedy"]
        }
    except Exception as e:
        return {"success": False, "message": f"Server Runtime Exception: {str(e)}"}

@app.get("/api/health")
def health_check():
    return {"status": "online", "model_loaded": True}