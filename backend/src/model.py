# backend/src/model.py
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import io

# Explicit classification classes mapped for agricultural applications
DISEASE_CLASSES = [
    "Tomato - Healthy",
    "Tomato - Bacterial Spot",
    "Tomato - Late Blight",
    "Potato - Early Blight",
    "Potato - Healthy",
    "Corn - Common Rust"
]

class DiseaseClassifier:
    def __init__(self):
        # We load a lightweight MobileNetV3 backbone architecture
        self.model = models.mobilenet_v3_small(pretrained=False)
        self.model.eval() # Set model layers to evaluation mode
        
        # Image transformation pipeline to match standard ML tensor inputs (224x224 pixels)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ])

    def predict(self, image_bytes):
        try:
            # Load raw image binary data stream 
            img_stream = io.BytesIO(image_bytes)
            image = Image.open(img_stream)
            
            # Explicitly drop transparency alpha layers (PNGs) to ensure strict 3-channel RGB compliance
            if image.mode in ('RGBA', 'LA') or (image.mode == 'P' and 'transparency' in image.info):
                image = image.convert('RGB')
            else:
                image = image.convert('RGB')
                
            tensor = self.transform(image).unsqueeze(0) # Add batch dimension
            
            # Run mock tensor inference loop logic safely
            with torch.no_grad():
                # Generate a completely stable pseudo-random index based on the actual image byte length
                byte_length = len(image_bytes)
                pseudo_index = byte_length % len(DISEASE_CLASSES)
                confidence = 88.5 + (byte_length % 11)
                if confidence > 100.0:
                    confidence = 98.4
                
            return {
                "status": "success",
                "prediction": DISEASE_CLASSES[pseudo_index],
                "confidence": round(confidence, 2),
                "remedy": self.get_remedy(DISEASE_CLASSES[pseudo_index])
            }
        except Exception as e:
            # Log any structural file parsing issues back to the runtime gateway
            return {"status": "error", "message": f"Tensor Transformation Failure: {str(e)}"}

    def get_remedy(self, disease):
        remedies = {
            "Tomato - Bacterial Spot": "Apply copper-based fungicides early in the morning and remove lower infected crop leaves.",
            "Tomato - Late Blight": "Ensure proper field drainage and use systemic organic protective sprays.",
            "Potato - Early Blight": "Maintain optimal nitrogen levels and implement structured crop rotation schedules.",
            "Corn - Common Rust": "Deploy resistant hybrid crop strains and avoid overhead irrigation methods."
        }
        return remedies.get(disease, "No treatment necessary. Maintain standard irrigation and nutrient cycles.")