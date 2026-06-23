# backend/src/model.py
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import io

# Explicit classification classes mapped directly from the Kaggle dataset folder structure
DISEASE_CLASSES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry___Powdery_mildew', 'Cherry___healthy', 'Corn___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn___Common_rust', 'Corn___Northern_Leaf_Blight', 'Corn___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

class DiseaseClassifier(nn.Module):
    def __init__(self, num_classes=38):
        super(DiseaseClassifier, self).__init__()
        
        # 1. Load a lightweight MobileNetV3 backbone architecture
        # Using weights=None (or pretrained=False) because we will train this on agricultural data
        self.backbone = models.mobilenet_v3_small(weights=None)
        
        # 2. Extract the number of input features going into the original classifier head
        # MobileNetV3 small uses a final linear layer in self.backbone.classifier[3]
        in_features = self.backbone.classifier[3].in_features
        
        # 3. Replace the final linear layer to map to our 38 custom agricultural disease classes
        self.backbone.classifier[3] = nn.Linear(in_features, num_classes)
        
        # Image transformation pipeline to match standard ML tensor inputs (224x224 pixels)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
    def forward(self, x):
        return self.backbone(x)

    def predict(self, image_bytes):
        try:
            # Load raw image binary data stream
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            tensor = self.transform(image).unsqueeze(0) # Add batch dimension (1, 3, 224, 224)
            
            self.eval() # Set model layers to evaluation mode
            with torch.no_grad():
                outputs = self.forward(tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                confidence, class_idx = torch.max(probabilities, 0)
                
            return {
                "status": "success",
                "class_name": DISEASE_CLASSES[class_idx.item()],
                "confidence": float(confidence.item())
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}