# train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from backend.src.model import DiseaseClassifier

def train_model():
    # 1. Device Configuration (Leverage Apple Silicon GPU if available)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training utilizing device execution target: {device}")

    # 2. Hyperparameters
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    EPOCHS = 5 # Keeping it low for initial validation run
    
   # Remove the space between 'Dataset' and '(Augmented)' on BOTH folder levels
    train_dir = "./data/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train"
    valid_dir = "./data/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/valid"

    # 3. Data Augmentation and Pipelines
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5), # Regularization to reduce overfitting
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    valid_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 4. Loading Datasets using PyTorch's native ImageFolder structure
    print("📋 Initializing Datasets and DataLoaders...")
    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
    valid_dataset = datasets.ImageFolder(root=valid_dir, transform=valid_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    # 5. Model, Loss Function, and Optimizer Setup
    model = DiseaseClassifier(num_classes=38).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 6. Training Loop Execution
    print("🏋️ Starting Model Training Execution...")
    for epoch in range(EPOCHS):
        model.train() # Set to training state
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            # Forward Pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward Pass and Optimization
            optimizer.zero_grad() # Clear gradients from previous step
            loss.backward()       # Compute partial derivatives via backpropagation
            optimizer.step()      # Update network weights

            # Metrics Calculation
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

            if batch_idx % 100 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}] | Batch [{batch_idx}/{len(train_loader)}] | Loss: {loss.item():.4f}")

        epoch_loss = running_loss / total_samples
        epoch_acc = (correct_predictions / total_samples) * 100
        print(f"🟩 Epoch [{epoch+1}/{EPOCHS}] Summary -> Train Loss: {epoch_loss:.4f} | Train Accuracy: {epoch_acc:.2f}%")

    # 7. Serialize trained weights to disk
    os.makedirs("./backend/weights", exist_ok=True)
    torch.save(model.state_dict(), "./backend/weights/plant_model.pth")
    print("💾 Training complete! Optimized model weights serialized to './backend/weights/plant_model.pth'")

if __name__ == "__main__":
    train_model()