# test.py
import torch
from torchvision import models
from utils import load_image
import os

# ----------------------------
# Load trained model offline
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 4  # Change if you have different classes

model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

trained_model_path = "models/flower_resnet18_trained.pth"
if os.path.exists(trained_model_path):
    model.load_state_dict(torch.load(trained_model_path, map_location=device))
    print("Loaded trained offline model")
else:
    print("Trained model not found! Train model first using train.py")
    exit()

model = model.to(device)
model.eval()

# ----------------------------
# Test single image
# ----------------------------
image_path = "data/val/daisy/1.jpg"  # Change path to your test image
image = load_image(image_path).to(device)

output = model(image)
_, predicted = torch.max(output, 1)

# Map index to class name
classes = ['daisy', 'rose', 'sunflower', 'tulip']  # Match your dataset classes
print(f"Predicted Class: {classes[predicted.item()]}")