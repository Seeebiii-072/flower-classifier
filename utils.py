# utils.py
from PIL import Image
from torchvision import transforms

# Image preprocessing for ResNet18
def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

# Load single image for testing
def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = get_transform()
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image