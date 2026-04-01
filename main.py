import os
import requests
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# ✨ Model file destination
MODEL_DIR = "models"
MODEL_FILE = "flowers102_resnet50.pth"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE)

# 🧩 URL for the pretrained model
# You should replace this with the direct download URL of a ResNet50 flowers102 model
DOWNLOAD_URL = "https://huggingface.co/anonauthors/flowers102-resnet50/resolve/main/pytorch_model.bin"

# 📌 Ensure model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# 1️⃣ Download model if not present
if not os.path.exists(MODEL_PATH):
    print("Pretrained model not found locally!")
    print("Downloading model...")

    response = requests.get(DOWNLOAD_URL, stream=True)
    if response.status_code == 200:
        with open(MODEL_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print("Model downloaded successfully!")
    else:
        print("Failed to download model. Status code:", response.status_code)
        exit()

# 2️⃣ Load the model architecture
print("Loading ResNet50 model… (offline inference)")
model = models.resnet50(pretrained=False)
model.fc = nn.Linear(in_features=2048, out_features=102)

# 3️⃣ Load downloaded weights
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

# 4️⃣ Image preprocessing
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 📸 Get image from user
img_path = input("Enter the path of the flower image: ").strip()
if not os.path.exists(img_path):
    print("❌ Image path invalid, please check and try again!")
    exit()

# 5️⃣ Prepare image for prediction
img = Image.open(img_path).convert("RGB")
input_tensor = preprocess(img).unsqueeze(0)

# 6️⃣ Predict
with torch.no_grad():
    outputs = model(input_tensor)
    _, predicted_idx = torch.max(outputs, 1)

print(f"\n🌼 Predicted flower class index: {predicted_idx.item()}")

# — Optional: index mapping to actual class names — use Oxford Flowers 102 list
flowers102 = [
    "pink primrose","hard-leaved pocket orchid","canterbury bells","sweet pea",
    "english marigold","tiger lily","moon orchid","bird of paradise","monkshood",
    "globe thistle","snapdragon","colt's foot","king protea","spear thistle",
    "yellow iris","globe-flower","purple coneflower","peruvian lily",
    "balloon flower","giant white arum lily","fire lily","pincushion flower",
    "french marigold","buttercup","daisy","common dandelion","petunia","wild pansy",
    "columbine","african daisy","spotted orchid","kiwi orchid","black-eyed susan",
    "silverbush","californian poppy","corn poppy","mexican aster","alpine sea holly",
    "ruby-lipped cattleya","cape flower","tulip","winter daphne","wallflower",
    "marigold daisy","butterfly orchid","magnolia","orange dahlia","rose","thorn apple",
    "morning glory","passion flower","lotus","toad lily","anemone","black tulip",
    "sweet william","carnation","garden phlox","lenten rose","barbeton daisy",
    "primula","sunflower","pelargonium","bishop of llandaff","gaura","geranium",
    "hibiscus","coltsfoot","saxifrage","mallow","alpine rose","teeny lily",
    # (list continues all 102 categories)
]

try:
    label = flowers102[predicted_idx.item()]
    print(f"➡️ Predicted flower name: {label}")
except:
    print("⚠️ Class index out of range")