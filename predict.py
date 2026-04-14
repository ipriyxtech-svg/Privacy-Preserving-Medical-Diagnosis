import torch
import cv2
import numpy as np
import torch.nn as nn
import os

# =====================
# MODEL (MATCH TRAINING 🔥)
# =====================
class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)

        self.pool = nn.MaxPool2d(2)

        self.fc1 = nn.Linear(32 * 56 * 56, 64)
        self.fc2 = nn.Linear(64, 2)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)

# =====================
# LOAD MODEL
# =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CNN().to(device)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()

# =====================
# CLASS MAPPING
# =====================
class_names = ["ABNORMAL", "NORMAL"]

# =====================
# AUTO IMAGE PICK
# =====================
img_path = None

for file in os.listdir():
    if file.lower().endswith((".png", ".jpg", ".jpeg")):
        img_path = file
        break

if img_path is None:
    print("❌ No image found")
    exit()

print("📸 Using image:", img_path)

# =====================
# IMAGE LOAD + PREPROCESS
# =====================
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    print("❌ Image read failed")
    exit()

img = cv2.resize(img, (224, 224))

# normalize (same as training)
img = img / 255.0
img = (img - 0.5) / 0.5

img = torch.tensor(img).float().unsqueeze(0).unsqueeze(0).to(device)

# =====================
# PREDICTION
# =====================
with torch.no_grad():
    output = model(img)

# stability
output = torch.clamp(output, -10, 10)

# 🔥 better scaling
prob = torch.softmax(output / 5.0, dim=1)

pred_index = int(torch.argmax(prob, dim=1).item())
confidence = prob[0][pred_index].item()

pred_label = class_names[pred_index]

# =====================
# SMART DECISION LOGIC 🔥
# =====================
if confidence < 0.65:
    result = "⚠️ UNCERTAIN"
elif pred_label == "ABNORMAL" and confidence > 0.7:
    result = "⚠️ ABNORMAL"
elif pred_label == "NORMAL" and confidence > 0.7:
    result = "✅ NORMAL"
else:
    result = "⚠️ UNCERTAIN"

# =====================
# OUTPUT
# =====================
print(result)
print("Confidence:", round(confidence, 2))
print("Predicted Label:", pred_label)
print("Probabilities:", prob.cpu().numpy())