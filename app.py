from flask import Flask, render_template, request, redirect, jsonify
import torch
import cv2
import numpy as np
import torch.nn as nn
import json
from datetime import datetime
import os
import hashlib

app = Flask(__name__)

# =====================
# MODEL
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

try:
    model.load_state_dict(torch.load("model.pth", map_location=device))
    model.eval()
    print("✅ Model Loaded")
except:
    print("⚠️ model.pth not found!")

class_names = ["ABNORMAL", "NORMAL"]

# =====================
# FILE INIT (SAFE)
# =====================
def init_file(file):
    if not os.path.exists(file):
        with open(file, "w") as f:
            json.dump([], f)

init_file("history.json")
init_file("metrics.json")

# =====================
# 🔗 BLOCKCHAIN
# =====================
def create_hash(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()

def create_block(prev_hash, data):
    block = {
        "data": data,
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "prev_hash": prev_hash
    }
    block["hash"] = create_hash(block)
    return block

def verify_chain(chain):
    for i in range(1, len(chain)):
        prev = chain[i-1]
        curr = chain[i]

        if curr["prev_hash"] != prev["hash"]:
            return False

        recalculated = create_hash({
            "data": curr["data"],
            "timestamp": curr["timestamp"],
            "prev_hash": curr["prev_hash"]
        })

        if curr["hash"] != recalculated:
            return False

    return True

# =====================
# SAFE JSON LOAD
# =====================
def load_json(file):
    try:
        with open(file) as f:
            return json.load(f)
    except:
        return []

# =====================
# HOME
# =====================
@app.route("/", methods=["GET", "POST"])
def home():
    result = ""
    confidence = 0

    if request.method == "POST":

        file = request.files.get("file")

        if not file:
            return render_template("index.html", result="❌ No file uploaded", confidence=0)

        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

        if img is None:
            return render_template("index.html", result="❌ Invalid image", confidence=0)

        img = cv2.resize(img, (224, 224))

        # =====================
        # XRAY CHECK
        # =====================
        mean_val = img.mean()
        std_val = img.std()
        edges = cv2.Canny(img, 50, 150)
        edge_density = edges.mean()

        if not (30 < mean_val < 180 and std_val > 25 and edge_density > 8):
            return render_template("index.html",
                result="❌ Please upload a valid Chest X-ray image",
                confidence=0
            )

        # =====================
        # NORMALIZE
        # =====================
        img = img / 255.0
        img = (img - 0.5) / 0.5
        img = torch.tensor(img).float().unsqueeze(0).unsqueeze(0).to(device)

        # =====================
        # PREDICTION
        # =====================
        with torch.no_grad():
            output = model(img)

        prob = torch.softmax(output, dim=1)

        pred = int(torch.argmax(prob, dim=1).item())
        conf = prob[0][pred].item()

        if conf < 0.7:
            result = "⚠️ UNCERTAIN"
        elif class_names[pred] == "ABNORMAL":
            result = "⚠️ ABNORMAL"
        else:
            result = "✅ NORMAL"

        confidence = float(round(conf, 2))

        # =====================
        # BLOCKCHAIN SAVE
        # =====================
        chain = load_json("history.json")

        data = {
            "result": result,
            "confidence": confidence
        }

        prev_hash = chain[-1]["hash"] if len(chain) > 0 else "0"

        block = create_block(prev_hash, data)
        chain.append(block)

        with open("history.json", "w") as f:
            json.dump(chain, f, indent=4)

    return render_template("index.html", result=result, confidence=confidence)

# =====================
# HISTORY
# =====================
@app.route("/history")
def history_page():

    chain = load_json("history.json")
    metrics = load_json("metrics.json")

    valid = verify_chain(chain) if len(chain) > 0 else True

    return render_template(
        "history.html",
        history=chain,
        metrics=metrics,
        valid=valid
    )

# =====================
# LIVE METRICS API
# =====================
@app.route("/metrics-data")
def metrics_data():
    data = load_json("metrics.json")
    return jsonify(data)

# =====================
# CLEAR
# =====================
@app.route("/clear")
def clear():
    with open("history.json", "w") as f:
        json.dump([], f)

    with open("metrics.json", "w") as f:
        json.dump([], f)

    return redirect("/history")

# =====================
# RUN
# =====================
if __name__ == "__main__":
    app.run(debug=True)