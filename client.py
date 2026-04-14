# type: ignore
import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import os
import numpy as np
import json
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

torch.manual_seed(42)
np.random.seed(42)

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_PATH = "dataset/client2"  # change for client1

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = datasets.ImageFolder(DATA_PATH, transform=transform)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

model = CNN().to(device)

class_counts = torch.bincount(torch.tensor(dataset.targets))
weights = 1.0 / class_counts.float()
weights = weights / weights.sum()

criterion = nn.CrossEntropyLoss(weight=weights.to(device))
optimizer = optim.Adam(model.parameters(), lr=0.0005)

# =====================
# METRICS FILE
# =====================
if not os.path.exists("metrics.json"):
    with open("metrics.json", "w") as f:
        json.dump([], f)

def save_metrics(acc, precision, recall, f1, cm, round_num):
    client_name = os.path.basename(DATA_PATH)

    data = {
        "client": client_name,
        "round": round_num,   # 🔥 IMPORTANT
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": cm.tolist()
    }

    try:
        with open("metrics.json", "r") as f:
            history = json.load(f)
    except:
        history = []

    history.append(data)

    with open("metrics.json", "w") as f:
        json.dump(history, f, indent=4)

def calculate_metrics():
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)

    return accuracy, precision, recall, f1, cm

def get_weights():
    return [val.detach().cpu().numpy() for val in model.parameters()]

def set_weights(parameters):
    for param, new_val in zip(model.parameters(), parameters):
        param.data = torch.tensor(new_val).to(device)

class FlowerClient(fl.client.NumPyClient):

    def get_parameters(self, config):
        return get_weights()

    def fit(self, parameters, config):
        set_weights(parameters)

        model.train()
        for epoch in range(3):
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        return get_weights(), len(dataset), {}

    def evaluate(self, parameters, config):
        set_weights(parameters)

        acc, precision, recall, f1, cm = calculate_metrics()

        round_num = config.get("server_round", 0)  # 🔥 KEY FIX

        save_metrics(acc, precision, recall, f1, cm, round_num)

        return 0.0, len(dataset), {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }

fl.client.start_numpy_client(
    server_address="localhost:8081",
    client=FlowerClient(),
)