import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# transform
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# dataset (client1)
train_data = datasets.ImageFolder("dataset/client1", transform=transform)
train_loader = DataLoader(train_data, batch_size=8, shuffle=True)

# CNN Model
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 16, 3)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(16*111*111, 2)

    def forward(self, x):
        x = self.pool(self.conv(x))
        x = x.view(x.size(0), -1)
        return self.fc(x)

model = CNN().to(device)

# loss + optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# training loop
for epoch in range(3):
    total_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

print("✅ Training Done")