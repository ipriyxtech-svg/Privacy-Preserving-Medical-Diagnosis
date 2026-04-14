import os, shutil, random

base = "dataset/balanced"

client1 = "dataset/client1"
client2 = "dataset/client2"

for c in ["NORMAL", "ABNORMAL"]:
    os.makedirs(f"{client1}/{c}", exist_ok=True)
    os.makedirs(f"{client2}/{c}", exist_ok=True)

    files = os.listdir(f"{base}/{c}")
    random.shuffle(files)

    split = len(files)//2

    for f in files[:split]:
        shutil.copy(f"{base}/{c}/{f}", f"{client1}/{c}")

    for f in files[split:]:
        shutil.copy(f"{base}/{c}/{f}", f"{client2}/{c}")

print("✅ Clients ready (Client1 & Client2)")