import os, shutil, random

# source
normal_src = "dataset/processed/NORMAL"
abnormal_src = "dataset/processed/ABNORMAL"

# destination
normal_dst = "dataset/balanced/NORMAL"
abnormal_dst = "dataset/balanced/ABNORMAL"

os.makedirs(normal_dst, exist_ok=True)
os.makedirs(abnormal_dst, exist_ok=True)

# select 162 normal images
normal_files = os.listdir(normal_src)
selected_normal = random.sample(normal_files, 162)

for f in selected_normal:
    shutil.copy(os.path.join(normal_src, f), normal_dst)

# copy all abnormal (already 162)
for f in os.listdir(abnormal_src):
    shutil.copy(os.path.join(abnormal_src, f), abnormal_dst)

print("✅ Balanced dataset ready (162-162)")