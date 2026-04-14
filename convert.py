import os
import pydicom
import cv2
import numpy as np

# input folders
normal_input = "dataset/normal"
abnormal_input = "dataset/abnormal"

# output folders
normal_output = "dataset/processed/NORMAL"
abnormal_output = "dataset/processed/ABNORMAL"

os.makedirs(normal_output, exist_ok=True)
os.makedirs(abnormal_output, exist_ok=True)

def convert_folder(input_folder, output_folder):
    for file in os.listdir(input_folder):
        if file.endswith(".dicom"):
            path = os.path.join(input_folder, file)

            ds = pydicom.dcmread(path)
            img = ds.pixel_array

            # normalize
            img = (img - np.min(img)) / (np.max(img) - np.min(img))
            img = (img * 255).astype(np.uint8)

            # resize
            img = cv2.resize(img, (224, 224))

            # save
            save_path = os.path.join(output_folder, file.replace(".dicom", ".png"))
            cv2.imwrite(save_path, img)

convert_folder(normal_input, normal_output)
convert_folder(abnormal_input, abnormal_output)

print("✅ Conversion Done")