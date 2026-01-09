import cv2 as cv
import os
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
source_dir = r"C:\yourSourceDirectory"
# //source directory

target_dir = os.path.join(BASE_DIR, "convertedGrayScaleDataset1")
# destination directory
os.makedirs(target_dir, exist_ok=True)

for root, dirs, files in os.walk(source_dir):
    for file in files:
        if not file.lower().endswith((".jpg", ".png")):
            continue
        img_name = os.path.splitext(file)[0]

        img_path = os.path.join(root, file)
        img = cv.imread(img_path)
        if img is None:
            print(f"Could not read image: {img_path}, skipping")
            continue

        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        resize = cv.resize(gray, (96, 96))

        rel_path = os.path.relpath(root, source_dir)
        out_folder = os.path.join(target_dir, rel_path)
        os.makedirs(out_folder, exist_ok=True)

        save_path = os.path.join(out_folder, f"{img_name}_gray.jpg")
        cv.imwrite(save_path, resize)
        print(f"Processed {img_name} -> {save_path}")

print("All images processed!")