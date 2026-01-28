# import cv2 as cv
import os
import json
from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator, img_to_array, load_img
)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
source_dir = r"C:\Users\lENOVO\Downloads\MInor\Final Datset\Four_finger"
target_dir = os.path.join(BASE_DIR, "augmentedData")

os.makedirs(target_dir, exist_ok=True)

datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

AUG_PER_IMAGE = 20

for dirpath, subDir, files in os.walk(source_dir):
    rel_path = os.path.relpath(dirpath, source_dir)
    save_dir = os.path.join(target_dir, rel_path)
    os.makedirs(save_dir, exist_ok=True)
    for file in files:
        if not file.lower().endswith((".jpg", ".png")):
            continue
        img_name = os.path.splitext(file)[0]

        img_path = os.path.join(dirpath, file)
        img = load_img(img_path)
        if img is None:
            print(f"Could not read image: {img_path}, skipping")
            continue

        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)

        i = 0
        for batch in datagen.flow(x, batch_size=1,
                                save_to_dir=save_dir, save_prefix=img_name, save_format='jpeg'):
            i += 1
            if i > AUG_PER_IMAGE:
                break 
        print(f"Processed {img_name} -> {save_dir}")

print("All images processed!")