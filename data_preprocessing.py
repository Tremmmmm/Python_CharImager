# File: data_preprocessing.py
import os
import shutil
import json
from PIL import Image
from sklearn.model_selection import train_test_split

def organize_and_split_by_filename(image_root, annotation_root, output_class_dir, output_split_dir, val_ratio=0.1, test_ratio=0.1):
    print("[STEP 1] Organizing images by label using JSON filenames...")
    os.makedirs(output_class_dir, exist_ok=True)
    count = 0

    for label_folder in os.listdir(annotation_root):
        label_path = os.path.join(annotation_root, label_folder)
        if not os.path.isdir(label_path):
            continue
        for json_file in os.listdir(label_path):
            if not json_file.endswith(".json"):
                continue
            image_file = json_file.replace(".json", ".jpg")
            src_img_path = os.path.join(image_root, label_folder, image_file)
            dst_label_dir = os.path.join(output_class_dir, label_folder)
            os.makedirs(dst_label_dir, exist_ok=True)
            dst_img_path = os.path.join(dst_label_dir, image_file)
            if os.path.exists(src_img_path):
                shutil.copy(src_img_path, dst_img_path)
                count += 1
                if count % 500 == 0:
                    print(f"  Copied {count} images...")
            else:
                print(f"[WARNING] Missing image: {src_img_path}")

    print(f"[DONE] Total organized images: {count}\n")

    print("[STEP 2] Splitting dataset into train/val/test...")
    class_names = os.listdir(output_class_dir)
    for class_name in class_names:
        class_path = os.path.join(output_class_dir, class_name)
        images = [f for f in os.listdir(class_path) if f.endswith(('jpg', 'png', 'jpeg'))]
        print(f"Class {class_name}: {len(images)} images")
        if not images:
            continue
        train_imgs, valtest_imgs = train_test_split(images, test_size=val_ratio+test_ratio, random_state=42)
        val_imgs, test_imgs = train_test_split(valtest_imgs, test_size=test_ratio/(val_ratio+test_ratio), random_state=42)

        for phase, img_list in zip(['train', 'val', 'test'], [train_imgs, val_imgs, test_imgs]):
            dest_dir = os.path.join(output_split_dir, phase, class_name)
            os.makedirs(dest_dir, exist_ok=True)
            for img in img_list:
                shutil.copy(os.path.join(class_path, img), os.path.join(dest_dir, img))
            print(f"  -> {phase.upper()} - {class_name}: {len(img_list)} images")

    print("[DONE] Dataset splitting completed.\n")

if __name__ == "__main__":
    image_root = "ICPR2022_CHARTINFO_UB_PMC_TRAIN_v1.0/ICPR2022_CHARTINFO_UB_PMC_TRAIN_v1.0/images"
    annotation_root = "ICPR2022_CHARTINFO_UB_PMC_TRAIN_v1.0/ICPR2022_CHARTINFO_UB_PMC_TRAIN_v1.0/annotations_JSON"
    output_class_dir = "data_by_class"
    output_split_dir = "data_by_class_split"

    organize_and_split_by_filename(image_root, annotation_root, output_class_dir, output_split_dir)
