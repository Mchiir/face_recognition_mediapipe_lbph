#!/usr/bin/env python3
"""
03_train_model.py
Train an LBPH face recognizer from the dataset and save model + label map.
"""

import os
import json
import cv2
import numpy as np
from pathlib import Path

DATASET_DIR = Path("dataset")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "lbph_model.xml"
LABEL_MAP_PATH = MODEL_DIR / "label_map.json"

def load_dataset(root: Path):
    images = []
    labels = []
    label_map = {}
    current_label = 0

    if not root.exists():
        print(f"[error] Dataset folder '{root}' does not exist.")
        return images, labels, label_map

    for person in sorted(os.listdir(root)):
        person_path = root / person
        if not person_path.is_dir():
            continue

        label_map[current_label] = person
        for img_name in sorted(os.listdir(person_path)):
            img_path = person_path / img_name
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            images.append(img)
            labels.append(current_label)
        current_label += 1

    return images, np.array(labels, dtype=np.int32), label_map

def train_and_save(images, labels, label_map, model_path: Path, label_map_path: Path):
    if not images or labels.size == 0:
        print("[error] No training data found.")
        return False

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(images, labels)
    recognizer.save(str(model_path))

    with open(str(label_map_path), "w") as f:
        json.dump(label_map, f)

    print("\nTraining completed!")
    print(f"Model saved to {model_path}")
    print(f"Label map saved to {label_map_path}")
    return True

def main():
    images, labels, label_map = load_dataset(DATASET_DIR)
    train_and_save(images, labels, label_map, MODEL_PATH, LABEL_MAP_PATH)

if __name__ == "__main__":
    main()