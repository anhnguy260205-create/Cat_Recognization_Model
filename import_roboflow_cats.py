"""
Copy the Roboflow "Cat Emotions" dataset into the cat_emotions/ folders
used by emotion_cat_mirror.py, mapping cat labels to human emotions.

Usage:
    python import_roboflow_cats.py path/to/unzipped_roboflow_folder

The Roboflow "Folder Structure" download looks like:
    train/angry/*.jpg   valid/relaxed/*.jpg   test/sad/*.jpg   ...
All splits (train/valid/test) are merged.
"""

import os
import shutil
import sys

OUT_DIR = "cat_emotions"

# Roboflow cat label -> human emotion folder(s) used by the mirror app
MAPPING = {
    "angry": ["angry"],
    "sad": ["sad"],
    "relaxed": ["happy", "neutral"],
    "attentive": ["surprise"],
    "uncomfortable": ["fear", "disgust"],
    "no clear emotion recognizable": ["neutral"],
}


def main():
    if len(sys.argv) < 2:
        print("Usage: python import_roboflow_cats.py path/to/roboflow_folder")
        return
    src = sys.argv[1]
    counts = {}

    for split in os.listdir(src):
        split_dir = os.path.join(src, split)
        if not os.path.isdir(split_dir):
            continue
        for label in os.listdir(split_dir):
            label_dir = os.path.join(split_dir, label)
            key = label.lower().replace("_", " ").replace("-", " ").strip()
            targets = MAPPING.get(key)
            if not targets or not os.path.isdir(label_dir):
                continue
            for file in os.listdir(label_dir):
                if not file.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                for emo in targets:
                    out = os.path.join(OUT_DIR, emo)
                    os.makedirs(out, exist_ok=True)
                    shutil.copy2(os.path.join(label_dir, file),
                                 os.path.join(out, f"{split}_{file}"))
                    counts[emo] = counts.get(emo, 0) + 1

    for emo, n in sorted(counts.items()):
        print(f"{emo}: {n} photos")
    if not counts:
        print(
            "No matching folders found. Check that you chose the 'Folder Structure' format.")


if __name__ == "__main__":
    main()
