"""
Sort a folder of cat photos into emotion folders automatically using CLIP.

Setup:
    pip install torch transformers pillow

Run:
    python sort_cats_by_emotion.py path/to/all_cat_photos

Photos are COPIED into cat_emotions/<emotion>/. The originals are not touched.
CLIP's guesses about cat emotions are rough, so look through the folders
afterwards and move or delete photos that don't fit.
"""

import os
import shutil
import sys
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

OUT_DIR = "cat_emotions"
MIN_CONFIDENCE = 0.35  # photos below this are put in "_unsure" for you to check

PROMPTS = {
    "happy":    "a photo of a happy, content, relaxed cat",
    "angry":    "a photo of an angry cat hissing with ears flat",
    "sad":      "a photo of a sad, gloomy cat",
    "surprise": "a photo of a surprised cat with wide eyes",
    "fear":     "a photo of a scared, frightened cat",
    "disgust":  "a photo of a disgusted cat making a face",
    "neutral":  "a photo of a calm cat with a neutral expression",
}


def main():
    if len(sys.argv) < 2:
        print("Usage: python sort_cats_by_emotion.py path/to/cat_photos")
        return
    src = sys.argv[1]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(
        "openai/clip-vit-base-patch32").to(device).eval()
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    labels = list(PROMPTS)
    texts = list(PROMPTS.values())

    for name in sorted(os.listdir(src)):
        path = os.path.join(src, name)
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            continue
        inputs = processor(text=texts, images=img,
                           return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            probs = model(**inputs).logits_per_image.softmax(dim=1)[0]
        best = int(probs.argmax())
        label = labels[best] if probs[best] >= MIN_CONFIDENCE else "_unsure"
        os.makedirs(os.path.join(OUT_DIR, label), exist_ok=True)
        shutil.copy2(path, os.path.join(OUT_DIR, label, name))
        print(f"{name} -> {label} ({probs[best]:.2f})")


if __name__ == "__main__":
    main()
