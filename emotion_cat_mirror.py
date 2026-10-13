"""
Emotion Cat Mirror
Reads your facial expression from the webcam and shows a cat with the same emotion.

Setup:
    pip install deepface tf-keras opencv-python numpy

Cat photo folders (one folder per emotion, any number of photos in each):
    cat_emotions/
        happy/     angry/     sad/     surprise/
        fear/      disgust/   neutral/
Tip: use sort_cats_by_emotion.py to fill these folders automatically.

Run:
    python emotion_cat_mirror.py
Press Q to quit.
"""

import os
import random
import cv2
import numpy as np
from deepface import DeepFace

CAT_DIR = "cat_emotions"
EMOTIONS = ["happy", "angry", "sad", "surprise", "fear", "disgust", "neutral"]
ANALYZE_EVERY = 5      # analyze every Nth frame (emotion analysis is slow)
STABLE_FRAMES = 3      # emotion must repeat this many analyses before the cat changes


def load_cat_images():
    cats = {}
    for emo in EMOTIONS:
        folder = os.path.join(CAT_DIR, emo)
        if not os.path.isdir(folder):
            continue
        imgs = [cv2.imread(os.path.join(folder, f))
                for f in sorted(os.listdir(folder))]
        imgs = [i for i in imgs if i is not None]
        if imgs:
            cats[emo] = imgs
            print(f"{emo}: {len(imgs)} cat photo(s)")
    if not cats:
        print(
            f"No cat photos found in '{CAT_DIR}/<emotion>/'. Add some and run again.")
    return cats


def fit_to_height(img, height):
    h, w = img.shape[:2]
    return cv2.resize(img, (int(w * height / h), height))


def placeholder(height, text):
    panel = np.full((height, height, 3), 40, dtype=np.uint8)
    cv2.putText(panel, text, (20, height // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
    return panel


def main():
    cats = load_cat_images()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open the camera.")
        return

    frame_no = 0
    shown_emotion, candidate, streak = None, None, 0
    cat_img, face_box, scores = None, None, {}

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_no += 1

        if frame_no % ANALYZE_EVERY == 0:
            try:
                result = DeepFace.analyze(
                    frame, actions=["emotion"], detector_backend="opencv",
                    enforce_detection=True, silent=True,
                )[0]
                emo = result["dominant_emotion"]
                scores = result["emotion"]
                r = result["region"]
                face_box = (r["x"], r["y"], r["w"], r["h"])
            except ValueError:  # no face found
                emo, face_box = None, None

            # Only switch cats once the emotion is stable, to avoid flicker
            if emo == candidate:
                streak += 1
            else:
                candidate, streak = emo, 1
            if candidate and streak >= STABLE_FRAMES and candidate != shown_emotion:
                shown_emotion = candidate
                if shown_emotion in cats:
                    cat_img = random.choice(cats[shown_emotion])
                else:
                    cat_img = None

        if face_box:
            x, y, w, h = face_box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            if shown_emotion:
                conf = scores.get(shown_emotion, 0)
                cv2.putText(frame, f"{shown_emotion} ({conf:.0f}%)", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        H = frame.shape[0]
        if cat_img is not None:
            right = fit_to_height(cat_img, H)
        elif shown_emotion:
            right = placeholder(H, f"No '{shown_emotion}' cat yet")
        else:
            right = placeholder(H, "Show me a face!")

        cv2.imshow("You vs Cat", np.hstack([frame, right]))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
