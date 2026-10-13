"""
Emotion Cat Mirror
Reads your facial expression from the webcam and shows a cat with the same emotion.

Setup:
    pip install hsemotion-onnx onnx opencv-contrib-python numpy

Cat photo folders (one folder per emotion, any number of photos in each):
    cat_emotions/
        happy/     angry/     sad/     surprise/
        fear/      disgust/   neutral/
Tip: use sort_cats_by_emotion.py to fill these folders automatically.

Run:
    python emotion_cat_mirror.py              # uses cat_emotions/
    python emotion_cat_mirror.py cat_memes    # uses meme images instead
Press Q to quit.
"""

import argparse
import os
import random
import urllib.request
from collections import deque
import cv2
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("folder", nargs="?", default="cat_emotions",
                    help="folder with one sub-folder of images per emotion")
ARGS = parser.parse_args()

CAT_DIR = ARGS.folder
EMOTIONS = ["happy", "angry", "sad", "surprise", "fear", "disgust", "neutral"]
YUNET_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/"
             "face_detection_yunet/face_detection_yunet_2023mar.onnx")
YUNET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "models", "face_detection_yunet_2023mar.onnx")
HSE_MODEL = "enet_b2_7"  # HSEmotion model trained on AffectNet, same 7 emotions
HSE_LABELS = {"Anger": "angry", "Disgust": "disgust", "Fear": "fear", "Happiness": "happy",
              "Neutral": "neutral", "Sadness": "sad", "Surprise": "surprise"}
FACE_MARGIN = 0.1      # extra space around the detected face before reading emotion
ANALYZE_EVERY = 2      # analyze every Nth frame
SMOOTH_OVER = 6        # average the emotion scores over this many analyses
MIN_CONFIDENCE = 35    # if no emotion scores above this (%), treat it as neutral
# Multiply raw scores to correct a model's bias; lower a value if that
# emotion shows up too often.
WEIGHTS = {"happy": 1.0, "angry": 1.0, "sad": 1.0, "surprise": 1.0,
           "fear": 1.0, "disgust": 1.0, "neutral": 1.0}


def read_image(path):
    img = cv2.imread(path)
    if img is None and path.lower().endswith(".gif"):  # OpenCV can't read GIFs
        cap = cv2.VideoCapture(path)  # take the first frame
        ok, img = cap.read()
        cap.release()
        img = img if ok else None
    return img


def load_cat_images():
    cats = {}
    for emo in EMOTIONS:
        folder = os.path.join(CAT_DIR, emo)
        if not os.path.isdir(folder):
            continue
        imgs = [read_image(os.path.join(folder, f))
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


class HSEmotionAnalyzer:
    """YuNet finds the face, HSEmotion reads the emotion."""

    def __init__(self):
        from hsemotion_onnx.facial_emotions import HSEmotionRecognizer
        if not os.path.isfile(YUNET_PATH):
            os.makedirs(os.path.dirname(YUNET_PATH), exist_ok=True)
            urllib.request.urlretrieve(YUNET_URL, YUNET_PATH)
        self.detector = cv2.FaceDetectorYN_create(YUNET_PATH, "", (320, 320), 0.8)
        self.fer = HSEmotionRecognizer(HSE_MODEL)
        self.labels = [HSE_LABELS[self.fer.idx_to_class[i]]
                       for i in range(len(self.fer.idx_to_class))]

    def analyze(self, frame):
        """Return (scores in %, face box) for the biggest face, or None."""
        H, W = frame.shape[:2]
        self.detector.setInputSize((W, H))
        _, faces = self.detector.detect(frame)
        if faces is None or len(faces) == 0:
            return None
        x, y, w, h = (int(v) for v in max(faces, key=lambda f: f[2] * f[3])[:4])
        m = int(FACE_MARGIN * max(w, h))
        x1, y1 = max(x - m, 0), max(y - m, 0)
        x2, y2 = min(x + w + m, W), min(y + h + m, H)
        if x2 <= x1 or y2 <= y1:
            return None
        face = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
        _, probs = self.fer.predict_emotions(face, logits=False)
        scores = {label: 100 * float(p) for label, p in zip(self.labels, probs)}
        return scores, (x, y, w, h)


def smoothed_emotion(history):
    """Average the recent score dicts and pick the top emotion."""
    avg = {e: np.mean([h[e] for h in history]) * WEIGHTS[e] for e in EMOTIONS}
    total = sum(avg.values()) or 1
    avg = {e: 100 * v / total for e, v in avg.items()}
    best = max(avg, key=avg.get)
    if avg[best] < MIN_CONFIDENCE:
        best = "neutral"
    return best, avg


def draw_scores(frame, scores, chosen):
    """Small bar chart of all emotion scores in the top-left corner."""
    for i, emo in enumerate(EMOTIONS):
        y = 20 + i * 22
        v = scores.get(emo, 0)
        color = (0, 255, 0) if emo == chosen else (200, 200, 200)
        cv2.rectangle(frame, (90, y - 12), (90 + int(v * 1.5), y + 4), color, -1)
        cv2.putText(frame, f"{emo}", (5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        cv2.putText(frame, f"{v:.0f}%", (95 + int(v * 1.5), y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)


def main():
    cats = load_cat_images()
    analyzer = HSEmotionAnalyzer()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open the camera.")
        return

    frame_no = 0
    history = deque(maxlen=SMOOTH_OVER)
    shown_emotion, cat_img, face_box, scores = None, None, None, {}

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)  # mirror view
        frame_no += 1

        if frame_no % ANALYZE_EVERY == 0:
            found = analyzer.analyze(frame)
            if found:
                face_scores, face_box = found
                history.append(face_scores)
            else:
                face_box = None
                history.clear()

            if history:
                emo, scores = smoothed_emotion(history)
                if emo != shown_emotion:
                    shown_emotion = emo
                    cat_img = random.choice(cats[emo]) if emo in cats else None

        if face_box:
            x, y, w, h = face_box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            if shown_emotion:
                cv2.putText(frame, f"{shown_emotion} ({scores[shown_emotion]:.0f}%)",
                            (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            draw_scores(frame, scores, shown_emotion)

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
