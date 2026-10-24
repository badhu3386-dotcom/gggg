"""
Build a signs file (my-signs.json) for the Sign to Voice website
from a folder of ISL videos or images.

Your dataset folder should be organised by word, e.g.
    dataset/Greetings/1. Hello/video1.mp4
    dataset/Greetings/5. Thank you/video2.mov
The script looks at every folder name and file name, and picks the
files whose name contains one of the words you ask for.

Example:
    python make_signs_file.py --dataset "C:/data/INCLUDE" ^
        --words "hello,thank you,love you,help,welcome" --out my-signs.json
"""
import argparse
import json
import math
import random
import re
import sys
from pathlib import Path

import cv2
import mediapipe as mp

VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MAX_WIDTH = 960  # shrink big frames (e.g. 4K) so processing is fast


def hand_vector(landmarks, aspect):
    base = landmarks[0]
    pts = [((q.x - base.x) * aspect, q.y - base.y, q.z - base.z) for q in landmarks]
    max_d = max(math.sqrt(x * x + y * y + z * z) for x, y, z in pts) or 1.0
    vec = []
    for pt in pts:
        for v in pt:
            vec.append(round(v / max_d * 10000) / 10000)
    return {"vec": vec, "scale": max_d, "wx": base.x * aspect, "wy": base.y}


def to_features(hand_lms, labels, aspect):
    """Same maths as toFeatures() in app.js. Do not change one without the other.
    Layout: [Left hand 63][Right hand 63][offset x, offset y] = 128 numbers."""
    items = []
    for lm, label in list(zip(hand_lms, labels))[:2]:
        d = hand_vector(lm, aspect)
        d["label"] = label
        items.append(d)
    A = B = None
    if len(items) == 2:
        if items[0]["label"] != items[1]["label"]:
            A = items[0] if items[0]["label"] == "Left" else items[1]
            B = items[1] if A is items[0] else items[0]
        else:
            A, B = (items[0], items[1]) if items[0]["wx"] <= items[1]["wx"] else (items[1], items[0])
    else:
        if items[0]["label"] == "Left":
            A = items[0]
        else:
            B = items[0]
    zeros = [0] * 63
    off_x = off_y = 0
    if A and B:
        s = max(A["scale"], B["scale"]) or 1.0
        off_x = round((B["wx"] - A["wx"]) / s * 10000) / 10000
        off_y = round((B["wy"] - A["wy"]) / s * 10000) / 10000
    return (A["vec"] if A else zeros) + (B["vec"] if B else zeros) + [off_x, off_y]


def features_from_frame(hands, frame):
    h, w = frame.shape[:2]
    if w > MAX_WIDTH:
        scale = MAX_WIDTH / w
        frame = cv2.resize(frame, (MAX_WIDTH, int(h * scale)))
        h, w = frame.shape[:2]
    result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if not result.multi_hand_landmarks:
        return None
    lms = [hnd.landmark for hnd in result.multi_hand_landmarks]
    labels = [c.classification[0].label for c in result.multi_handedness]
    return to_features(lms, labels, w / h)


def clean(name):
    name = re.sub(r"^[\d\s._-]+", "", name.lower())
    return re.sub(r"[_\-]+", " ", name).strip()


def match_word(path, root, patterns):
    parts = [clean(p) for p in path.relative_to(root).parts[:-1]] + [clean(path.stem)]
    for word, pattern in patterns.items():
        if any(pattern.search(p) for p in parts):
            return word
    return None


def frames_from_video(path, count):
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return
    step = max(total // count, 1)
    for idx in range(step // 2, total, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok:
            yield frame
    cap.release()


def main():
    ap = argparse.ArgumentParser(description="Make a signs file from an ISL dataset folder")
    ap.add_argument("--dataset", required=True, help="path to the dataset folder")
    ap.add_argument("--words", required=True, help='comma separated, e.g. "hello,thank you,help"')
    ap.add_argument("--out", default="my-signs.json")
    ap.add_argument("--max-files", type=int, default=15, help="max videos/images used per word")
    ap.add_argument("--frames", type=int, default=10, help="frames taken from each video")
    args = ap.parse_args()

    root = Path(args.dataset)
    if not root.is_dir():
        sys.exit(f"Folder not found: {root}")

    words = [w.strip() for w in args.words.split(",") if w.strip()]
    patterns = {w: re.compile(r"\b" + re.escape(w.lower()) + r"\b") for w in words}

    found = {w: [] for w in words}
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in VIDEO_EXT | IMAGE_EXT:
            w = match_word(path, root, patterns)
            if w:
                found[w].append(path)

    random.seed(0)
    samples = []
    with mp.solutions.hands.Hands(
        static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5
    ) as hands:
        for word in words:
            files = found[word]
            if not files:
                print(f"[missing] no files found for '{word}'")
                continue
            if len(files) > args.max_files:
                files = random.sample(files, args.max_files)
            label = word[0].upper() + word[1:]
            before = len(samples)
            for f in files:
                if f.suffix.lower() in IMAGE_EXT:
                    frame = cv2.imread(str(f))
                    frames = [frame] if frame is not None else []
                else:
                    frames = frames_from_video(f, args.frames)
                for frame in frames:
                    feats = features_from_frame(hands, frame)
                    if feats:
                        samples.append({"label": label, "f": feats})
            print(f"[ok] {label}: {len(files)} files -> {len(samples) - before} samples")

    Path(args.out).write_text(json.dumps(samples), encoding="utf-8")
    print(f"\nSaved {len(samples)} samples to {args.out}")
    print("Now open the website and use 'Import signs' to load this file.")


if __name__ == "__main__":
    main()
