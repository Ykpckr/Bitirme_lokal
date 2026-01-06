import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset

# =====================
# AYARLAR
# =====================
FPS = 2
WINDOW_SEC = 20
WINDOW_SIZE = FPS * WINDOW_SEC
NUM_CLASSES = 17
BACKGROUND_ID = NUM_CLASSES

LABEL2ID = {
    "Kick-off": 0,
    "Throw-in": 1,
    "Goal": 2,
    "Corner": 3,
    "Free-kick": 4,
    "Penalty": 5,
    "Offside": 6,
    "Foul": 7,
    "Yellow card": 8,
    "Red card": 9,
    "Substitution": 10,
    "Clearance": 11,
    "Shot": 12,
    "Save": 13,
    "Ball out of play": 14,
    "Direct free-kick": 15,
    "Indirect free-kick": 16,
}

# =====================
# DATASET
# =====================
class SoccerNetDataset(Dataset):
    def __init__(self, root_dir):
        self.index = []  # (feature_path, label_path, start_frame)

        print("Dataset indexleniyor...")

        for root, _, files in os.walk(root_dir):

            if "Labels-v2.json" not in files:
                continue

            npy_files = [f for f in files if f.endswith(".npy")]
            if len(npy_files) == 0:
                continue

            feat_path = os.path.join(root, npy_files[0])
            label_path = os.path.join(root, "Labels-v2.json")

            features = np.load(feat_path, mmap_mode="r")
            T, _ = features.shape

            if T < WINDOW_SIZE:
                continue

            stride = WINDOW_SIZE // 2
            for t in range(0, T - WINDOW_SIZE, stride):
                self.index.append((feat_path, label_path, t))

        print(f"Toplam window sayısı: {len(self.index)}")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        feat_path, label_path, start = self.index[idx]

        features = np.load(feat_path, mmap_mode="r")
        x = features[start:start + WINDOW_SIZE]

        cls_targets = self._parse_labels(label_path, features.shape[0])
        y = cls_targets[start:start + WINDOW_SIZE]

        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.long)
        )

    # =====================
    # LABEL PARSER
    # =====================
    def _parse_labels(self, label_path, num_frames):
        cls_targets = np.full(num_frames, BACKGROUND_ID, dtype=np.int64)

        with open(label_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for event in data.get("annotations", []):
            label = event["label"]
            if label not in LABEL2ID:
                continue

            frame = int(float(event["position"]) * FPS)
            if frame < num_frames:
                cls_targets[frame] = LABEL2ID[label]

        return cls_targets
