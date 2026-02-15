# src/config.py
import torch

ROOT_DIR = r"C:\FoMAC_Dataset\action_spotting"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- VERİ SETİ ---
FEATURE_DIM = 8576
FRAMERATE = 2.0
WINDOW_SIZE_SEC = 20
WINDOW_FRAME = int(WINDOW_SIZE_SEC * FRAMERATE) # 40 Frame

# --- SOTA TEKNİĞİ: GAUSSIAN LABELS ---
# Etiketi sadece 1 kareye değil, etrafına yayacağız.
# Sigma=2 demek, olayın +/- 2 saniye etrafı "sıcak bölge" demektir.
LABEL_SIGMA = 2.0 

EVENT_DICTIONARY = {
    "Penalty": 0, "Kick-off": 1, "Goal": 2, "Substitution": 3, "Offside": 4, 
    "Shots on target": 5, "Shots off target": 6, "Clearance": 7, "Corner": 8, 
    "Foul": 9, "Yellow card": 10, "Red card": 11, "Yellow->Red card": 12, 
    "Ball out of play": 13, "Throw-in": 14, "Indirect free-kick": 15, 
    "Direct free-kick": 16
}
NUM_CLASSES = 17

BATCH_SIZE = 8       # Düşük VRAM için 8 ideal
LEARNING_RATE = 1e-4 # Hassas öğrenme
EPOCHS = 50