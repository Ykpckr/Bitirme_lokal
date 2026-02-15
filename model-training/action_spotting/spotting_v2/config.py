import os
from pathlib import Path
import torch

# ==============================================================================
# 1. PATH SETTINGS (DOSYA YOLLARI)
# ==============================================================================
# Veri seti kök dizini (Baidu/ResNet özelliklerinin olduğu yer)
DATASET_DIR = Path("C:/FoMAC_Dataset/action_spotting")

# Model kayıt yeri
CHECKPOINT_DIR = Path("./checkpoints")
LOG_DIR = Path("./logs")

# ==============================================================================
# 2. FEATURE SETTINGS (ÖZELLİK AYARLARI)
# ==============================================================================
# "baidu" (8576 dim, 1 FPS) veya "resnet" (2048 dim, 2 FPS)
FEATURE_TYPE = "baidu" 

if FEATURE_TYPE == "baidu":
    FEATURE_DIM = 8576
    FPS = 1
    # Baidu dosyaları genelde "1_baidu_soccer_embeddings.npy" gibi isimlendirilir
    FEATURE_SUFFIX = "_baidu_soccer_embeddings.npy"
elif FEATURE_TYPE == "resnet":
    FEATURE_DIM = 512 # PCA sonrası (Genelde 2048 veya 512)
    FPS = 2
    FEATURE_SUFFIX = "_ResNET_TF2_PCA512.npy"

# Pencere Boyutu (Saniye cinsinden)
WINDOW_SIZE_SEC = 20 
# Pencere Boyutu (Frame cinsinden)
WINDOW_SIZE_FRAMES = int(WINDOW_SIZE_SEC * FPS)

# ==============================================================================
# 3. CLASS SETTINGS (SINIF AYARLARI)
# ==============================================================================
# SoccerNet-v2 (17 Sınıf)
EVENT_DICTIONARY = {
    "Kick-off": 0, "Throw-in": 1, "Goal": 2, "Corner": 3, "Free-kick": 4, 
    "Penalty": 5, "Offside": 6, "Foul": 7, "Yellow card": 8, "Red card": 9, 
    "Substitution": 10, "Clearance": 11, "Shot": 12, "Save": 13, 
    "Ball out of play": 14, "Direct free-kick": 15, "Indirect free-kick": 16
}
# Ters sözlük (ID -> İsim)
ID_TO_EVENT = {v: k for k, v in EVENT_DICTIONARY.items()}

NUM_CLASSES = 17
BACKGROUND_CLASS = 17 # 17. index background olacak (Toplam 18 sınıf)

USE_CLASS_WEIGHTS = True

# ==============================================================================
# 4. MODEL SETTINGS (MİMARİ)
# ==============================================================================
# "transformer" veya "cnn" (NetVLAD++)
MODEL_TYPE = "cnn" 

# CNN / NetVLAD Ayarları
PROJECTION_DIM = 512       # 8576 -> 512 Reduction
NETVLAD_CLUSTERS = 16      # K=16 (4GB VRAM için güvenli, PDF önerisi: 64)
NETVLAD_GHOST_CLUSTERS = 0 # İleri seviye

# ==============================================================================
# 5. TRAINING SETTINGS (EĞİTİM)
# ==============================================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 8          # Hız ve stabilite dengesi
ACCUMULATION_STEPS = 2  # Sanal Batch Size = 16
EPOCHS = 20
LEARNING_RATE = 1e-4    # CNN için
WEIGHT_DECAY = 1e-4

# Masking (RMS-Net taktiği)
MASK_PROB = 0.5 

# Loss Ayarları
LAMBDA_REG = 10.0       # Regresyon kaybının ağırlığı
IGNORE_RADIUS = 1       # Olayın +/- 1 saniye yanını "Ignore" et (Loss hesaplama)

# DataLoader
NUM_WORKERS = 0         # Windows için 0
PIN_MEMORY = True
