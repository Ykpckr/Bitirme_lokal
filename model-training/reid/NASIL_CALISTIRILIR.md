# ReID Modülü - Hızlı Başlangıç Rehberi

## 📋 Kurulum

```bash
cd /home/airborne/Desktop/eray/FoMAC/model-training/reid

# Bağımlılıkları kur
pip install -r requirements.txt

# Kurulumu doğrula
python verify_installation.py
```

## 🎯 Kullanım Şekilleri

### Yöntem 1: Config Dosyasıyla (ÖNERİLEN)

1. **Config dosyasını düzenle:**
```bash
nano configs/run_config.yaml
```

```yaml
extraction:
  video: /path/to/your/match.mp4  # Video yolunu gir
  output_dir: /home/airborne/Desktop/eray/FoMAC/data/reid
  tracks: null  # Tracking varsa yolunu gir
  conf_threshold: 0.5
  frame_interval: 10

training:
  config: configs/example_config.yaml

evaluation:
  config: configs/example_config.yaml

export:
  config: configs/example_config.yaml
  test_after_export: true

pipeline:
  enable_extraction: true
  enable_training: true
  enable_evaluation: true
  enable_export: true
```

2. **Tek komutla çalıştır:**
```bash
python main.py run --run-config configs/run_config.yaml
```

Bu komut sırasıyla:
- ✅ Player crop extraction
- ✅ Model training  
- ✅ Model evaluation
- ✅ Model export

### Yöntem 2: Manuel Adım Adım

#### 1️⃣ Player Crop Extraction
```bash
python main.py extract \
    --video /path/to/match.mp4 \
    --out data/reid \
    --conf 0.5 \
    --frame-interval 10
```

**Tracking results ile:**
```bash
python main.py extract \
    --video /path/to/match.mp4 \
    --out data/reid \
    --tracks tracking_results.txt \
    --conf 0.5
```

#### 2️⃣ Model Training
```bash
python main.py train --config configs/example_config.yaml
```

**Checkpoint'ten devam et:**
```bash
python main.py train \
    --config configs/example_config.yaml \
    --resume outputs/reid/checkpoints/latest.pt
```

#### 3️⃣ Model Evaluation
```bash
python main.py eval --config configs/example_config.yaml
```

#### 4️⃣ Model Export
```bash
python main.py export --config configs/example_config.yaml --test
```

### Yöntem 3: Pipeline (Extraction Hariç)

Eğer zaten crop'lar hazırsa:

1. `configs/run_config.yaml` içinde:
```yaml
pipeline:
  enable_extraction: false  # Bu satırı false yap
  enable_training: true
  enable_evaluation: true
  enable_export: true
```

2. Çalıştır:
```bash
python main.py run --run-config configs/run_config.yaml
```

## 🔧 Yardımcı Komutlar

### Bilgileri Göster
```bash
python main.py info --config configs/example_config.yaml
```

### Modülleri Test Et
```bash
python main.py test
```

### Yardım
```bash
python main.py --help
python main.py extract --help
python main.py train --help
```

## 📂 Beklenen Dosya Yapısı

### Extraction Sonrası:
```
data/reid/
├── train/
│   ├── pid_0001/
│   │   ├── frame_000100_det_000001_conf_0.95.jpg
│   │   └── ...
│   ├── pid_0002/
│   └── ...
├── query/
│   └── pid_0001/
└── gallery/
    └── pid_0001/
```

### Training Sonrası:
```
outputs/reid/
├── checkpoints/
│   ├── best_reid.pt         ← En iyi model
│   ├── latest.pt            ← Son checkpoint
│   └── checkpoint_epoch_X.pt
└── logs/
    └── [tensorboard logs]
```

## 📊 Training'i İzle

```bash
tensorboard --logdir outputs/reid/logs
```

Tarayıcıda: http://localhost:6006

## ⚙️ Config Dosyalarını Anla

### `configs/run_config.yaml`
- Tüm çalıştırma parametreleri
- Hangi adımların çalışacağı
- Video/data yolları

### `configs/example_config.yaml` 
- Model parametreleri
- Training hiperparametreleri
- Data augmentation ayarları

## 🚨 Yaygın Sorunlar

### 1. Out of Memory
```yaml
# configs/example_config.yaml içinde:
train:
  batch_size: 16  # 32'den küçült
```

### 2. Video Bulunamadı
```bash
# Mutlak yol kullan
video: /home/airborne/Desktop/eray/FoMAC/data/videos/match.mp4
```

### 3. YOLO Weights Yok
```yaml
# configs/example_config.yaml içinde:
paths:
  yolo_weights: /tam/yol/best.pt
```

## 📝 Örnek Tam Çalıştırma

```bash
# 1. Config'i düzenle
nano configs/run_config.yaml

# Video yolunu gir:
#   video: /path/to/match.mp4

# 2. Çalıştır
python main.py run --run-config configs/run_config.yaml

# 3. Sonuçları kontrol et
ls -lh outputs/reid/checkpoints/best_reid.pt

# 4. Evaluation sonuçlarına bak
cat outputs/reid/eval_results.txt
```

## 🎓 İlk Kullanım İçin Önerilen Adımlar

1. **Test Et:**
```bash
python main.py test
```

2. **Bilgileri Gör:**
```bash
python main.py info
```

3. **Küçük Test İçin Config Düzenle:**
```yaml
train:
  epochs: 10  # Test için 10 epoch yeter
  batch_size: 16
extraction:
  max_frames: 1000  # Sadece 1000 frame
```

4. **Çalıştır:**
```bash
python main.py run --run-config configs/run_config.yaml
```

## 💡 İpuçları

- İlk defa kullanıyorsan `max_frames: 1000` ile başla
- Training sırasında `tensorboard` ile metrikleri izle
- Her 5 epoch'ta checkpoint kaydediliyor
- En iyi model otomatik olarak `best_reid.pt` olarak kaydedilir

---

**Yardım için:** `python main.py --help`
