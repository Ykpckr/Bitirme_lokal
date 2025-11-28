# Player Tracking & Re-Identification

Bu modül, **SoccerNet ReID** dataset'i ve **SNMOT** formatındaki tracking dataset'lerini kullanarak re-identification destekli çoklu oyuncu takibini FoMAC projesine entegre eder. Pipeline, mevcut **player/ball/referee** YOLO modelinizin (`models/player_ball_detector/weights/best.pt`) çıktıları üzerinde inşa edilir.

## 🚧 Mimari Özeti

- **Algı (Detection):** `best.pt` modeli Ultralytics YOLO arayüzü ile her karede oyuncu/top/hakem tespiti üretir.
- **Özellik Öğrenimi (ReID):** SoccerNet ReID dataset'inden türetilen özel bir ResNet tabanlı embedder ile oyuncu kimlik vektörleri üretilir.
- **Takip:** YOLO'nun StrongSORT/ByteTrack arayüzü ile ID eşleşmesi yapılır, MOTChallenge formatında sonuçlar yazılır.

```
Frames -> YOLO best.pt -> Detections -> StrongSORT (+ReID embeddings) -> Tracks (.txt)
                             ↑
                      train_reid.py (SoccerNet)
```

## 📦 Bağımlılıklar

```powershell
cd model-training/tracking
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Başlıca paketler: `ultralytics`, `torch`, `torchvision`, `opencv-python`, `lap`, `filterpy`, `scipy`, `pyyaml`.

## 🗂️ Dataset Hazırlığı

### 1. Tracking Dataset (SNMOT / MOT formatı)

1. Tüm SNMOT sekansları `c:\Users\kaan.aggunlu\Desktop\git\FoMAC\tracking` altında zaten `train/` ve `test/` klasörlerine ayrılmış durumda (örnek: `train/SNMOT-060`, `test/SNMOT-197`).
2. Her sekans aşağıdaki yapıya sahip olmalıdır:
   ```
   SNMOT-060/
     ├─ seqinfo.ini
     ├─ gameinfo.ini
     ├─ img1/
     ├─ det/det.txt (opsiyonel)
     └─ gt/gt.txt  (opsiyonel)
   ```
3. `configs/tracking.yaml` içindeki `datasets.root` alanını bu dizine (ya da eşdeğerine) işaret edecek şekilde güncelleyin ve `datasets.splits` listesi ile hangi alt klasörlerin otomatik taranacağını belirtin (varsayılan: `train` + `test`).
4. `datasets.sequences` alanını `null` bırakırsanız belirtilen split'lerdeki **tüm** sekanslar otomatik olarak işlenecektir.

> Not: Train klasörü ağırlıklı olarak `SNMOT-060`–`SNMOT-077` ve `SNMOT-097`–`SNMOT-170` dizilerini içeriyor; test klasörü ise `SNMOT-116`–`SNMOT-200` aralığındaki ~70 maç klibini barındırıyor. Bu yapı MOTChallenge formatına uyumlu olduğu sürece direkt kullanılabilir.

### 2. SoccerNet ReID Dataset'i

1. [sn-reid](https://github.com/SoccerNet/sn-reid) yönergelerini izleyerek `train/ valid/ test/ challenge` klasörlerini indirin.
2. Bu klasörleri şu yapıda tutun:
   ```
   <datasets_root>/soccernetv3/reid/{train,valid,test,challenge}
   ```
3. `configs/reid.yaml` içindeki `reid_dataset.root` alanını bu yol ile değiştirin.

## ⚙️ Konfigürasyon Dosyaları

| Dosya | Amaç |
|-------|------|
| `configs/reid.yaml` | ReID eğitimi için dataset yolları, hiperparametreler |
| `configs/tracking.yaml` | YOLO tracking, StrongSORT ve çıkış yolları |
| `configs/trackers/strongsort_soccer.yaml` | StrongSORT ayarları + ReID ağı ağırlıkları |

## 🧠 ReID Eğitimi

```powershell
cd model-training/tracking
python train_reid.py --config configs/reid.yaml
```

- Çıktılar `outputs/reid/` altında saklanır (`best_reid.pt`).
- Config üzerinden `epochs`, `batch_size`, `image_size` gibi değerleri değiştirebilirsiniz.

## 🎯 Tracking Çalıştırma

```powershell
cd model-training/tracking
python run_tracking.py --config configs/tracking.yaml
```

- YOLO modeli `models/player_ball_detector/weights/best.pt` dosyasından yüklenir.
- Her sekans için `outputs/tracks/<sequence>.txt` dosyası oluşturulur (MOTChallenge formatı: `frame,id,x,y,w,h,conf,-1,-1,-1`).
- `tracking.save_visualizations = true` ayarı ile `runs/track/` altında video/frame kayıtları saklanır.

## 📁 Klasör Yapısı

```
tracking/
├── configs/
│   ├── reid.yaml
│   ├── tracking.yaml
│   └── trackers/strongsort_soccer.yaml
├── modules/
│   ├── __init__.py
│   ├── datasets.py
│   ├── reid_training.py
│   └── tracker_runner.py
├── requirements.txt
├── run_tracking.py
└── train_reid.py
```

## ✅ Gelecek Adımlar

- `tools/evaluate_mot.py`: GT ile IDF1/IDSW hesaplamak için eklenebilir.
- Daha hafif ReID mimarileri (EfficientNet tabanlı) eklenip `configs/reid.yaml` üzerinden seçilebilir.
- Çoklu GPU / DDP desteği için `train_reid.py` genişletilebilir.

Tüm pipeline tek config üzerinden yönetilebilir durumdadır; varsayılan ayarlar FoMAC içindeki `best.pt` ağırlığını ve örnek dataset yollarını referans alır.
