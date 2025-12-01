# Tracking Task Tree (SoccerNet Pipeline)

- `[*]` = zorunlu (senin tasarımında)
- `[~]` = opsiyonel ama çok faydalı

---

## TRACKING (SoccerNet-Tracking MOT) `[*]`

TRACKING (SoccerNet-Tracking MOT) `[*]`
├─ 1. Detection Modülü (YOLO11 player/ball/ref) `[*]`
│  ├─ 1.1. Dataset Hazırlığı `[*]`
│  │  ├─ 1.1.1. SoccerNet-Tracking anotasyonlarını içe aktar
│  │  ├─ 1.1.2. Class tanımları (player, referee, ball, background)
│  │  └─ 1.1.3. Train/val/test split ve augmentasyon politikası
│  ├─ 1.2. Model & Config `[*]`
│  │  ├─ 1.2.1. YOLO11 mimarisi ve input pipeline
│  │  └─ 1.2.2. Hyperparametreler (imgsz, batch, lr, epochs...)
│  └─ 1.3. Eğitim & Validasyon `[*]`
│     ├─ 1.3.1. Training loop (optimizer, scheduler, logger)
│     ├─ 1.3.2. mAP, loss, overfit kontrolü
│     └─ 1.3.3. Export: inference için weights (`best.pt`)
│
├─ 2. Re-Identification Modülü `[*]`  ← Tracking’te appearance cost için şart
│  ├─ 2.1. ReID Dataset `[*]`
│  │  ├─ 2.1.1. SoccerNet ReID crop’larını okuma
│  │  ├─ 2.1.2. (Opsiyonel) Kendi YOLO deteksiyonlarınla crop üretme
│  │  └─ 2.1.3. ID label’ları ve split yapısı (train/query/gallery)
│  ├─ 2.2. Model & Loss `[*]`
│  │  ├─ 2.2.1. Backbone (ResNet/Swin/YOLO-head tabanlı embedding)
│  │  ├─ 2.2.2. Loss fonksiyonları (Cross-Entropy, Triplet vb.)
│  │  └─ 2.2.3. Embedding normalizasyonu (L2, cosine space vs.)
│  ├─ 2.3. Eğitim & Değerlendirme `[*]`
│  │  ├─ 2.3.1. Augmentations (rotation, perspective, erasing…)
│  │  ├─ 2.3.2. Retrieval mAP / Rank@K ölçümü
│  │  └─ 2.3.3. En iyi modeli export (`best_reid.pt`)
│  └─ 2.4. Tracking’e Entegrasyon `[*]`
│     ├─ 2.4.1. Frame içinde player bbox → crop → embedding hesaplama
│     ├─ 2.4.2. Cost matrix’e appearance terimi ekleme  
│     │       cost = α · (1 − IoU) + β · dist(embedding)
│     └─ 2.4.3. α, β, `max_distance` hyperparam tuning
│
├─ 3. Tracker Çekirdeği (Association + State) `[*]`
│  ├─ 3.1. Track State Tasarımı `[*]`
│  │  ├─ 3.1.1. Track struct: `bbox`, `score`, `class_id`, `embedding`, `age`, `hits`
│  │  └─ 3.1.2. ID generator & track lifecycle (birth / confirmed / dead)
│  ├─ 3.2. Association Mekanizması `[*]`
│  │  ├─ 3.2.1. IoU + appearance distance cost matrix
│  │  ├─ 3.2.2. Hungarian / linear assignment çözümü
│  │  └─ 3.2.3. Gating (IoU min, embedding `max_distance`)
│  ├─ 3.3. Motion Model (Kalman) `[~]`
│  │  ├─ 3.3.1. State tanımı `[x, y, w, h, vx, vy, vw]`
│  │  ├─ 3.3.2. Predict / Update fonksiyonları
│  │  └─ 3.3.3. Noise kovaryanslarının tune edilmesi
│  ├─ 3.4. Occlusion & Yaşlandırma `[*]`
│  │  ├─ 3.4.1. `max_age`, `min_hits`, score threshold ayarları
│  │  └─ 3.4.2. Kitleme (occlusion) senaryoları için test setleri
│  └─ 3.5. Değerlendirme `[*]`
│     ├─ 3.5.1. MOTA, MOTP, IDF1, ID switches hesaplama
│     └─ 3.5.2. SoccerNet-Tracking devkit ile sonuç karşılaştırma
│
└─ 4. Destekleyici & Çevre Task’lar
   ├─ 4.1. Camera Shot Segmentation `[~]`  ← Scene-change handling için
   │  ├─ 4.1.1. Video → frame/feature çıkarma
   │  ├─ 4.1.2. Shot boundary / shot class modeli
   │  └─ 4.1.3. Tracker ile entegrasyon
   │     ├─ 4.1.3.1. Yeni sahnede tracker reset / soft reset
   │     └─ 4.1.3.2. Replay vs live ayrımı (tracking’i durdur/başlat)
   │
   └─ 4.2. (DOWNSTREAM) Camera Calibration + Field Localization `[~]`
      ├─ 4.2.1. Pitch Marking & Goal Post Detection `[~]`
      │  ├─ 4.2.1.1. Çizgi / kale tespiti için model
      │  └─ 4.2.1.2. Pitch template ile eşleme
      ├─ 4.2.2. Camera Calibration `[~]`
      │  ├─ 4.2.2.1. Homografi / PnP solver
      │  └─ 4.2.2.2. Kalibrasyon parametrelerinin saklanması
      └─ 4.2.3. Field Localization `[~]`
         ├─ 4.2.3.1. Tracking output (ID + bbox) → pitch koordinatı
         └─ 4.2.3.2. Minimap, heatmap, koşu mesafesi gibi istatistikler
