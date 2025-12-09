import numpy as np
import logging
from .kalman_filter import KalmanFilter
from .tracker_utils import compute_cosine_distance

logger = logging.getLogger(__name__)

class Track:
    def __init__(self, mean, covariance, track_id, feature, init_frame=0):
        self.mean = mean
        self.covariance = covariance
        self.track_id = int(track_id)
        self.feature = feature
        self.last_feature = feature
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.last_seen_frame = init_frame
        self.is_confirmed = False

    def update(self, kf, detection, feature, frame_id=None):
        self.mean, self.covariance = kf.update(self.mean, self.covariance, detection)
        if feature is not None:
            self.feature = feature
            self.last_feature = feature
        self.hits += 1
        self.time_since_update = 0
        if frame_id is not None:
            self.last_seen_frame = frame_id
        self.is_confirmed = True

    def predict(self, kf):
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1

    def to_xyxy(self):
        cx, cy, w, h = self.mean[:4]
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        return [float(x1), float(y1), float(x2), float(y2)]

    def center(self):
        x1, y1, x2, y2 = self.to_xyxy()
        return ((x1 + x2) / 2, (y1 + y2) / 2)

class DeepSortTracker:
    def __init__(self, max_age=30, max_cosine=0.5, max_spatial_dist=150.0):
        self.kf = KalmanFilter()
        self.tracks = []
        self._next_id = 1
        self.max_age = max_age
        self.max_cosine = max_cosine
        self.max_spatial_dist = max_spatial_dist

    def _bbox_to_z(self, xyxy):
        x1, y1, x2, y2 = xyxy
        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)
        return np.array([x1 + w / 2, y1 + h / 2, w, h], float)

    def _fallback_feature(self, xyxy):
        """ReID modeli olmadığında kullanılan geometrik özellik."""
        x1, y1, x2, y2 = xyxy
        w, h = x2 - x1, y2 - y1
        vec = np.array([x1 + w/2, y1 + h/2, w, h], float)
        norm = np.linalg.norm(vec) + 1e-8
        return (vec / norm).astype(np.float32)

    def update(self, detections, features=None, frame_id=None):
        # 1. Tahmin (Predict)
        for t in self.tracks:
            t.predict(self.kf)

        # Tespit yoksa temizle ve dön
        if len(detections) == 0:
            self._prune()
            return [(t.track_id, t.to_xyxy()) for t in self.tracks if t.is_confirmed]

        dets = np.asarray(detections, float)
        
        # Feature yoksa fallback kullan (Geometrik takip)
        if features is None:
            features = np.stack([self._fallback_feature(b) for b in dets])
        else:
            features = np.asarray(features, float)

        # 2. Eşleştirme (Matching) - İlk karede direkt ata
        if len(self.tracks) == 0:
            for i in range(len(dets)):
                z = self._bbox_to_z(dets[i][:4])
                self.tracks.append(Track(self.kf.initiate(z)[0], self.kf.initiate(z)[1], self._next_id, features[i], frame_id))
                self._next_id += 1
            return [(t.track_id, t.to_xyxy()) for t in self.tracks]

        # Cost Matrix
        track_feats = np.stack([t.last_feature for t in self.tracks], axis=0)
        cost = compute_cosine_distance(track_feats, features)
        
        assigned_dets = set()
        
        # Basit Greedy Eşleştirme (Hungarian algoritması yerine)
        for ti in range(len(self.tracks)):
            if len(cost[ti]) == 0: continue
            di = int(np.argmin(cost[ti]))
            dist = cost[ti, di]
            
            # Eşik Değer Kontrolleri
            if dist > self.max_cosine: continue
            if di in assigned_dets: continue
            
            # Uzamsal Kontrol (Çok uzaktaki nesneyle eşleşmeyi önle)
            det_c = ((dets[di,0]+dets[di,2])/2, (dets[di,1]+dets[di,3])/2)
            tr_c = self.tracks[ti].center()
            if np.linalg.norm(np.array(det_c) - np.array(tr_c)) > self.max_spatial_dist:
                continue

            # Güncelle
            z = self._bbox_to_z(dets[di][:4])
            self.tracks[ti].update(self.kf, z, features[di], frame_id)
            assigned_dets.add(di)

        # 3. Yeni İzler Oluştur
        for di in range(len(dets)):
            if di not in assigned_dets:
                z = self._bbox_to_z(dets[di][:4])
                self.tracks.append(Track(self.kf.initiate(z)[0], self.kf.initiate(z)[1], self._next_id, features[di], frame_id))
                self._next_id += 1

        self._prune()
        # trackers/deepsort_tracker.py dosyasının EN ALTI (update fonksiyonu sonu)
    
    # ESKİSİ:
    # return [(t.track_id, t.to_xyxy()) for t in self.tracks if t.is_confirmed]

    # YENİSİ (Bunu yapıştır):
    # Sadece son 1 karede güncellenmiş (yani gerçekten görülmüş) olanları döndür
        return [(t.track_id, t.to_xyxy()) for t in self.tracks if t.is_confirmed and t.time_since_update <= 1]

    def _prune(self):
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]