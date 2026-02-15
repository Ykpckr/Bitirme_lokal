import numpy as np
from .kalman_filter import KalmanFilter
from .track import Track
from . import matching

class Tracker:
    def __init__(self, cfg):
        self.cfg = cfg
        # Config değerlerini yükle
        self.max_iou_distance = cfg.tracker.get('max_iou_distance', 0.7)
        self.max_age = cfg.tracker.get('max_age', 30)
        self.n_init = cfg.tracker.get('n_init', 3)
        self.max_dist = cfg.tracker.get('max_dist', 0.2) # ReID eşiği

        self.kf = KalmanFilter()
        self.tracks = []
        self._next_id = 1

    def predict(self):
        """ Mevcut tüm izleri bir adım ileri (geleceğe) taşır """
        for track in self.tracks:
            track.predict(self.kf)

    def update(self, detections, embeddings):
        """
        Ana güncelleme fonksiyonu.
        
        Args:
            detections: [x1, y1, x2, y2, score, class_id]
            embeddings: [N, 512] ReID vektörleri
        """
        # 1. Veri Hazırlığı (Detection formatını Kalman formatına çevir: cx, cy, aspect_ratio, h)
        tlwh_dets = []
        for det in detections:
            x1, y1, x2, y2 = det[:4]
            w, h = x2 - x1, y2 - y1
            tlwh_dets.append([x1 + w/2, y1 + h/2, w/h, h]) # Merkez, AspectRatio, Height
        
        # Detaylı veri yapısı: [tlwh, score, class_id, original_bbox, feature]
        full_dets = []
        for i, det in enumerate(detections):
            full_dets.append([
                tlwh_dets[i],   # 0: Kalman measurement
                det[4],         # 1: Score
                det[5],         # 2: Class ID
                det[:4],        # 3: Original bbox
                embeddings[i]   # 4: Feature
            ])

        # 2. Eşleştirme (Matching)
        confirmed_tracks = [i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

        # A) Cascade Matching (Önce ReID + Kalman Gating) - Sadece Confirmed olanlar
        # (Bu, DeepSORT'un kalbidir)
        matches_a, unmatched_tracks_a, unmatched_dets_a = \
            self._match_cascade(full_dets, confirmed_tracks)

        # B) IoU Matching (ReID'in yetmediği veya yeni/onaysız trackler için)
        iou_track_candidates = [k for k in unmatched_tracks_a if self.tracks[k].time_since_update == 1] + unconfirmed_tracks
        unmatched_dets_candidates = unmatched_dets_a
        
        matches_b, unmatched_tracks_b, unmatched_dets_b = \
            self._match_iou(full_dets, iou_track_candidates, unmatched_dets_candidates)

        # Tüm eşleşmeleri birleştir
        matches = list(matches_a) + list(matches_b)
        
        # Son kalan eşleşmeyen trackler
        unmatched_tracks = list(set(unmatched_tracks_a) - set(k for k, _ in matches_b)) # Confirmed ama eşleşmeyenler
        unmatched_tracks += list(set(unconfirmed_tracks) - set(k for k, _ in matches_b)) # Unconfirmed ve eşleşmeyenler

        # 3. İzleri Güncelleme
        for t_idx, d_idx in matches:
            t = self.tracks[t_idx]
            d = full_dets[d_idx]
            t.update(self.kf, d[0], d[4]) # Kalman update + feature update

        # 4. Kayıp İzleri İşleme
        for t_idx in unmatched_tracks:
            self.tracks[t_idx].mark_missed()

        # 5. Yeni İzler Oluşturma
        for d_idx in unmatched_dets_b:
            self._initiate_track(full_dets[d_idx])

        # 6. Silinmişleri Temizle
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # 7. Çıktı Üretme
        # Sadece CONFIRMED olanları döndür
        outputs = []
        for t in self.tracks:
            if t.is_confirmed():
                bbox = t.to_tlbr()
                outputs.append({
                    'track_id': t.track_id,
                    'bbox': bbox,
                    'class_id': 0 # Şimdilik hepsi oyuncu varsayıyoruz
                })
        return outputs

    def _match_cascade(self, detections, track_indices):
        """ ReID distance kullanarak yaşa (age) göre öncelikli eşleştirme """
        if len(track_indices) == 0 or len(detections) == 0:
            return [], track_indices, list(range(len(detections)))

        # Cost matrix hesapla (ReID Cosine Distance)
        # Sadece ilgili trackleri gönder
        tracks_subset = [self.tracks[i] for i in track_indices]
        cost_matrix = matching.embedding_distance(tracks_subset, detections)

        # Gating (Kalman mesafesi çok uzak olanları engelle)
        # DeepSORT'ta bu adım ReID cost'unu sonsuz yaparak engellenir
        for i, t_idx in enumerate(track_indices):
            track = self.tracks[t_idx]
            gating_dist = self.kf.gating_distance(
                track.mean, track.covariance, 
                np.array([d[0] for d in detections]).T
            )
            # Eğer gating threshold'u geçerse cost'u max yap (eşleşmez)
            cost_matrix[i, gating_dist > 9.48] = 1e5 # Chi-square 0.05

        matches, unmatched_tracks, unmatched_dets = matching.linear_assignment(cost_matrix, self.max_dist)
        
        # İndeksleri geri dönüştür (subset index -> global index)
        global_matches = []
        for m in matches:
            global_matches.append((track_indices[m[0]], m[1]))
        
        global_unmatched_tracks = [track_indices[i] for i in unmatched_tracks]
        
        return global_matches, global_unmatched_tracks, unmatched_dets

    def _match_iou(self, detections, track_indices, detection_indices):
        """ Kalanlar için IoU eşleştirmesi """
        if len(track_indices) == 0 or len(detection_indices) == 0:
            return [], track_indices, detection_indices

        # Trackleri bbox formatına çevir
        track_bboxes = np.array([self.tracks[i].to_tlbr() for i in track_indices])
        det_bboxes = np.array([detections[i][3] for i in detection_indices])

        iou_matrix = 1 - matching.iou_batch(track_bboxes, det_bboxes)
        
        matches, unmatched_tracks_idx, unmatched_dets_idx = \
            matching.linear_assignment(iou_matrix, self.max_iou_distance)

        global_matches = []
        for m in matches:
            global_matches.append((track_indices[m[0]], detection_indices[m[1]]))
            
        global_unmatched_tracks = [track_indices[i] for i in unmatched_tracks_idx]
        global_unmatched_dets = [detection_indices[i] for i in unmatched_dets_idx]
        
        return global_matches, global_unmatched_tracks, global_unmatched_dets

    def _initiate_track(self, detection):
        mean, covariance = self.kf.initiate(detection[0])
        self.tracks.append(Track(
            mean, covariance, self._next_id, 
            self.n_init, self.max_age, 
            feature=detection[4]
        ))
        self._next_id += 1