import numpy as np

class Track:
    """
    Tek bir iz (track) durumunu tutar.
    """
    def __init__(self, mean, covariance, track_id, n_init, max_age, feature=None):
        self.mean = mean
        self.covariance = covariance
        self.track_id = track_id
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        
        self.state = 1 # 1: Tentative (Deneme), 2: Confirmed (Onaylı), 3: Deleted
        self.features = []
        if feature is not None:
            self.features.append(feature)

        self._n_init = n_init
        self._max_age = max_age

    def to_tlwh(self):
        """ (top, left, width, height) formatına çevirir """
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    def to_tlbr(self):
        """ (min_x, min_y, max_x, max_y) formatına çevirir """
        ret = self.to_tlwh()
        ret[2:] += ret[:2]
        return ret

    def predict(self, kf):
        """ Kalman filtresi ile bir sonraki konumu tahmin et """
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1

    def update(self, kf, detection, feature=None):
        """ Yeni bir detection ile izi güncelle """
        self.mean, self.covariance = kf.update(self.mean, self.covariance, detection)
        
        if feature is not None:
            self.features.append(feature)
            # Hafıza optimizasyonu: Son 100 özelliği tut
            if len(self.features) > 100:
                self.features.pop(0)

        self.hits += 1
        self.time_since_update = 0
        
        if self.state == 1 and self.hits >= self._n_init:
            self.state = 2 # Confirmed

    def mark_missed(self):
        """ Eğer detection ile eşleşmezse çağrılır """
        if self.state == 1:
            self.state = 3 # Tentative iken kaçırırsa hemen sil
        elif self.time_since_update > self._max_age:
            self.state = 3 # Max age dolduysa sil

    def is_confirmed(self):
        return self.state == 2

    def is_deleted(self):
        return self.state == 3