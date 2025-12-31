import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from sklearn.cluster import KMeans
import numpy as np

class DeepTeamClassifierPro:
    def __init__(self, device='cuda'):
        self.device = device
        self.n_clusters = 2
        # K-Means++ algoritması vektörleri ayırır
        self.kmeans = KMeans(n_clusters=self.n_clusters, init="k-means++", n_init=10)
        self.trained = False
        
        print("🧠 Derin Zeka (ResNet50) Hazırlanıyor...")
        
        # 1. ResNet50'yi indir (Hazır ağırlıklarla)
        # Bu model ImageNet üzerinde eğitildiği için desenleri çok iyi tanır.
        weights = models.ResNet50_Weights.DEFAULT
        self.model = models.resnet50(weights=weights)
        
        # 2. Son katmanı (fc layer) söküyoruz.
        # Amacımız sınıflandırma değil, "Feature Extraction" (Özellik Çıkarma).
        # Son katman yerine Identity koyarak çıktıyı ham vektör olarak alıyoruz.
        self.model.fc = nn.Identity()
        
        self.model.to(device)
        self.model.eval() # Sadece analiz modu (Eğitim yok)
        
        # ResNet'in gözüne uygun hale getirme standartları
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def get_features(self, image_crop):
        """
        Oyuncunun resmini alır, 2048 uzunluğunda bir sayı dizisine (vektör) çevirir.
        """
        if image_crop.size == 0: return None
        
        # ResNet'e sok (Batch boyutu ekle: unsqeeze)
        img_tensor = self.transform(image_crop).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Modelden geçir
            features = self.model(img_tensor)
            # CPU'ya al ve düzleştir (Vektör haline getir)
            features = features.cpu().numpy().flatten()
            
        return features

    def fit(self, crops):
        """
        Sahadaki oyunculardan örnekler alır ve takımları öğrenir.
        """
        feature_list = []
        print(f"🧠 {len(crops)} oyuncu derinlemesine analiz ediliyor...")
        
        for crop in crops:
            feat = self.get_features(crop)
            if feat is not None:
                feature_list.append(feat)
        
        if len(feature_list) > 0:
            print("🤖 Yapay Zeka (K-Means) Vektörleri Grupluyor...")
            
            self.kmeans.fit(feature_list)
            self.trained = True
            print("✅ Takımlar ResNet ile Ayrıştırıldı!")
            return True
        return False

    def predict(self, crop):
        """
        Bu oyuncunun vektörü hangi takıma benziyor?
        """
        if not self.trained: return -1
        
        feat = self.get_features(crop)
        if feat is None: return -1
        
        # En yakın kümeyi bul
        label = self.kmeans.predict([feat])[0]
        return label