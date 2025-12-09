import yaml
import sys
from pathlib import Path

class Config:
    def __init__(self, config_path):
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config dosyası bulunamadı: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self._cfg = yaml.safe_load(f)

    @property
    def video(self):
        return self._cfg.get('video', {})

    @property
    def detection(self):
        return self._cfg.get('detection', {})

    @property
    def reid(self):
        return self._cfg.get('reid', {})

    @property
    def tracker(self):
        return self._cfg.get('tracker', {})
    
    @property
    def spotter(self):
        return self._cfg.get('spotter', {})

    def dump(self):
        """Ayarları konsola basar (Debug için)"""
        print(yaml.dump(self._cfg, default_flow_style=False))

def load_config(path='configs/config.yaml'):
    return Config(path)

if __name__ == "__main__":
    # Test bloğu
    try:
        cfg = load_config()
        print("Config başarıyla yüklendi.")
        cfg.dump()
    except Exception as e:
        print(f"Hata: {e}")