Backend

Proje
Bu klasör uygulamanın Python tabanlı backend bileşenini içerir. Ana giriş noktası `main.py` dosyasıdır; çalıştırma için ayrıca `run.sh` ve bağımlılıklar `requirements.txt` içinde tanımlanmıştır.

Gereksinimler
- Python 3.8 veya daha yeni
- `pip` ve sanal ortam aracı (venv)

Kurulum
1. Proje dizinine girin:

   cd backend

2. Sanal ortam oluşturup etkinleştirin:

   python -m venv .venv
   source .venv/bin/activate

3. Bağımlılıkları yükleyin:

   pip install -r requirements.txt

Çalıştırma
- Hazır betik ile (izin verildiyse):

   chmod +x run.sh
   ./run.sh start

   Durum/log:
   ./run.sh status
   ./run.sh logs
   ./run.sh follow
   ./run.sh stop

- Config dosyası ile (önerilen):

   ```bash
   # örnek config: run.env.example
   ./run.sh start --config ./run.env.example
   ```

   Not: config dosyası `KEY=VALUE` formatında bir env dosyasıdır.
   En sık kullanılanlar: `FOMAC_CONDA_ENV`, `FOMAC_YOLO_MODEL_PATH`, `FOMAC_VIDEO_DIR`.

- Windows (PowerShell) ile:

   ```powershell
   powershell -ExecutionPolicy Bypass -File .\run.ps1 start  -Config .\run.env.example
   powershell -ExecutionPolicy Bypass -File .\run.ps1 status -Config .\run.env.example
   powershell -ExecutionPolicy Bypass -File .\run.ps1 logs   -Config .\run.env.example
   powershell -ExecutionPolicy Bypass -File .\run.ps1 follow -Config .\run.env.example
   powershell -ExecutionPolicy Bypass -File .\run.ps1 stop   -Config .\run.env.example
   ```

- Alternatif olarak doğrudan Python ile:

  python main.py

Dosyalar
- `main.py`: Uygulama giriş noktası.
- `requirements.txt`: Python bağımlılıkları.
- `run.sh`: Ortamı hazırlayıp sunucuyu başlatmaya yardımcı betik.
- `uploads/`: Uygulamanın yüklenen verileri sakladığı dizin (örnek JSON dosyaları içerir).

Geliştirme
- Değişiklik yapmadan önce sanal ortamı etkinleştirin.
- Geliştirme sırasında çıktı ve hatalar `main.py` tarafından konsola yazdırılır; loglama eklemek isterseniz mevcut kodu düzenleyin.

Sorun Giderme
- Bağımlılık hatası alıyorsanız sanal ortamın etkin olduğundan ve `pip install -r requirements.txt` komutunun başarılı tamamlandığından emin olun.
- Port çakışması varsa çalışan başka bir süreç yoksa farklı bir port belirleyin veya mevcut süreci sonlandırın.

Commentary TTS (XTTS v2)
- Varsayılan backend: `xttsv2` (Coqui XTTS v2).
- XTTS v2, bir referans ses dosyası ister. Windows'ta env ile verin:

   - `COMMENTARY_SPEAKER_WAV=C:\\path\\to\\speaker.wav`
   - (opsiyonel) `COMMENTARY_TTS_BACKEND=xttsv2` (veya `sapi` / `pyttsx3`)

- XTTS kurulumu için: `pip install -r requirements.txt` (gerekirse ayrıca uygun `torch` wheel'i kurmanız gerekebilir).
- Eğer `ModuleNotFoundError: No module named 'pkg_resources'` görürseniz, `setuptools==70.3.0` pin'i gereklidir (requirements içinde var).

Katkıda Bulunma
- Küçük düzeltmeler ve hata düzeltmeleri için issue açın veya pull request gönderin.

İletişim
- Proje ile ilgili sorular için depo sahibiyle iletişime geçin.
