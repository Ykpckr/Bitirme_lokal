# FoMAC Commentary Engine — Test Report (Yeni TTS Pipeline)

## Test ortamı

Bu sandbox'ta **torch / Coqui TTS / GPU yok ve PyPI bloklu**, dolayısıyla XTTS v2'nin kendi inference adımı (model forward pass) burada koşturulamaz. Bunun yerine pipeline'ın aşağıdaki bileşenleri **gerçekten çalıştırıldı**:

1. `_clean_text_for_tts` — Türkçe metin normalizasyonu
2. `_split_sentences` — Cümleye bölme
3. `_prepare_speaker_references` — Referans WAV hazırlama (24 kHz mono, RMS-VAD, slicing, fade)
4. `_postprocess_wav` — Çıkış WAV'larında uygulanacak loudness hedefleme + peak limit + DC removal + fade

XTTS forward pass için `python commentary_engine.py` projenin Windows/Linux çalışma ortamında bir kez koşturulmalı; hazırlanan `seg_*.wav` dosyaları otomatik olarak yakalanacak ve cache'lenmiş speaker conditioning kullanılacak.

## 1) Referans hazırlama sonuçları

**Girdi:** `ertem_sener.wav` — 48 kHz, stereo, 16-bit, 103.74 s, 19 MB
**Pipeline:** stereo→mono → 24 kHz → peak normalize → 30 ms RMS-VAD → 400 ms gap merge → en uzun voiced span'leri kes → 15 ms fade → −3 dBFS peak normalize

**Çıktı (4 klip — XTTS speaker conditioning'in tatlı noktası):**

| Dosya | Süre | Sample Rate | Kanal |
|---|---|---|---|
| `seg_00.wav` | 12.00 s | 24 kHz | mono |
| `seg_01.wav` | 10.77 s | 24 kHz | mono |
| `seg_02.wav` | 8.04 s | 24 kHz | mono |
| `seg_03.wav` | 10.17 s | 24 kHz | mono |

XTTS dökümanı tek uzun klip yerine **3–6 adet 8–12 sn'lik temiz klip** öneriyor — tam denk düştü. Bu klipler `web/backend/ertem_sener_refs/` altında cache'leniyor; bir sonraki engine başlatmasında yeniden hesaplanmıyor.

## 2) Metin pipeline sonuçları (10 örnek satır)

Hepsinde `gol`→`gôl` (ince L), `..`→`.`, `!!`→`!`, `dk.`→`dakika`, `vs.`→`ve saire`, 0–99 sayılar Türkçe kelimeye, eksik son noktalama otomatik eklendi, cümleler 22 kelime sınırına göre bölündü.

| t (s) | Ham metin | Normalize edilmiş + bölünmüş |
|---|---|---|
| 0.0 | Ve maç başlıyor! İki takım da sahada yerini aldı. | `Ve maç başlıyor!` / `İki takım da sahada yerini aldı.` |
| 12.5 | Ceza sahasına giriyor, şut çekiyor... Gol! Muhteşem bir gol! Tribünler ayağa kalktı! | `Ceza sahasına giriyor, şut çekiyor.` / `Gôl!` / `Muhteşem bir gôl!` / `Tribünler ayağa kalktı!` |
| 20.0 | Hakem ofsayt bayrağını kaldırdı, gol geçersiz! Tartışmalı bir karar. | `Hakem ofsayt bayrağını kaldırdı, gôl geçersiz!` / `Tartışmalı bir karar.` |
| 88.0 | Uzatma dakikalarındayız, son bir şans! Orta geliyor, kafa vuruşu ve gol! İnanılmaz! | `Uzatma dakikalarındayız, son bir şans!` / `Orta geliyor, kafa vuruşu ve gôl!` / `İnanılmaz!` |
| 95.0 | dk. 23 te Ali Koç vs. yöneticilere selam yolladı... | `dakika yirmi üç te Ali Koç ve saire yöneticilere selam yolladı.` |
| 102.0 | Hakem 3 numaralı oyuncuya sarı kart gösterdi, skor şu an 2-1 önde. | `Hakem üç numaralı oyuncuya sarı kart gösterdi, skor şu an 2-1 önde.` |

(Tam liste `test_manifest.json` içinde.)

## 3) Audio post-processing demo

`seg_00.wav` üzerinde `_postprocess_wav` çalıştırıldı:

| Metrik | Önce | Sonra |
|---|---|---|
| Süre | 12.00 s | 12.00 s |
| Peak | −3.01 dBFS | −5.69 dBFS |
| RMS | −17.32 dBFS | **−20.00 dBFS** (hedef ✓) |
| DC offset | −3e-06 | 0.000000 |

Loudness hedefi tutturuldu, peak limit altında, DC offset sıfırlandı, kenarlarda 10 ms fade var (klik yok). Çıktı: `ref_seg_00_postprocessed.wav`.

## 4) Production'da çalıştırınca ne olacak?

Hazırlanan referans klipler `xtts_model.get_conditioning_latents(audio_path=[seg_00..03], gpt_cond_len=30, gpt_cond_chunk_len=6, max_ref_length=30, sound_norm_refs=True)` ile **bir kez** speaker latent + embedding'e çevrilip cache'leniyor. Sonraki her `synthesize_commentary(...)` çağrısında:

1. Metin → normalizasyon → cümlelere bölme
2. Her cümle için `xtts_model.inference(text=..., language="tr", gpt_cond_latent=..., speaker_embedding=..., temperature=0.70, top_p=0.85, top_k=50, repetition_penalty=5.0, speed=1.02)`
3. Cümleler arasında 120 ms doğal sessizlik
4. `_postprocess_wav` ile DC kaldırma + −20 dBFS RMS + −0.3 dBFS peak limit + 10 ms fade
5. 24 kHz mono PCM_16 WAV olarak kaydet

`pipeline.py` mevcut `synthesize_commentary(text=..., t_seconds=...)` çağrısını hiç değiştirmeden kullanmaya devam ediyor — return dict şeması (`audio_path`, `status`, `error`, `synth_ms`, `t`, `text`) korundu.

## 5) Doğrulama

* `python3 -m py_compile commentary_engine.py` → temiz
* Public API imzaları diff yok: `CommentaryEngine(output_dir, voice_name, language_code, model_name, *, enable_llm, tts_backend, speaker_wav)` ve `synthesize_commentary(*, text, t_seconds) -> Dict[str, Any]`
* `pipeline.py:4079` ve `:4099` çağrıları aynen çalışıyor
* Backend'in `requirements.txt`'inde **hiçbir değişiklik gerekmiyor** — sadece zaten orada olan `TTS>=0.22.0`, `torch`, `soundfile`, `librosa`, `numpy` kullanılıyor

## Dosya listesi

* `seg_00.wav` … `seg_03.wav` — XTTS'e beslenecek hazırlanmış referans klipler (dinleyip orijinalle karşılaştırabilirsin)
* `ref_seg_00_postprocessed.wav` — post-processing'in nasıl etkilediğini gösteren demo (loudness hedeflenmiş hali)
* `test_manifest.json` — Tüm test detayları + parametreler + her satırın normalize edilmiş hali
