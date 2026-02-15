"""Video üstüne action spotting tahminlerini yazarak çıktı video üretir.

KULLANIM:
- Bu dosyanın en üstündeki VIDEO_PATH / FEATURE_PATH / CHECKPOINT_PATH değişkenlerini güncelle.
- Ardından: `python video_inference.py`

Notlar:
- FEATURE_PATH: SoccerNet feature .npy dosyası (örn: 1_ResNET_TF2_PCA512.npy)
- VIDEO_PATH: Aynı half'a karşılık gelen ham video (mp4 vb.)
- Feature FPS'i `config.py` içindeki cfg.FPS ile belirlenir. Video FPS'i video dosyasından okunur.
"""

from __future__ import annotations

# ============================================================================
# HARDCODED INPUTS (DOSYANIN EN ÜSTÜNDE)
# ============================================================================
VIDEO_PATH = r"H:\SoccerNetVideos\england_epl\2014-2015\2015-02-21 - 18-00 Chelsea 1 - 1 Burnley\1_720p.mkv"
FEATURE_PATH = r"H:\soccerNet\england_epl\2014-2015\2015-02-21 - 18-00 Chelsea 1 - 1 Burnley\1_ResNET_TF2_PCA512.npy"
CHECKPOINT_PATH = r".\checkpoints\v3_cnn_0211_2000_best_map.pth"
OUTPUT_VIDEO_PATH = r".\outputs\video_predictions_clips.mp4"

# Tahmin ayarları
THRESHOLD = 0.50
NMS_WINDOW_SEC = 10
# Sadece tahmin yapılan kısımlar clip olarak çıkarılır.
# Her event için clip aralığı: [t-CLIP_PRE_SEC, t+CLIP_POST_SEC]
CLIP_PRE_SEC = 2.0
CLIP_POST_SEC = 2.0
# Yakın event clip'lerini birleştirmek için boşluk toleransı
MERGE_GAP_SEC = 0.25

# ============================================================================

import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

import config as cfg


def _extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    if isinstance(checkpoint, dict) and all(isinstance(k, str) for k in checkpoint.keys()):
        # raw state_dict kaydedilmiş olabilir
        return checkpoint  # type: ignore[return-value]
    raise RuntimeError("Checkpoint formatı beklenenden farklı (state_dict bulunamadı).")


def _infer_cnn_hparams_from_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, int]:
    # NetVLAD cluster sayısı
    if "netvlad_past.cluster_weights" not in state_dict:
        raise RuntimeError("Checkpoint CNNActionSpotter/NetVLAD anahtarlarını içermiyor.")
    k_clusters = int(state_dict["netvlad_past.cluster_weights"].shape[0])

    # Projection dim ve input feature dim
    if "input_proj.0.weight" not in state_dict:
        raise RuntimeError("Checkpoint input_proj ağırlıklarını içermiyor.")
    proj_dim = int(state_dict["input_proj.0.weight"].shape[0])
    feature_dim = int(state_dict["input_proj.0.weight"].shape[1])

    return {"k_clusters": k_clusters, "proj_dim": proj_dim, "feature_dim": feature_dim}


def _require_cv2():
    try:
        import cv2  # type: ignore

        return cv2
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "OpenCV (cv2) bulunamadı. Kurmak için: `pip install opencv-python`\n"
            "Conda kullanıyorsan: `conda install -c conda-forge opencv`"
        ) from exc


def load_model(checkpoint_path: str) -> torch.nn.Module:
    checkpoint_path = str(checkpoint_path)
    print(f"📥 Model yükleniyor: {checkpoint_path}")

    checkpoint: Any = torch.load(checkpoint_path, map_location=cfg.DEVICE)
    state_dict = _extract_state_dict(checkpoint)

    # Not: Model mimarisi config'ten okunuyor; ama config değişmişse checkpoint yüklenmez.
    # Bu yüzden CNN için gerekli hparam'ları checkpoint'ten okuyup cfg'yi override ediyoruz.
    if cfg.MODEL_TYPE == "cnn":
        inferred = _infer_cnn_hparams_from_state_dict(state_dict)

        if cfg.NETVLAD_CLUSTERS != inferred["k_clusters"]:
            print(
                f"⚠️  NETVLAD_CLUSTERS config={cfg.NETVLAD_CLUSTERS} ama checkpoint={inferred['k_clusters']}. "
                "Checkpoint değerine göre override ediyorum."
            )
        if cfg.PROJECTION_DIM != inferred["proj_dim"]:
            print(
                f"⚠️  PROJECTION_DIM config={cfg.PROJECTION_DIM} ama checkpoint={inferred['proj_dim']}. "
                "Checkpoint değerine göre override ediyorum."
            )
        if cfg.FEATURE_DIM != inferred["feature_dim"]:
            print(
                f"⚠️  FEATURE_DIM config={cfg.FEATURE_DIM} ama checkpoint={inferred['feature_dim']}. "
                "Checkpoint değerine göre override ediyorum."
            )

        cfg.NETVLAD_CLUSTERS = inferred["k_clusters"]
        cfg.PROJECTION_DIM = inferred["proj_dim"]
        cfg.FEATURE_DIM = inferred["feature_dim"]

        from model import CNNActionSpotter

        model = CNNActionSpotter().to(cfg.DEVICE)
        model.load_state_dict(state_dict)
    elif cfg.MODEL_TYPE == "transformer":
        from model import ActionTransformer

        model = ActionTransformer().to(cfg.DEVICE)
        model.load_state_dict(state_dict)
    else:
        raise ValueError(f"Unknown MODEL_TYPE: {cfg.MODEL_TYPE}")

    model.eval()
    return model


def predict_events(model: torch.nn.Module, feature_path: str, threshold: float) -> list[dict[str, Any]]:
    print(f"📂 Özellikler okunuyor: {feature_path}")
    features = np.load(feature_path, mmap_mode="r")
    total_frames = int(features.shape[0])

    predictions: list[dict[str, Any]] = []

    stride = cfg.FPS
    window_half = cfg.WINDOW_SIZE_FRAMES // 2

    print("🔎 Inference (sliding window) ...")
    for frame_idx in range(window_half, total_frames - window_half, stride):
        start = frame_idx - window_half
        end = frame_idx + window_half

        chunk = features[start:end].copy()
        tensor = torch.from_numpy(chunk).float().unsqueeze(0).to(cfg.DEVICE)

        with torch.no_grad():
            cls_logits, reg_offset = model(tensor)

        probs = torch.sigmoid(cls_logits).squeeze(0)  # (18,)
        score, cls_id = torch.max(probs[:-1], dim=0)  # background hariç

        if float(score.item()) > threshold:
            predicted_shift_frames = float(reg_offset.item()) * cfg.WINDOW_SIZE_FRAMES
            event_frame = float(frame_idx) + predicted_shift_frames
            event_time = event_frame / float(cfg.FPS)

            predictions.append(
                {
                    "time": float(event_time),
                    "label": cfg.ID_TO_EVENT[int(cls_id.item())],
                    "score": float(score.item()),
                }
            )

    return predictions


def nms(predictions: list[dict[str, Any]], window_sec: float) -> list[dict[str, Any]]:
    if not predictions:
        return []

    preds = sorted(predictions, key=lambda x: x["score"], reverse=True)
    keep: list[dict[str, Any]] = []

    while preds:
        current = preds.pop(0)
        keep.append(current)

        cur_label = current["label"]
        cur_time = float(current["time"])

        preds = [
            p
            for p in preds
            if not (p["label"] == cur_label and abs(float(p["time"]) - cur_time) < window_sec)
        ]

    keep.sort(key=lambda x: x["time"])
    return keep


def export_prediction_clips(
    video_path: str,
    output_path: str,
    events: list[dict[str, Any]],
    clip_pre_sec: float,
    clip_post_sec: float,
    merge_gap_sec: float,
) -> None:
    cv2 = _require_cv2()

    video_path = str(video_path)
    output_path = str(output_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Video açılamadı: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = 25.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"🎬 Video FPS: {fps:.2f} | Frame: {total} | Size: {width}x{height}")
    print(f"💾 Output: {output_path}")

    # 1) Event -> clip interval (frame) listesi
    video_duration_sec = (total / fps) if total > 0 else None
    raw_intervals: list[dict[str, Any]] = []

    for e in events:
        t = float(e["time"])
        start_sec = max(0.0, t - float(clip_pre_sec))
        end_sec = t + float(clip_post_sec)
        if video_duration_sec is not None:
            end_sec = min(end_sec, float(video_duration_sec))

        start_f = int(round(start_sec * fps))
        end_f = int(round(end_sec * fps))
        start_f = max(0, min(start_f, max(0, total - 1)))
        end_f = max(start_f + 1, min(end_f, total))

        raw_intervals.append({"start_f": start_f, "end_f": end_f, "events": [e]})

    raw_intervals.sort(key=lambda x: x["start_f"])

    # 2) Overlap / yakın aralıkları birleştir
    gap_frames = int(round(float(merge_gap_sec) * fps))
    merged: list[dict[str, Any]] = []

    for itv in raw_intervals:
        if not merged:
            merged.append(itv)
            continue

        last = merged[-1]
        if itv["start_f"] <= int(last["end_f"]) + gap_frames:
            last["end_f"] = max(int(last["end_f"]), int(itv["end_f"]))
            last["events"].extend(itv["events"])
        else:
            merged.append(itv)

    if not merged:
        print("⚠️  Hiç event yok; output yazılmadı.")
        cap.release()
        writer.release()
        return

    out_written_frames = 0
    print(f"✂️  Clip sayısı (merge sonrası): {len(merged)}")

    # 3) Her clip'i sırayla yaz
    for idx, itv in enumerate(merged, start=1):
        start_f = int(itv["start_f"])
        end_f = int(itv["end_f"])

        # interval içindeki eventlerden en yüksek skorlu olanı etiket olarak göster
        top_event = max((ev for ev in itv["events"]), key=lambda ev: float(ev.get("score", 0.0)))
        t_ev = float(top_event["time"])
        mm, ss = divmod(int(t_ev), 60)
        overlay_text = f"{top_event['label']} ({float(top_event['score']):.2f})  @ {mm:02d}:{ss:02d}"

        print(f"  [{idx}/{len(merged)}] frames {start_f}-{end_f}  | {overlay_text}")

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
        cur = start_f

        while cur < end_f:
            ok, frame = cap.read()
            if not ok:
                break

            # Overlay: clip info + en iyi event
            cv2.putText(
                frame,
                overlay_text,
                (30, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            writer.write(frame)
            out_written_frames += 1
            cur += 1

    cap.release()
    writer.release()

    out_sec = out_written_frames / fps
    print(f"✅ Clip-video yazıldı: ~{out_sec:.1f}s (frames={out_written_frames})")


def main() -> None:
    # Basit path kontrolleri
    if not Path(CHECKPOINT_PATH).exists():
        raise FileNotFoundError(
            f"Checkpoint bulunamadı: {CHECKPOINT_PATH}\n"
            "Mevcut checkpoint'leri görmek için ./checkpoints klasörünü kontrol et."
        )
    if not Path(FEATURE_PATH).exists():
        raise FileNotFoundError(f"Feature .npy bulunamadı: {FEATURE_PATH}")
    if not Path(VIDEO_PATH).exists():
        raise FileNotFoundError(f"Video bulunamadı: {VIDEO_PATH}")

    model = load_model(CHECKPOINT_PATH)
    preds = predict_events(model, FEATURE_PATH, threshold=THRESHOLD)
    final_preds = nms(preds, window_sec=float(NMS_WINDOW_SEC))

    print(f"\n🎯 Event sayısı (NMS sonrası): {len(final_preds)}")
    for p in final_preds[:10]:
        print(f"- {p['time']:.2f}s | {p['label']} | {p['score']:.2f}")
    if len(final_preds) > 10:
        print(f"... (+{len(final_preds) - 10} tane daha)")

    export_prediction_clips(
        video_path=VIDEO_PATH,
        output_path=OUTPUT_VIDEO_PATH,
        events=final_preds,
        clip_pre_sec=float(CLIP_PRE_SEC),
        clip_post_sec=float(CLIP_POST_SEC),
        merge_gap_sec=float(MERGE_GAP_SEC),
    )

    print("✅ Tamamlandı.")


if __name__ == "__main__":
    main()
