from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import time
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

try:
    import httpx  # type: ignore
except Exception:  # pragma: no cover
    httpx = None


@dataclass
class FullPipelineConfig:
    # segment
    start_seconds: float = 0.0
    duration_seconds: Optional[float] = None

    # calibration (must run before tracking)
    run_calibration: bool = True
    calibration_detector_weights: Optional[str] = None
    calibration_kp_weights: Optional[str] = None
    calibration_line_weights: Optional[str] = None
    calibration_conf_thres: float = 0.30
    calibration_write_frames_jsonl: bool = False
    calibration_frames_stride: int = 1

    # tracking
    run_tracking: bool = True
    tracking_device: Optional[str] = None
    tracking_config_path: Optional[str] = None
    detector_weights: Optional[str] = None
    reid_weights: Optional[str] = None

    # action spotting
    run_action_spotting: bool = True
    features_path: Optional[str] = None
    checkpoint_path: Optional[str] = None
    action_threshold: float = 0.50
    action_nms_window_sec: float = 10.0

    # overlay
    overlay_event_window_sec: float = 3.0

    # jersey number recognition (Qwen-VL)
    run_jersey_number_recognition: bool = True
    qwen_vl_url: str = "http://localhost:8080/"
    qwen_vl_model: str = "qwen3vl8b"
    jersey_prompt: str = (
        "You are reading a football jersey.\n\n"
        "Visually read the number printed on the player's shirt.\n\n"
        "Rules:\n\n"
        "* Output digits only.\n"
        "* If no number is visible or readable, output: -1\n"
        "* Do not guess.\n"
        "* Do not describe the image.\n"
        "* Do not output words.\n\n"
        "Return only the final answer.\n"
    )
    # 0 or negative means: no cap (process all player tracks)
    jersey_max_tracks: int = 0
    jersey_max_samples_per_track: int = 5
    jersey_min_det_conf: float = 0.55
    jersey_min_box_area: int = 40 * 40
    jersey_min_frame_gap: int = 20
    # When doing jersey in tracking: query up to this many player tracks per frame.
    jersey_frame_topk: int = 5
    # Optional: save jersey crops for debugging.
    jersey_crops_dir: Optional[str] = None
    # Optional: visibility filter to reduce wasted Qwen calls.
    jersey_vis_filter: bool = False
    jersey_vis_min_score: float = 0.12
    # Prefer doing jersey crops while tracking processes frames (uses in-memory frames; avoids random seeks)
    jersey_in_tracking: bool = True
    jersey_merge_same_number: bool = True
    jersey_merge_min_confidence: float = 0.60
    jersey_merge_max_overlap_frames: int = 5

    # commentary (Qwen text -> TTS -> mix into video)
    run_commentary: bool = True
    commentary_max_events: int = 30
    commentary_possession_max_age_sec: float = 8.0
    # TTS + audio mixing onto overlay video
    commentary_enable_tts: bool = True
    # Default to XTTS v2 (Coqui). If unavailable or speaker wav is missing, the audio manifest
    # will include errors; you can switch to 'sapi' for a lightweight Windows fallback.
    commentary_tts_backend: str = "xttsv2"  # or: sapi, pyttsx3, xtts/xttsv2
    commentary_speaker_wav: Optional[str] = None

    # heuristics
    possession_dist_norm: float = 0.08
    possession_stable_frames: int = 6
    possession_stride_frames: int = 5
    ball_cls_id: int = 1
    player_cls_id: int = 0


def _repo_root() -> Path:
    # web/backend/pipeline.py -> parents: backend(0), web(1), repo(2)
    return Path(__file__).resolve().parents[2]


def _default_action_checkpoint() -> Optional[str]:
    p = _repo_root() / "model-training" / "action_spotting" / "spotting_v2" / "checkpoints" / "v3_cnn_0211_2000_best_map.pth"
    return str(p) if p.exists() else None


def _default_tracking_config() -> Optional[str]:
    p = _repo_root() / "model-training" / "tracking-reid-osnet" / "config.yaml"
    return str(p) if p.exists() else None


def _default_calibration_detector_weights() -> Optional[str]:
    p = _repo_root() / "model-training" / "calibration" / "best.pt"
    return str(p) if p.exists() else None


def _default_calibration_kp_weights() -> Optional[str]:
    p = _repo_root() / "model-training" / "calibration" / "SV_kp.pth"
    return str(p) if p.exists() else None


def _default_calibration_line_weights() -> Optional[str]:
    p = _repo_root() / "model-training" / "calibration" / "SV_lines.pth"
    return str(p) if p.exists() else None


def run_calibration_pipeline(
    *,
    video_path: str,
    out_map: str,
    out_events: str,
    out_frames: Optional[str] = None,
    cfg: Optional[FullPipelineConfig] = None,
    progress_cb: Optional[Callable[[str, int, int, str], None]] = None,
) -> Dict[str, Any]:
    """Run calibration as a subprocess and return artifact paths.

    Best-effort: callers should catch exceptions and proceed.
    """

    repo = _repo_root()
    script = repo / "model-training" / "calibration" / "run_pipeline_calibration.py"
    if not script.exists():
        raise FileNotFoundError(str(script))

    det_w = None
    kp_w = None
    line_w = None
    conf = 0.30
    if cfg is not None:
        try:
            det_w = getattr(cfg, "calibration_detector_weights", None)
            kp_w = getattr(cfg, "calibration_kp_weights", None)
            line_w = getattr(cfg, "calibration_line_weights", None)
            conf = float(getattr(cfg, "calibration_conf_thres", 0.30) or 0.30)
        except Exception:
            det_w, kp_w, line_w, conf = None, None, None, 0.30

    det_w = str(det_w).strip() if det_w else (_default_calibration_detector_weights() or "")
    kp_w = str(kp_w).strip() if kp_w else (_default_calibration_kp_weights() or "")
    line_w = str(line_w).strip() if line_w else (_default_calibration_line_weights() or "")

    if not det_w or not os.path.isfile(det_w):
        raise FileNotFoundError(f"calibration detector weights not found: {det_w}")
    if not kp_w or not os.path.isfile(kp_w):
        raise FileNotFoundError(f"calibration kp weights not found: {kp_w}")
    if not line_w or not os.path.isfile(line_w):
        raise FileNotFoundError(f"calibration line weights not found: {line_w}")

    cmd: List[str] = [
        sys.executable,
        "-u",
        str(script),
        "--source",
        str(Path(video_path).resolve()),
        "--out_map",
        str(Path(out_map).resolve()),
        "--out_events",
        str(Path(out_events).resolve()),
        "--detector",
        str(Path(det_w).resolve()),
        "--kp_weights",
        str(Path(kp_w).resolve()),
        "--line_weights",
        str(Path(line_w).resolve()),
        "--conf",
        str(float(conf)),
    ]

    frames_stride = 1
    if cfg is not None:
        try:
            frames_stride = int(getattr(cfg, "calibration_frames_stride", 1) or 1)
        except Exception:
            frames_stride = 1
    if frames_stride < 1:
        frames_stride = 1

    if out_frames and str(out_frames).strip():
        cmd += ["--out_frames", str(Path(str(out_frames)).resolve()), "--frames_stride", str(int(frames_stride))]

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("PYTHONFAULTHANDLER", "1")
    env.setdefault("TORCH_SHOW_CPP_STACKTRACES", "1")

    # Stream stdout to extract progress and final JSON payload.
    p = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        bufsize=1,
        universal_newlines=True,
    )

    last_json: Optional[Dict[str, Any]] = None
    tail: List[str] = []
    try:
        assert p.stdout is not None
        for raw_line in p.stdout:
            line = (raw_line or "").strip()
            if not line:
                continue

            tail.append(line)
            if len(tail) > 200:
                tail = tail[-200:]

            if line.startswith("__PROGRESS__"):
                try:
                    payload = json.loads(line[len("__PROGRESS__") :].strip())
                    stage = str(payload.get("stage") or "calibration")
                    cur = int(payload.get("current") or 0)
                    total = int(payload.get("total") or 0)
                    msg = str(payload.get("message") or "")
                    if progress_cb is not None:
                        try:
                            progress_cb(stage, cur, total, msg)
                        except Exception:
                            pass
                except Exception:
                    continue
                continue

            # The runner prints a JSON object at the end with artifact paths.
            if line.startswith("{") and "map_video_path" in line and "events_json_path" in line:
                try:
                    last_json = json.loads(line)
                except Exception:
                    pass
    finally:
        try:
            if p.stdout is not None:
                p.stdout.close()
        except Exception:
            pass

    rc = p.wait()
    if rc != 0:
        err = "\n".join(tail[-200:])
        raise RuntimeError(f"calibration subprocess failed (code {rc}):\n{err[-8000:]}")

    if isinstance(last_json, dict):
        return last_json

    # Best-effort fallback: return declared output paths.
    return {
        "map_video_path": str(Path(out_map).resolve()),
        "events_json_path": str(Path(out_events).resolve()),
        **({"frames_jsonl_path": str(Path(str(out_frames)).resolve())} if out_frames and str(out_frames).strip() else {}),
    }


def _timecode(t: float) -> str:
    if t < 0:
        t = 0
    m = int(t // 60)
    s = int(t % 60)
    return f"{m:02d}:{s:02d}"


def _event_desc_tr(label: str) -> str:
    mapping = {
        "Kick-off": "Santra yapıldı, maç başladı.",
        "Throw-in": "Taç atışı kullanıldı.",
        "Goal": "Gol oldu!",
        "Corner": "Korner kullanılıyor.",
        "Free-kick": "Serbest vuruş.",
        "Penalty": "Penaltı!",
        "Offside": "Ofsayt bayrağı kalktı.",
        "Foul": "Faul düdüğü çaldı.",
        "Yellow card": "Sarı kart çıktı.",
        "Red card": "Kırmızı kart!",
        "Substitution": "Oyuncu değişikliği var.",
        "Clearance": "Savunmadan uzaklaştırma.",
        "Shot": "Şut!",
        "Save": "Kaleci kurtardı.",
        "Ball out of play": "Top oyun alanını terk etti.",
        "Direct free-kick": "Direkt serbest vuruş.",
        "Indirect free-kick": "Endirekt serbest vuruş.",
    }
    return mapping.get(label, f"Olay: {label}")


def _clamp_int(v: float, lo: int, hi: int) -> int:
    try:
        iv = int(round(float(v)))
    except Exception:
        iv = int(lo)
    return max(int(lo), min(int(hi), iv))


def _is_special_track_id(track_id: int) -> bool:
    """Reserved/special IDs produced by the tracker (referee/goalkeeper).

    These tracks should never be sent to jersey-number inference.
    """
    try:
        return int(track_id) >= 800_000_000
    except Exception:
        return False


def _qwen_text_openai_compatible(
    *,
    base_url: str,
    model: str,
    prompt: str,
    timeout_sec: float = 60.0,
) -> Tuple[Optional[str], Optional[str]]:
    """Best-effort OpenAI-compatible text call (llama.cpp/vLLM style)."""
    if httpx is None:
        return None, None

    base = _normalize_base_url(base_url)
    if not base:
        return None, None

    endpoint = base + "/v1/chat/completions"
    payload = {
        "model": str(model or "qwen3vl8b"),
        "messages": [
            {
                "role": "system",
                "content": "You are a Turkish football commentator.",
            },
            {
                "role": "user",
                "content": str(prompt or ""),
            },
        ],
        "temperature": 0.7,
        "max_tokens": 800,
    }

    try:
        with httpx.Client(timeout=float(timeout_sec)) as client:
            r = client.post(endpoint, json=payload)
            r.raise_for_status()
            data = r.json()
    except Exception as e:
        return None, str(e)

    raw = ""
    try:
        choices = data.get("choices") or []
        msg = (choices[0] or {}).get("message") or {}
        raw = str(msg.get("content") or "").strip()
    except Exception:
        raw = ""

    return raw, None


def _build_commentary_prompt(items: List[Dict[str, Any]]) -> str:
    """Hardcoded prompt for Qwen (requested)."""
    return (
        "Sen bir futbol maç spikerisin. Aşağıda zaman damgası ile olay listesi var.\n"
        "Her olay için en fazla 2 cümlelik, doğal ve heyecanlı Türkçe yorum yaz.\n"
        "Kurallar:\n"
        "- Sadece Türkçe yaz.\n"
        "- Aynı zaman damgasını koru.\n"
        "- Oyuncu bilgisi varsa (#track_id ve forma_no) onu kullan.\n"
        "- Tahmin etme; belirsizse genel konuş.\n"
        "- ÇIKTI FORMAT: Sadece JSON array döndür.\n"
        "JSON şeması: [{\"t\": <float saniye>, \"timecode\": \"MM:SS\", \"text\": \"...\"}]\n\n"
        "Olaylar JSON:\n"
        + json.dumps(items, ensure_ascii=False, indent=2)
    )


def _extract_json_array_best_effort(raw: str) -> Optional[str]:
    """Extract the first top-level JSON array from a model response.

    Qwen (or OpenAI-compatible servers) sometimes wrap JSON in markdown fences
    or add extra prose. We best-effort extract the first `[...]` block.
    """
    s = str(raw or "").strip()
    if not s:
        return None
    # Strip common markdown fences
    s = re.sub(r"^```(?:json)?\s*", "", s.strip(), flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s.strip())
    # Find first JSON array
    start = s.find("[")
    end = s.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return None
    return s[start : end + 1]


def _timecode_mmss(t: float) -> str:
    return _timecode(float(t))


def _assign_actor_track_id_to_action(
    *,
    action_t: float,
    possession_events: List[Dict[str, Any]],
    max_age_sec: float,
) -> Optional[int]:
    """Pick the last known ball owner at/before action_t."""
    best_t = -1e9
    best_id: Optional[int] = None
    for e in possession_events:
        try:
            et = float(e.get("t", 0.0))
        except Exception:
            continue
        if et > float(action_t):
            break
        pid = e.get("player_track_id")
        if pid is None:
            continue
        try:
            pid_i = int(pid)
        except Exception:
            continue
        if _is_special_track_id(pid_i):
            continue
        if et >= best_t:
            best_t = et
            best_id = pid_i

    if best_id is None:
        return None
    if float(action_t) - float(best_t) > float(max_age_sec):
        return None
    return int(best_id)


def _mix_commentary_audio_into_video(
    *,
    base_video_path: str,
    out_path: str,
    clips: List[Tuple[float, str]],
) -> Optional[str]:
    """Mix timestamped audio clips into the video.

    Produces a new MP4 that contains:
    - Video stream copied/re-encoded from base_video_path
    - Audio stream composed ONLY from the commentary clips placed at timestamps
      (i.e., the base video's original audio is not kept)
    """
    ffmpeg_bin = _ffmpeg_exe()
    if not ffmpeg_bin:
        return None
    if not clips:
        return None

    # Keep only existing audio files that look non-empty.
    kept: List[Tuple[float, str]] = []
    for t, p in clips:
        try:
            # Empty WAV headers can be ~44-60 bytes; require at least 1KB.
            if os.path.isfile(p) and os.path.getsize(p) > 1024:
                kept.append((float(t), str(p)))
        except Exception:
            continue
    if not kept:
        return None

    inputs = ["-i", str(Path(base_video_path).resolve())]
    for _, ap in kept:
        inputs += ["-i", str(Path(ap).resolve())]

    # Best-effort: compute video duration so we can create a silent base track of equal length.
    duration_sec: Optional[float] = None
    try:
        cap = cv2.VideoCapture(str(base_video_path))
        if cap is not None and cap.isOpened():
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            frames = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
            if fps > 1e-3 and frames > 1:
                duration_sec = float(frames / fps)
        try:
            cap.release()
        except Exception:
            pass
    except Exception:
        duration_sec = None

    # Build filter_complex.
    # Audio inputs start at index 1.
    parts: List[str] = []
    amix_inputs: List[str] = []

    # Normalize all audio to a consistent format before mixing.
    # Add a silent baseline so the output audio track spans the full video duration.
    use_silence = duration_sec is not None and duration_sec > 0.25
    if use_silence:
        # anullsrc is infinite; trim it to the video duration.
        dur = max(0.25, float(duration_sec))
        parts.append(
            "anullsrc=channel_layout=stereo:sample_rate=44100,atrim=0:{:.3f},asetpts=N/SR/TB[sil]".format(dur)
        )
        amix_inputs.append("[sil]")

    for i, (t, _ap) in enumerate(kept, start=1):
        delay_ms = max(0, int(round(float(t) * 1000.0)))
        tag = f"a{i}"
        # aformat makes mix more robust across WAV/AAC input variations.
        parts.append(
            f"[{i}:a]aformat=sample_fmts=fltp:sample_rates=44100:channel_layouts=stereo,adelay={delay_ms}|{delay_ms}[{tag}]"
        )
        amix_inputs.append(f"[{tag}]")

    # If we have a silent baseline, set duration=first to match it; else match the longest clip.
    mix_duration = "first" if use_silence else "longest"
    parts.append(
        "".join(amix_inputs)
        + f"amix=inputs={len(amix_inputs)}:duration={mix_duration}:dropout_transition=0[outa]"
    )
    filter_complex = ";".join(parts)

    out_path = str(Path(out_path).resolve())
    os.makedirs(str(Path(out_path).parent), exist_ok=True)

    cmd = [
        ffmpeg_bin,
        "-y",
        *inputs,
        "-filter_complex",
        filter_complex,
        "-map",
        "0:v:0",
        "-map",
        "[outa]",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "160k",
        "-movflags",
        "+faststart",
        out_path,
    ]

    try:
        subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", check=True)
        return out_path
    except Exception:
        # Fallback: re-encode video if stream copy fails.
        try:
            cmd2 = [
                ffmpeg_bin,
                "-y",
                *inputs,
                "-filter_complex",
                filter_complex,
                "-map",
                "0:v:0",
                "-map",
                "[outa]",
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "20",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-b:a",
                "160k",
                "-movflags",
                "+faststart",
                out_path,
            ]
            subprocess.run(cmd2, capture_output=True, text=True, encoding="utf-8", errors="replace", check=True)
            return out_path
        except Exception:
            return None


def _jersey_crop_from_player_bbox(
    frame_bgr: np.ndarray,
    *,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
) -> Optional[np.ndarray]:
    """Return a crop suitable for reading the jersey number.

    The jersey-number-recognition project effectively classifies a player crop.
    Here we keep it similar but bias toward torso area to reduce background.
    """
    h, w = frame_bgr.shape[:2]
    x1 = _clamp_int(x1, 0, w - 1)
    x2 = _clamp_int(x2, 0, w - 1)
    y1 = _clamp_int(y1, 0, h - 1)
    y2 = _clamp_int(y2, 0, h - 1)
    if x2 <= x1 or y2 <= y1:
        return None

    bw = x2 - x1
    bh = y2 - y1
    # torso-ish region (centered)
    cx1 = x1 + int(0.15 * bw)
    cx2 = x2 - int(0.15 * bw)
    cy1 = y1 + int(0.20 * bh)
    cy2 = y1 + int(0.90 * bh)
    cx1 = _clamp_int(cx1, 0, w - 1)
    cx2 = _clamp_int(cx2, 0, w - 1)
    cy1 = _clamp_int(cy1, 0, h - 1)
    cy2 = _clamp_int(cy2, 0, h - 1)
    if cx2 <= cx1 or cy2 <= cy1:
        # fallback to full bbox crop
        crop = frame_bgr[y1:y2, x1:x2]
    else:
        crop = frame_bgr[cy1:cy2, cx1:cx2]
    if crop is None or crop.size == 0:
        return None
    return crop


def _parse_jersey_number_from_text(text: str) -> Optional[str]:
    """Parse model output following strict rules: digits only or -1."""
    if not text:
        return None
    t = str(text).strip()

    # allow single -1
    if t == "-1":
        return "-1"

    # Some servers may wrap output with whitespace/newlines; keep only first token.
    t = t.split()[0] if t.split() else t

    # strict digits only
    if not re.fullmatch(r"\d{1,2}", t):
        return None
    try:
        n = int(t)
    except Exception:
        return None
    if 0 <= n <= 99:
        return str(n)
    return None


def _normalize_base_url(url: str) -> str:
    u = str(url or "").strip()
    if not u:
        return ""
    return u[:-1] if u.endswith("/") else u


def _qwen_vl_openai_compatible(
    *,
    crop_bgr: np.ndarray,
    base_url: str,
    model: str,
    prompt: str,
    timeout_sec: float = 30.0,
) -> Tuple[Optional[str], Optional[str]]:
    """Try OpenAI-compatible chat completions (common for llama.cpp/vllm servers)."""
    try:
        import base64
        import httpx
    except Exception:
        return None, None

    # Encode image as base64 data URL
    try:
        ok, buf = cv2.imencode(".png", crop_bgr)
        if not ok:
            return None, None
        b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
        data_url = f"data:image/png;base64,{b64}"
    except Exception:
        return None, None

    endpoint = _normalize_base_url(base_url) + "/v1/chat/completions"
    payload = {
        "model": str(model or "qwen3vl8b"),
        "temperature": 0,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
    }

    try:
        with httpx.Client(timeout=float(timeout_sec)) as client:
            r = client.post(endpoint, json=payload)
            if r.status_code >= 400:
                return None, None
            obj = r.json()
    except Exception:
        return None, None

    raw_text: Optional[str] = None
    try:
        raw_text = (
            obj.get("choices", [{}])[0]
            .get("message", {})
            .get("content")
        )
        if raw_text is None:
            raw_text = str(obj)
    except Exception:
        raw_text = None

    jersey = _parse_jersey_number_from_text(str(raw_text or ""))
    return jersey, raw_text


def _qwen_vl_ask_jersey_number(
    *,
    crop_bgr: np.ndarray,
    qwen_url: str,
    model: Optional[str] = None,
    prompt: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """Ask Qwen-VL for jersey number from an image crop.

    Returns: (jersey_number, raw_text)
    """
    # prompt must enforce digits-only or -1
    prompt = prompt or (
        "You are reading a football jersey.\n\n"
        "Visually read the number printed on the player's shirt.\n\n"
        "Rules:\n\n"
        "* Output digits only.\n"
        "* If no number is visible or readable, output: -1\n"
        "* Do not guess.\n"
        "* Do not describe the image.\n"
        "* Do not output words.\n\n"
        "Return only the final answer.\n"
    )

    # First try OpenAI-compatible HTTP API (common on :8080)
    jersey_http, raw_http = _qwen_vl_openai_compatible(
        crop_bgr=crop_bgr,
        base_url=str(qwen_url),
        model=str(model or "qwen3vl8b"),
        prompt=str(prompt),
    )
    if jersey_http is not None:
        return jersey_http, raw_http

    # Fallback: Gradio client (common for localhost:7860)
    try:
        from gradio_client import Client, handle_file  # type: ignore
    except Exception:
        return None, None

    # Save a temporary PNG for upload.
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
            tmp_path = tf.name
        ok, buf = cv2.imencode(".png", crop_bgr)
        if not ok:
            try:
                os.remove(tmp_path)
            except Exception:
                pass
            return None, None
        with open(tmp_path, "wb") as f:
            f.write(buf.tobytes())
    except Exception:
        return None, None

    raw_text: Optional[str] = None
    try:
        client = Client(qwen_url)
        img = handle_file(tmp_path)

        # Try a few common Gradio signatures / endpoints.
        attempts: List[Tuple[Optional[str], Tuple[Any, ...]]] = [
            ("/predict", (prompt, img)),
            ("/predict", (img, prompt)),
            ("/chat", (prompt, img)),
            ("/chat", (img, prompt)),
            (None, (prompt, img)),
            (None, (img, prompt)),
        ]

        last_exc: Optional[Exception] = None
        for api_name, args in attempts:
            try:
                if api_name is None:
                    res = client.predict(*args)
                else:
                    res = client.predict(*args, api_name=api_name)

                # Normalize result to a string.
                if isinstance(res, str):
                    raw_text = res
                elif isinstance(res, (list, tuple)) and res:
                    # take first string-like item
                    cand = None
                    for item in res:
                        if isinstance(item, str):
                            cand = item
                            break
                        if isinstance(item, dict) and "text" in item and isinstance(item["text"], str):
                            cand = item["text"]
                            break
                    if cand is None:
                        raw_text = str(res[0])
                    else:
                        raw_text = cand
                elif isinstance(res, dict):
                    raw_text = str(res.get("text") or res.get("output") or res)
                else:
                    raw_text = str(res)

                jersey = _parse_jersey_number_from_text(raw_text)
                return jersey, raw_text
            except Exception as e:
                last_exc = e
                continue

        # If all attempts failed, return no prediction.
        _ = last_exc
        return None, raw_text
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass


def _select_jersey_samples_from_tracks_csv(
    *,
    tracks_csv_path: str,
    cfg: FullPipelineConfig,
) -> Dict[int, List[Dict[str, Any]]]:
    """Pick a few high-quality player bboxes per track to query Qwen-VL."""
    per_track: Dict[int, List[Dict[str, Any]]] = {}

    def can_add(existing: List[Dict[str, Any]], frame_id: int) -> bool:
        gap = max(1, int(cfg.jersey_min_frame_gap))
        for s in existing:
            try:
                if abs(int(s["frame_id"]) - int(frame_id)) < gap:
                    return False
            except Exception:
                continue
        return True

    with open(tracks_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                cls_id = int(float(row.get("cls_id", -1)))
                if cls_id != int(cfg.player_cls_id):
                    continue
                track_id = int(float(row.get("track_id", 0)))
                if track_id <= 0:
                    continue
                if _is_special_track_id(track_id):
                    continue
                frame_id = int(float(row.get("frame_id", 0)))
                conf = float(row.get("conf", 0.0) or 0.0)
                if conf < float(cfg.jersey_min_det_conf):
                    continue

                x1 = int(float(row.get("x1", 0)))
                y1 = int(float(row.get("y1", 0)))
                x2 = int(float(row.get("x2", 0)))
                y2 = int(float(row.get("y2", 0)))
                area = max(0, x2 - x1) * max(0, y2 - y1)
                if area < int(cfg.jersey_min_box_area):
                    continue

                score = float(area) * float(conf)
            except Exception:
                continue

            lst = per_track.setdefault(track_id, [])
            if not can_add(lst, frame_id):
                continue
            lst.append(
                {
                    "frame_id": frame_id,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "conf": conf,
                    "score": score,
                }
            )
            # keep best-N by score
            lst.sort(key=lambda s: float(s.get("score", 0.0)), reverse=True)
            if len(lst) > int(cfg.jersey_max_samples_per_track):
                del lst[int(cfg.jersey_max_samples_per_track) :]

    # Keep only top tracks by their best sample score (cap total work).
    track_scores: List[Tuple[float, int]] = []
    for tid, samples in per_track.items():
        if not samples:
            continue
        track_scores.append((float(samples[0].get("score", 0.0)), tid))
    track_scores.sort(reverse=True)

    keep = set(tid for _, tid in track_scores[: int(cfg.jersey_max_tracks)])
    return {tid: per_track[tid] for tid in keep if tid in per_track}


def infer_jersey_numbers_from_tracking(
    *,
    video_path: str,
    tracks_csv_path: str,
    out_dir: str,
    cfg: FullPipelineConfig,
    progress_cb: Optional[Callable[[str, int, int, str], None]] = None,
) -> Dict[int, Dict[str, Any]]:
    """Return track_id -> {jersey_number, confidence, samples...}. Best-effort."""
    if not cfg.run_jersey_number_recognition:
        return {}

    if not os.path.isfile(tracks_csv_path) or not os.path.isfile(video_path):
        return {}

    by_frame = _parse_tracks_csv(tracks_csv_path)
    if not by_frame:
        return {}

    frame_ids = sorted(by_frame.keys())

    # Pre-pass: collect per-track metadata and (optionally) select top-N tracks by best bbox score.
    track_best_score: Dict[int, float] = {}
    track_first_frame: Dict[int, int] = {}
    track_last_frame: Dict[int, int] = {}
    track_team_counts: Dict[int, Dict[int, int]] = {}

    for fid in frame_ids:
        rows = by_frame.get(fid, [])
        for r in rows:
            try:
                cls_id = int(float(r.get("cls_id", -1)))
            except Exception:
                continue
            if cls_id != int(cfg.player_cls_id):
                continue
            try:
                track_id = int(float(r.get("track_id", 0)))
            except Exception:
                continue
            if track_id <= 0:
                continue
            if _is_special_track_id(track_id):
                continue
            try:
                conf = float(r.get("conf", 0.0))
            except Exception:
                conf = 0.0
            try:
                x1 = float(r.get("x1", 0.0))
                y1 = float(r.get("y1", 0.0))
                x2 = float(r.get("x2", 0.0))
                y2 = float(r.get("y2", 0.0))
                area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            except Exception:
                continue
            if area <= 1.0:
                continue

            score = float(area) * float(conf)
            prev = float(track_best_score.get(track_id, 0.0))
            if score > prev:
                track_best_score[track_id] = score

            track_first_frame[track_id] = min(int(track_first_frame.get(track_id, fid)), int(fid))
            track_last_frame[track_id] = max(int(track_last_frame.get(track_id, fid)), int(fid))

            team_raw = r.get("team_id", None)
            if team_raw is not None and str(team_raw).strip() != "":
                try:
                    team_id = int(float(team_raw))
                    counts = track_team_counts.setdefault(track_id, {})
                    counts[team_id] = int(counts.get(team_id, 0)) + 1
                except Exception:
                    pass

    if not track_best_score:
        return {}

    limit = int(getattr(cfg, "jersey_max_tracks", 0) or 0)
    if limit <= 0:
        keep_tracks = set(int(tid) for tid in track_best_score.keys())
    else:
        top = sorted(((s, tid) for tid, s in track_best_score.items()), reverse=True)
        keep_tracks = set(tid for _, tid in top[:limit])

    track_team: Dict[int, Optional[int]] = {}
    for tid, counts in track_team_counts.items():
        if not counts:
            track_team[tid] = None
            continue
        best_team = sorted(counts.items(), key=lambda kv: (kv[1], -kv[0]), reverse=True)[0][0]
        track_team[tid] = int(best_team)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {}
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    crop_dir = str(Path(out_dir) / "jersey_crops")
    os.makedirs(crop_dir, exist_ok=True)

    # Stateful scan: query on first appearance and then re-query only if still unknown (-1), with min gap.
    gap = max(1, int(cfg.jersey_min_frame_gap))
    max_attempts = max(1, int(cfg.jersey_max_samples_per_track))

    per_track_votes: Dict[int, Dict[str, int]] = {}
    per_track_raw: Dict[int, List[str]] = {}
    per_track_attempts: Dict[int, int] = {}
    per_track_last_query: Dict[int, int] = {}
    jersey_final: Dict[int, str] = {}
    jersey_conf: Dict[int, float] = {}
    jersey_source: Dict[int, int] = {}

    done = 0
    total_est = int(len(keep_tracks) * max_attempts)

    def best_vote(votes: Dict[str, int]) -> Tuple[Optional[str], float, int, int]:
        # returns (best_label, confidence, best_count, total_votes)
        if not votes:
            return (None, 0.0, 0, 0)
        items = sorted(votes.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
        best_lbl, best_cnt = items[0][0], int(items[0][1])
        tot = int(sum(int(x) for x in votes.values()))
        conf = float(best_cnt) / float(max(1, tot))
        return (str(best_lbl), conf, best_cnt, tot)

    for fid in frame_ids:
        rows = by_frame.get(fid, [])

        # Pick best player row per track in this frame for querying.
        cand: Dict[int, Dict[str, Any]] = {}
        cand_score: Dict[int, float] = {}
        cand_conf: Dict[int, float] = {}
        cand_area: Dict[int, float] = {}
        for r in rows:
            try:
                cls_id = int(float(r.get("cls_id", -1)))
            except Exception:
                continue
            if cls_id != int(cfg.player_cls_id):
                continue
            try:
                track_id = int(float(r.get("track_id", 0)))
            except Exception:
                continue
            if track_id <= 0 or track_id not in keep_tracks:
                continue
            if track_id in jersey_final:
                continue

            try:
                conf = float(r.get("conf", 0.0))
            except Exception:
                conf = 0.0

            try:
                x1 = float(r.get("x1", 0.0))
                y1 = float(r.get("y1", 0.0))
                x2 = float(r.get("x2", 0.0))
                y2 = float(r.get("y2", 0.0))
                area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            except Exception:
                continue
            if area <= 1.0:
                continue
            score = float(area) * float(conf)
            prev = float(cand_score.get(track_id, -1.0))
            if score > prev:
                cand_score[track_id] = score
                cand[track_id] = r
                cand_conf[track_id] = float(conf)
                cand_area[track_id] = float(area)

        # Propagate jersey via relink_source_id before running any Qwen calls.
        for track_id, r in list(cand.items()):
            src_raw = r.get("relink_source_id", None)
            if src_raw is None or str(src_raw).strip() == "":
                continue
            try:
                src_id = int(float(src_raw))
            except Exception:
                continue
            if src_id > 0 and src_id in jersey_final:
                jersey_final[track_id] = str(jersey_final[src_id])
                jersey_conf[track_id] = float(jersey_conf.get(src_id, 1.0))
                jersey_source[track_id] = int(src_id)

        # Decide which tracks to query at this frame.
        to_query: List[Tuple[int, Dict[str, Any]]] = []
        for track_id, r in cand.items():
            if track_id in jersey_final:
                continue
            attempts = int(per_track_attempts.get(track_id, 0))
            if attempts >= max_attempts:
                continue
            last_q = per_track_last_query.get(track_id, None)
            if last_q is not None and int(fid) - int(last_q) < gap:
                continue

            # Always allow the very first attempt for each track.
            if attempts > 0:
                c = float(cand_conf.get(track_id, 0.0))
                a = float(cand_area.get(track_id, 0.0))
                if c < float(cfg.jersey_min_det_conf) or a < float(cfg.jersey_min_box_area):
                    continue
            to_query.append((track_id, r))

        if not to_query:
            continue

        # Load the frame once for all queries in this frame.
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fid))
            ret, frame = cap.read()
        except Exception:
            ret, frame = False, None
        if not ret or frame is None:
            continue

        if (w and h) and (frame.shape[1] != w or frame.shape[0] != h):
            try:
                frame = cv2.resize(frame, (w, h))
            except Exception:
                pass

        for track_id, r in to_query:
            try:
                x1 = int(float(r.get("x1", 0)))
                y1 = int(float(r.get("y1", 0)))
                x2 = int(float(r.get("x2", 0)))
                y2 = int(float(r.get("y2", 0)))
            except Exception:
                continue

            crop = _jersey_crop_from_player_bbox(frame, x1=x1, y1=y1, x2=x2, y2=y2)
            if crop is None:
                continue

            crop_path = str(Path(crop_dir) / f"track{track_id}_frame{fid}.png")
            try:
                cv2.imwrite(crop_path, crop)
            except Exception:
                pass

            per_track_last_query[int(track_id)] = int(fid)
            per_track_attempts[int(track_id)] = int(per_track_attempts.get(int(track_id), 0)) + 1

            jersey, raw = _qwen_vl_ask_jersey_number(
                crop_bgr=crop,
                qwen_url=str(cfg.qwen_vl_url),
                model=str(getattr(cfg, "qwen_vl_model", "") or "qwen3vl8b"),
                prompt=str(getattr(cfg, "jersey_prompt", "") or ""),
            )
            if raw:
                per_track_raw.setdefault(int(track_id), []).append(str(raw))

            done += 1
            if progress_cb is not None:
                try:
                    progress_cb("jersey", done, max(1, total_est), f"Forma no okunuyor ({done})")
                except Exception:
                    pass

            if jersey is None:
                continue
            jersey_s = str(jersey).strip()
            if jersey_s == "-1":
                continue

            votes = per_track_votes.setdefault(int(track_id), {})
            votes[jersey_s] = int(votes.get(jersey_s, 0)) + 1
            best_lbl, conf, best_cnt, tot = best_vote(votes)

            # Accept immediately on first readable digits.
            if best_lbl is not None and best_cnt >= 1:
                jersey_final[int(track_id)] = str(best_lbl)
                jersey_conf[int(track_id)] = float(conf)

    cap.release()

    out: Dict[int, Dict[str, Any]] = {}
    # Include all attempted tracks: digits if found, otherwise -1 (so the caller can see it was processed).
    attempted_ids = set(int(t) for t in per_track_attempts.keys())
    for tid in attempted_ids:
        jersey_num = str(jersey_final.get(int(tid), "-1")).strip()
        votes = per_track_votes.get(int(tid), {})
        raws = per_track_raw.get(int(tid), [])
        out[int(tid)] = {
            "track_id": int(tid),
            "jersey_number": jersey_num,
            "confidence": float(jersey_conf.get(int(tid), 0.0 if jersey_num == "-1" else 1.0)),
            "votes": votes,
            "raw": raws[-3:],
            **({"relink_source_id": int(jersey_source[tid])} if int(tid) in jersey_source else {}),
            **(
                {"team_id": int(track_team[tid])}
                if tid in track_team and track_team[tid] is not None
                else {}
            ),
            **(
                {"first_frame": int(track_first_frame.get(tid, 0)), "last_frame": int(track_last_frame.get(tid, 0))}
                if tid in track_first_frame and tid in track_last_frame
                else {}
            ),
        }

    return out


def _resolve_track_id_remap(track_id: int, remap: Dict[int, int]) -> int:
    cur = int(track_id)
    seen = set()
    while cur in remap and cur not in seen:
        seen.add(cur)
        cur = int(remap[cur])
    return int(cur)


def _build_track_id_remap_from_jerseys(
    *,
    jersey_by_track: Dict[int, Dict[str, Any]],
    cfg: FullPipelineConfig,
) -> Dict[int, int]:
    """Return old_track_id -> canonical_track_id.

    Conservative: only merges within same team_id and same jersey_number,
    requires minimum confidence, and rejects merges where track lifetimes overlap.
    """

    if not bool(getattr(cfg, "jersey_merge_same_number", False)):
        return {}

    min_conf = float(getattr(cfg, "jersey_merge_min_confidence", 0.6) or 0.6)
    max_overlap = int(getattr(cfg, "jersey_merge_max_overlap_frames", 5) or 5)

    groups: Dict[Tuple[int, str], List[int]] = {}
    for tid, info in (jersey_by_track or {}).items():
        try:
            jersey = str(info.get("jersey_number", "")).strip()
            if not jersey or jersey == "-1":
                continue
            conf = float(info.get("confidence", 0.0))
            if conf < min_conf:
                continue
            team_raw = info.get("team_id", None)
            if team_raw is None:
                continue
            team_id = int(team_raw)
        except Exception:
            continue
        groups.setdefault((team_id, jersey), []).append(int(tid))

    remap: Dict[int, int] = {}

    def lifetime(tid: int) -> Tuple[int, int]:
        info = jersey_by_track.get(int(tid), {})
        try:
            a = int(info.get("first_frame", 0))
            b = int(info.get("last_frame", 0))
        except Exception:
            return (0, 0)
        return (min(a, b), max(a, b))

    for (_, _), tids in groups.items():
        uniq = sorted(set(int(x) for x in tids if int(x) > 0))
        if len(uniq) < 2:
            continue

        # Reject merging if any pair overlaps too much in time.
        ok = True
        ranges = {tid: lifetime(tid) for tid in uniq}
        for i in range(len(uniq)):
            for j in range(i + 1, len(uniq)):
                a0, a1 = ranges[uniq[i]]
                b0, b1 = ranges[uniq[j]]
                overlap = max(0, min(a1, b1) - max(a0, b0))
                if overlap > max_overlap:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            continue

        # Canonical = earliest first_frame, tie-breaker smallest id.
        uniq.sort(key=lambda tid: (lifetime(tid)[0], tid))
        canonical = int(uniq[0])
        for tid in uniq[1:]:
            remap[int(tid)] = canonical

    # Resolve chains
    for k in list(remap.keys()):
        remap[k] = _resolve_track_id_remap(remap[k], remap)
    return remap


def extract_segment_to_mp4(*, src_path: str, out_path: str, start_sec: float, duration_sec: Optional[float]) -> str:
    """Extract/re-encode a segment using ffmpeg if available, otherwise OpenCV."""
    src_path = str(Path(src_path).resolve())
    out_path = str(Path(out_path).resolve())

    out_dir = str(Path(out_path).parent)
    os.makedirs(out_dir, exist_ok=True)

    ffmpeg_bin = _ffmpeg_exe()
    if ffmpeg_bin:
        pbar = tqdm(total=1, desc="Segment", unit="job")
        cmd = [
            ffmpeg_bin,
            "-y",
            "-ss",
            str(float(start_sec or 0.0)),
            "-i",
            src_path,
        ]
        if duration_sec is not None:
            cmd += ["-t", str(float(duration_sec))]
        cmd += [
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            out_path,
        ]
        subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", check=True)
        try:
            pbar.update(1)
            pbar.close()
        except Exception:
            pass
    else:
        cap = cv2.VideoCapture(src_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {src_path}")

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)

        start_frame = int(float(start_sec or 0.0) * fps)
        if duration_sec is None:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            frames_to_write = max(0, total_frames - start_frame)
        else:
            frames_to_write = int(float(duration_sec) * fps)

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        if not writer.isOpened():
            cap.release()
            raise RuntimeError(f"Failed to create writer: {out_path}")

        written = 0
        pbar = tqdm(total=max(1, int(frames_to_write)), desc="Segment", unit="frame")
        for _ in range(frames_to_write):
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            if frame.shape[1] != w or frame.shape[0] != h:
                frame = cv2.resize(frame, (w, h))
            writer.write(frame)
            written += 1
            try:
                pbar.update(1)
            except Exception:
                pass
        try:
            pbar.close()
        except Exception:
            pass

        writer.release()
        cap.release()
        if written == 0:
            try:
                os.remove(out_path)
            except Exception:
                pass
            raise RuntimeError("No frames written during extraction")

    if not os.path.isfile(out_path) or os.path.getsize(out_path) == 0:
        raise RuntimeError(f"Extraction failed, output missing: {out_path}")

    return out_path


def _which(name: str) -> Optional[str]:
    # avoid importing shutil in hot path; simple PATH scan
    exts = [""]
    if os.name == "nt":
        exts = [".exe", ".cmd", ".bat", ""]

    paths = os.environ.get("PATH", "").split(os.pathsep)
    for base in paths:
        base = base.strip('"')
        for ext in exts:
            p = os.path.join(base, name + ext)
            if os.path.isfile(p):
                return p
    return None


def _ffmpeg_exe() -> Optional[str]:
    """Return a usable ffmpeg executable path.

    Tries system PATH first, then falls back to `imageio-ffmpeg` (which can
    provide an ffmpeg binary on Windows without a global install).
    """
    p = _which("ffmpeg")
    if p:
        return p
    try:
        import imageio_ffmpeg  # type: ignore

        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and os.path.isfile(exe):
            return exe
    except Exception:
        return None
    return None


def _write_tracks_csv_with_jersey(
    *,
    in_csv_path: str,
    out_csv_path: str,
    jersey_by_track: Dict[int, Dict[str, Any]],
    track_id_remap: Optional[Dict[int, int]] = None,
    player_cls_id: int = 0,
) -> str:
    """Copy tracking CSV and add a `jersey_number` column.

    For player rows (cls_id == player_cls_id), writes jersey digits or "-1".
    For non-player rows, leaves it empty.
    """

    track_id_remap = track_id_remap or {}
    jersey_by_track = jersey_by_track or {}

    with open(in_csv_path, "r", encoding="utf-8", errors="ignore") as rf:
        reader = csv.DictReader(rf)
        fieldnames = list(reader.fieldnames or [])
        if "jersey_number" not in fieldnames:
            fieldnames.append("jersey_number")

        os.makedirs(str(Path(out_csv_path).resolve().parent), exist_ok=True)
        with open(out_csv_path, "w", encoding="utf-8", newline="") as wf:
            writer = csv.DictWriter(wf, fieldnames=fieldnames)
            writer.writeheader()

            for row in reader:
                try:
                    cls_id = int(float(row.get("cls_id", -1)))
                except Exception:
                    cls_id = -1

                jersey_val = ""
                if cls_id == int(player_cls_id):
                    try:
                        tid_raw = int(float(row.get("track_id", 0)))
                    except Exception:
                        tid_raw = 0

                    if tid_raw > 0:
                        tid = _resolve_track_id_remap(int(tid_raw), track_id_remap)
                        info = jersey_by_track.get(int(tid))
                        if info is not None:
                            jersey_val = str(info.get("jersey_number", "-1")).strip()
                        else:
                            jersey_val = "-1"

                row["jersey_number"] = jersey_val
                writer.writerow(row)

    return out_csv_path


def run_tracking_reid_osnet(
    *,
    video_path: str,
    out_dir: str,
    device: Optional[str],
    config_path: Optional[str],
    detector_weights: Optional[str],
    reid_weights: Optional[str],
    cfg: Optional[FullPipelineConfig] = None,
    progress_cb: Optional[Callable[[str, int, int, str], None]] = None,
) -> Dict[str, Any]:
    repo = _repo_root()
    script = repo / "model-training" / "tracking-reid-osnet" / "run_botsort_team_reid.py"
    if not script.exists():
        raise FileNotFoundError(str(script))

    run_id = datetime.utcnow().strftime("%Y%m%d%H%M%S") + f"_{int(time.time()*1000)%100000:05d}"

    config_path = config_path or _default_tracking_config()
    if not config_path:
        raise FileNotFoundError("tracking config.yaml not found")

    # If OSNet is enabled in config but weights are missing, create a temp config disabling OSNet.
    # The tracking script does not expose osnet.enabled via CLI overrides.
    try:
        import yaml  # type: ignore

        cfg_obj = None
        with open(config_path, "r", encoding="utf-8") as f:
            cfg_obj = yaml.safe_load(f) or {}

        osnet = cfg_obj.get("osnet") if isinstance(cfg_obj, dict) else None
        if isinstance(osnet, dict) and bool(osnet.get("enabled", False)):
            w_rel = str(osnet.get("weights", "") or "").strip()
            # resolve weights relative to config file
            weights_path = Path(w_rel)
            if w_rel and not weights_path.is_absolute():
                weights_path = (Path(config_path).resolve().parent / weights_path).resolve()
            if (not w_rel) or (not weights_path.exists()):
                cfg_obj.setdefault("osnet", {})
                cfg_obj["osnet"]["enabled"] = False
                tmp_cfg = Path(out_dir) / f"tracking_config_{run_id}_no_osnet.yaml"
                with open(tmp_cfg, "w", encoding="utf-8") as wf:
                    yaml.safe_dump(cfg_obj, wf, sort_keys=False, allow_unicode=True)
                config_path = str(tmp_cfg)
    except Exception:
        # Best-effort; if yaml not available or parsing fails, we'll just run with given config.
        pass

    save_video = str(Path(out_dir) / f"tracking_{run_id}.mp4")
    save_txt = str(Path(out_dir) / f"tracking_{run_id}.csv")
    log_path = str(Path(out_dir) / f"tracking_{run_id}.log")

    cmd = [
        sys.executable,
        "-u",
        str(script),
        "--config",
        str(Path(config_path).resolve()),
        "--video",
        str(Path(video_path).resolve()),
        "--save_video",
        save_video,
        "--save_txt",
        save_txt,
    ]

    # Optional: run jersey inference during tracking (best-effort)
    jersey_json_path = str(Path(out_dir) / f"tracking_{run_id}.jersey.json")
    qwen_url = ""
    if cfg is not None:
        try:
            qwen_url = str(getattr(cfg, "qwen_vl_url", "") or "").strip()
        except Exception:
            qwen_url = ""

    jersey_cmd_enabled = False
    jersey_args: List[str] = []
    if (
        cfg is not None
        and bool(getattr(cfg, "run_jersey_number_recognition", False))
        and bool(getattr(cfg, "jersey_in_tracking", True))
        and bool(qwen_url)
    ):
        jersey_cmd_enabled = True
        jersey_args = [
            "--jersey_enable",
            "1",
            "--jersey_qwen_url",
            qwen_url,
            "--jersey_model",
            str(getattr(cfg, "qwen_vl_model", "qwen3vl8b")),
            "--jersey_prompt",
            str(getattr(cfg, "jersey_prompt", "") or ""),
            "--jersey_min_frame_gap",
            str(int(getattr(cfg, "jersey_min_frame_gap", 20))),
            "--jersey_max_samples_per_track",
            str(int(getattr(cfg, "jersey_max_samples_per_track", 3))),
            "--jersey_min_det_conf",
            str(float(getattr(cfg, "jersey_min_det_conf", 0.55))),
            "--jersey_min_box_area",
            str(int(getattr(cfg, "jersey_min_box_area", 1600))),
            "--jersey_frame_topk",
            str(int(getattr(cfg, "jersey_frame_topk", 5))),
            "--jersey_vis_filter",
            "1" if bool(getattr(cfg, "jersey_vis_filter", False)) else "0",
            "--jersey_vis_min_score",
            str(float(getattr(cfg, "jersey_vis_min_score", 0.12))),
            "--jersey_out_json",
            jersey_json_path,
        ]

        cmd += jersey_args

        crops_dir = getattr(cfg, "jersey_crops_dir", None)
        if crops_dir is not None and str(crops_dir).strip():
            jersey_args += ["--jersey_crops_dir", str(crops_dir)]
            cmd += ["--jersey_crops_dir", str(crops_dir)]

    if device:
        cmd += ["--device", device]
    if detector_weights:
        cmd += ["--detector_weights", detector_weights]
    if reid_weights:
        cmd += ["--reid_weights", reid_weights]

    def _run_tracking_subprocess(cmd_to_run: List[str], log_file: str) -> subprocess.Popen:
        os.makedirs(str(Path(log_file).resolve().parent), exist_ok=True)
        lf = open(log_file, "w", encoding="utf-8", errors="ignore")
        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")
        env.setdefault("PYTHONFAULTHANDLER", "1")
        env.setdefault("TORCH_SHOW_CPP_STACKTRACES", "1")
        p = subprocess.Popen(cmd_to_run, stdout=lf, stderr=subprocess.STDOUT, text=True, env=env)
        # attach so caller can close it later
        p._pipeline_log_handle = lf  # type: ignore[attr-defined]
        return p

    def _strip_jersey_flags(argv: List[str]) -> List[str]:
        """Remove in-tracking jersey CLI flags from argv (best-effort)."""
        jersey_flags_1 = {
            "--jersey_enable",
            "--jersey_qwen_url",
            "--jersey_model",
            "--jersey_prompt",
            "--jersey_min_frame_gap",
            "--jersey_max_samples_per_track",
            "--jersey_min_det_conf",
            "--jersey_min_box_area",
            "--jersey_frame_topk",
            "--jersey_vis_filter",
            "--jersey_vis_min_score",
            "--jersey_out_json",
            "--jersey_crops_dir",
        }

        out: List[str] = []
        i = 0
        n = len(argv)
        while i < n:
            tok = argv[i]
            if tok in jersey_flags_1:
                i += 2  # drop flag + its value
                continue
            out.append(tok)
            i += 1
        return out

    start_ts = time.time()
    p = _run_tracking_subprocess(cmd, log_path)

    # Prefer progress in frames/FPS (parsed from the tracking script logs).
    # Fallback: elapsed-seconds ticker (but without misleading rate info).
    progress_re = re.compile(
        r"\[(?P<cur>\d+)\/(?P<total>\d+)\]\s+(?P<pct>[0-9.]+)%\s+\|\s+(?P<fps>[0-9.]+)\s+FPS\s+\|\s+ETA\s+(?P<eta>\d\d:\d\d:\d\d)"
    )

    last_frame = 0
    inferred_total: Optional[int] = None
    pbar: Optional[tqdm] = None
    tick_pbar = tqdm(total=None, desc="Tracking", unit="s", bar_format="{desc}: {elapsed}")

    try:
        # Open the log for tailing.
        try:
            rf = open(log_path, "r", encoding="utf-8", errors="ignore")
        except Exception:
            rf = None

        while True:
            rc = p.poll()
            any_update = False

            # Tail log and parse progress lines.
            if rf is not None:
                while True:
                    line = rf.readline()
                    if not line:
                        break
                    m = progress_re.search(line)
                    if not m:
                        continue

                        try:
                            cur = int(m.group("cur"))
                            total = int(m.group("total"))
                            fps_val = float(m.group("fps"))
                            eta = str(m.group("eta"))
                        except Exception:
                            continue

                        if inferred_total is None and total > 0:
                            inferred_total = total
                            try:
                                pbar = tqdm(total=inferred_total, desc="Tracking", unit="frame")
                            except Exception:
                                pbar = None

                        if cur >= last_frame:
                            delta = cur - last_frame
                            last_frame = cur
                            if pbar is not None and delta > 0:
                                try:
                                    pbar.update(delta)
                                except Exception:
                                    pass

                        if progress_cb is not None and total > 0:
                            try:
                                pct = int((cur * 100) / max(1, total))
                                progress_cb(
                                    "tracking",
                                    cur,
                                    total,
                                    f"Tracking {pct}% | {fps_val:.2f} FPS | ETA {eta}",
                                )
                            except Exception:
                                pass

                        any_update = True

                if rc is not None:
                    break

                # If we didn't get a parsed progress line, still tick elapsed time.
                if not any_update:
                    time.sleep(1.0)
                    try:
                        tick_pbar.update(1)
                    except Exception:
                        pass
                    if progress_cb is not None:
                        try:
                            elapsed = int(time.time() - start_ts)
                            progress_cb("tracking", elapsed, 0, "Tracking çalışıyor")
                        except Exception:
                            pass
                else:
                    time.sleep(0.2)
    finally:
        try:
            if rf is not None:
                rf.close()
        except Exception:
            pass
        try:
            if pbar is not None:
                pbar.close()
        except Exception:
            pass
        try:
            tick_pbar.close()
        except Exception:
            pass

        try:
            lh = getattr(p, "_pipeline_log_handle", None)
            if lh is not None:
                lh.close()
        except Exception:
            pass

    # If the jersey-enabled run crashed (often native/OOM with no traceback), retry once without jersey.
    tracking_retry_log_path: Optional[str] = None
    if p.returncode != 0 and jersey_cmd_enabled:
        try:
            # Keep the failing log; write retry to a new file.
            tracking_retry_log_path = str(Path(out_dir) / f"tracking_{run_id}_retry_no_jersey.log")

            # Remove potentially partial outputs before retry.
            for pp in (save_video, save_txt):
                try:
                    if os.path.isfile(pp):
                        os.remove(pp)
                except Exception:
                    pass

            cmd_no_jersey = _strip_jersey_flags(list(cmd))

            p = _run_tracking_subprocess(cmd_no_jersey, tracking_retry_log_path)
            # Tail retry log for progress updates (best-effort).
            rf2 = None
            try:
                rf2 = open(tracking_retry_log_path, "r", encoding="utf-8", errors="ignore")
            except Exception:
                rf2 = None

            retry_last_frame = 0
            retry_total: Optional[int] = None
            while True:
                rc2 = p.poll()
                any_update2 = False

                if rf2 is not None:
                    while True:
                        line2 = rf2.readline()
                        if not line2:
                            break
                        m2 = progress_re.search(line2)
                        if not m2:
                            continue
                        try:
                            cur2 = int(m2.group("cur"))
                            total2 = int(m2.group("total"))
                            fps2 = float(m2.group("fps"))
                            eta2 = str(m2.group("eta"))
                        except Exception:
                            continue

                        if retry_total is None and total2 > 0:
                            retry_total = total2
                        if cur2 >= retry_last_frame:
                            retry_last_frame = cur2

                        if progress_cb is not None and total2 > 0:
                            try:
                                pct2 = int((cur2 * 100) / max(1, total2))
                                progress_cb(
                                    "tracking",
                                    cur2,
                                    total2,
                                    f"Tracking (retry) {pct2}% | {fps2:.2f} FPS | ETA {eta2}",
                                )
                            except Exception:
                                pass

                        any_update2 = True

                if rc2 is not None:
                    break

                if not any_update2:
                    time.sleep(1.0)
                    if progress_cb is not None:
                        try:
                            elapsed2 = int(time.time() - start_ts)
                            progress_cb("tracking", elapsed2, 0, "Tracking (retry) çalışıyor")
                        except Exception:
                            pass
                else:
                    time.sleep(0.2)

            try:
                if rf2 is not None:
                    rf2.close()
            except Exception:
                pass
            try:
                lh2 = getattr(p, "_pipeline_log_handle", None)
                if lh2 is not None:
                    lh2.close()
            except Exception:
                pass
            log_path = tracking_retry_log_path
        except Exception:
            # If retry setup fails, fall through to standard error handling below.
            pass

    if p.returncode != 0:
        tail = ""
        try:
            with open(log_path, "r", encoding="utf-8", errors="ignore") as rf:
                tail = rf.read()[-4000:]
        except Exception:
            tail = ""
        rc = p.returncode
        rc_hex = None
        try:
            if isinstance(rc, int) and rc < 0:
                rc_hex = hex((1 << 32) + rc)
            elif isinstance(rc, int):
                rc_hex = hex(rc)
        except Exception:
            rc_hex = None
        raise RuntimeError(
            "tracking-reid-osnet failed\n"
            f"cmd: {' '.join(cmd)}\n"
            f"returncode: {rc} ({rc_hex})\n"
            f"log: {log_path}\n"
            f"tail: {tail}"
        )

    if not os.path.isfile(save_video):
        raise RuntimeError(f"Tracking video not produced: {save_video}\nlog: {log_path}")
    if not os.path.isfile(save_txt):
        raise RuntimeError(f"Tracking CSV not produced: {save_txt}\nlog: {log_path}")

    jersey_by_track: Dict[int, Dict[str, Any]] = {}
    track_id_remap: Dict[int, int] = {}
    out_csv_path = save_txt

    # Prefer jersey mapping produced during tracking.
    jersey_in_tracking_cfg = bool(getattr(cfg, "jersey_in_tracking", True)) if cfg is not None else False
    if cfg is not None and bool(getattr(cfg, "run_jersey_number_recognition", False)) and jersey_in_tracking_cfg:
        try:
            if os.path.isfile(jersey_json_path):
                with open(jersey_json_path, "r", encoding="utf-8") as f:
                    items = json.load(f) or []
                for it in items:
                    try:
                        tid = int(it.get("track_id"))
                        jersey_by_track[int(tid)] = dict(it)
                    except Exception:
                        continue
        except Exception:
            jersey_by_track = {}

        # If jersey was requested in-tracking but didn't produce outputs (e.g., retry without jersey),
        # fall back to post-tracking inference.
        if not jersey_by_track:
            try:
                jersey_by_track = infer_jersey_numbers_from_tracking(
                    video_path=video_path,
                    tracks_csv_path=save_txt,
                    out_dir=out_dir,
                    cfg=cfg,
                    progress_cb=progress_cb,
                )
            except Exception:
                jersey_by_track = {}

        try:
            track_id_remap = _build_track_id_remap_from_jerseys(jersey_by_track=jersey_by_track, cfg=cfg)
        except Exception:
            track_id_remap = {}

        if track_id_remap and jersey_by_track:
            canonical: Dict[int, Dict[str, Any]] = {}
            for tid, info in jersey_by_track.items():
                canon = _resolve_track_id_remap(int(tid), track_id_remap)
                cur = canonical.get(int(canon))
                if cur is None:
                    canonical[int(canon)] = {**info, "track_id": int(canon)}
                    continue
                try:
                    if float(info.get("confidence", 0.0)) > float(cur.get("confidence", 0.0)):
                        canonical[int(canon)] = {**info, "track_id": int(canon)}
                except Exception:
                    pass
            jersey_by_track = canonical

        # Enrich CSV in a separate file (keeps original script output intact)
        try:
            enriched = str(Path(out_dir) / f"tracking_{run_id}_with_jersey.csv")
            out_csv_path = _write_tracks_csv_with_jersey(
                in_csv_path=save_txt,
                out_csv_path=enriched,
                jersey_by_track=jersey_by_track,
                track_id_remap=track_id_remap,
                player_cls_id=int(getattr(cfg, "player_cls_id", 0)),
            )
        except Exception:
            out_csv_path = save_txt

    # Fallback: if jersey was not done in tracking, infer after tracking.
    elif cfg is not None and bool(getattr(cfg, "run_jersey_number_recognition", False)):
        try:
            jersey_by_track = infer_jersey_numbers_from_tracking(
                video_path=video_path,
                tracks_csv_path=save_txt,
                out_dir=out_dir,
                cfg=cfg,
                progress_cb=progress_cb,
            )
        except Exception:
            jersey_by_track = {}

        try:
            track_id_remap = _build_track_id_remap_from_jerseys(jersey_by_track=jersey_by_track, cfg=cfg)
        except Exception:
            track_id_remap = {}

        try:
            enriched = str(Path(out_dir) / f"tracking_{run_id}_with_jersey.csv")
            out_csv_path = _write_tracks_csv_with_jersey(
                in_csv_path=save_txt,
                out_csv_path=enriched,
                jersey_by_track=jersey_by_track,
                track_id_remap=track_id_remap,
                player_cls_id=int(getattr(cfg, "player_cls_id", 0)),
            )
        except Exception:
            out_csv_path = save_txt

    return {
        "tracking_video_path": save_video,
        "tracks_csv_path": out_csv_path,
        "tracks_csv_raw_path": save_txt,
        "jersey_json_path": jersey_json_path if os.path.isfile(jersey_json_path) else None,
        "jersey_by_track": jersey_by_track,
        "track_id_remap": track_id_remap,
        "tracking_log_path": log_path,
        **({"tracking_retry_log_path": tracking_retry_log_path} if tracking_retry_log_path else {}),
        "tracking_seconds": float(time.time() - start_ts),
    }


def run_action_spotting_spotting_v2(
    *,
    features_path: str,
    checkpoint_path: Optional[str],
    threshold: float,
    nms_window_sec: float,
) -> List[Dict[str, Any]]:
    repo = _repo_root()
    module_dir = repo / "model-training" / "action_spotting" / "spotting_v2"
    if not module_dir.exists():
        raise FileNotFoundError(str(module_dir))

    features_path = str(Path(features_path).resolve())
    if not os.path.isfile(features_path):
        raise FileNotFoundError(features_path)

    checkpoint_path = checkpoint_path or _default_action_checkpoint()
    if not checkpoint_path:
        raise FileNotFoundError("Action spotting checkpoint not found")
    checkpoint_path = str(Path(checkpoint_path).resolve())

    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))

    try:
        import inference as spotting_inference  # type: ignore
    except Exception as e:
        raise RuntimeError(f"Failed to import spotting_v2 inference.py: {e}")

    model = spotting_inference.load_model(checkpoint_path)
    preds = spotting_inference.predict(model, features_path, threshold=threshold)
    final_preds = spotting_inference.nms(preds, window_sec=nms_window_sec)

    # normalize output
    out: List[Dict[str, Any]] = []
    for p in final_preds:
        t = float(p.get("time", 0.0))
        label = str(p.get("label", ""))
        score = float(p.get("score", 0.0))
        out.append(
            {
                "type": "soccer_event",
                "source": "action_spotting",
                "label": label,
                "t": t,
                "timecode": _timecode(t),
                "confidence": score,
                "description_tr": _event_desc_tr(label),
            }
        )
    return out


def _parse_tracks_csv(csv_path: str) -> Dict[int, List[Dict[str, Any]]]:
    by_frame: Dict[int, List[Dict[str, Any]]] = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                frame_id = int(float(row["frame_id"]))
            except Exception:
                continue
            by_frame.setdefault(frame_id, []).append(row)
    return by_frame


def _xyxy_center(row: Dict[str, Any]) -> Tuple[float, float]:
    x1 = float(row["x1"])
    y1 = float(row["y1"])
    x2 = float(row["x2"])
    y2 = float(row["y2"])
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def derive_possession_events_from_tracks(
    *,
    tracks_csv_path: str,
    fps: float,
    width: int,
    height: int,
    cfg: FullPipelineConfig,
) -> List[Dict[str, Any]]:
    by_frame = _parse_tracks_csv(tracks_csv_path)
    diag = float(np.sqrt(width * width + height * height)) + 1e-8

    # iterate frames in order, with stride
    frame_ids = sorted(by_frame.keys())
    owners: List[Tuple[float, Optional[int]]] = []

    stride = max(1, int(cfg.possession_stride_frames))
    ids = frame_ids[::stride]
    for fid in tqdm(ids, total=len(ids) if ids else None, desc="Possession", unit="frame"):
        rows = by_frame.get(fid, [])

        # select ball
        ball_rows = [r for r in rows if int(float(r.get("cls_id", -1))) == int(cfg.ball_cls_id)]
        if not ball_rows:
            owners.append((fid / fps, None))
            continue

        ball_row = max(ball_rows, key=lambda r: float(r.get("conf", 0.0)))
        bx, by = _xyxy_center(ball_row)

        # players
        player_rows = [r for r in rows if int(float(r.get("cls_id", -1))) == int(cfg.player_cls_id)]
        best_id: Optional[int] = None
        best_dist = 1e9
        for pr in player_rows:
            px, py = _xyxy_center(pr)
            d = float(np.sqrt((bx - px) ** 2 + (by - py) ** 2)) / diag
            if d < best_dist:
                best_dist = d
                try:
                    best_id = int(float(pr.get("track_id", 0)))
                except Exception:
                    best_id = None

        if best_id is not None and best_dist <= float(cfg.possession_dist_norm):
            owners.append((fid / fps, best_id))
        else:
            owners.append((fid / fps, None))

    # stable segments -> events
    stable_n = max(1, int(cfg.possession_stable_frames))
    window: List[Optional[int]] = []
    cur_stable: Optional[int] = None
    prev_emitted: Optional[int] = None
    events: List[Dict[str, Any]] = []

    def emit(t: float, new_owner: Optional[int], prev_owner: Optional[int]):
        if new_owner == prev_owner:
            return
        if new_owner is None:
            etype = "ball_uncontrolled"
            desc = "Top kontrolsüz kaldı."
        elif prev_owner is None:
            etype = "possession_start"
            desc = f"Top kontrolü #{new_owner} oyuncusuna geçti."
        else:
            etype = "possession_change"
            desc = f"Top kontrolü #{prev_owner} -> #{new_owner} değişti."

        events.append(
            {
                "type": etype,
                "source": "tracking",
                "t": float(t),
                "timecode": _timecode(float(t)),
                "player_track_id": int(new_owner) if new_owner is not None else None,
                "from_player_track_id": int(prev_owner) if prev_owner is not None else None,
                "confidence": 0.55,
                "description_tr": desc,
            }
        )

    for t, owner in owners:
        window.append(owner)
        if len(window) > stable_n:
            window.pop(0)
        if len(window) < stable_n:
            continue
        if all(o == window[0] for o in window):
            stable_owner = window[0]
            if stable_owner != cur_stable:
                cur_stable = stable_owner
                # emit immediately on stable confirmation
                if cur_stable != prev_emitted:
                    emit(t, cur_stable, prev_emitted)
                    prev_emitted = cur_stable

    return events


def overlay_events_on_video(
    *,
    base_video_path: str,
    out_path: str,
    events: List[Dict[str, Any]],
    window_sec: float,
    tracks_csv_path: Optional[str] = None,
    jersey_by_track: Optional[Dict[int, Dict[str, Any]]] = None,
    track_id_remap: Optional[Dict[int, int]] = None,
    progress_cb: Optional[Callable[[str, int, int, str], None]] = None,
) -> str:
    out_path = str(Path(out_path).resolve())
    tmp_path = str(Path(out_path).with_suffix(".tmp.mp4"))

    cap = cv2.VideoCapture(base_video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open base video: {base_video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_path, fourcc, fps, (w, h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to open writer: {tmp_path}")

    # build event display intervals
    intervals: List[Tuple[float, float, str]] = []
    for e in events:
        try:
            t = float(e.get("t", 0.0))
        except Exception:
            continue
        label = str(e.get("label") or e.get("type") or "event")
        score = e.get("confidence")
        text = f"{_timecode(t)}  {label}" + (f"  ({float(score):.2f})" if isinstance(score, (float, int)) else "")
        intervals.append((t, t + float(window_sec), text))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_idx = 0
    pbar = tqdm(total=total_frames if total_frames > 0 else None, desc="Overlay", unit="frame")
    # If we have tracking CSV, stream it alongside video frames (memory-light).
    csv_f = None
    csv_reader = None
    next_row: Optional[Dict[str, Any]] = None
    if tracks_csv_path and os.path.isfile(tracks_csv_path):
        try:
            csv_f = open(tracks_csv_path, "r", encoding="utf-8", errors="ignore")
            csv_reader = csv.DictReader(csv_f)
            next_row = next(csv_reader, None)
        except Exception:
            csv_reader = None
            next_row = None

    jersey_by_track = jersey_by_track or {}
    track_id_remap = track_id_remap or {}

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        t = frame_idx / fps

        active = [txt for (t0, t1, txt) in intervals if t0 <= t <= t1]
        draw = frame
        y = 30
        for txt in active[:3]:
            cv2.putText(draw, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3)
            cv2.putText(draw, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
            y += 30

        # Draw tracking boxes + labels (ID + jersey number; no team label)
        if csv_reader is not None:
            # advance csv to current frame
            rows: List[Dict[str, Any]] = []
            try:
                while next_row is not None:
                    fid = int(float(next_row.get("frame_id", -1)))
                    if fid < frame_idx:
                        next_row = next(csv_reader, None)
                        continue
                    if fid > frame_idx:
                        break
                    rows.append(next_row)
                    next_row = next(csv_reader, None)
            except Exception:
                rows = []

            for r in rows:
                try:
                    cls_id = int(float(r.get("cls_id", -1)))
                    if cls_id != 0:
                        continue
                    track_id_raw = int(float(r.get("track_id", 0)))
                    if track_id_raw <= 0:
                        continue
                    track_id = _resolve_track_id_remap(int(track_id_raw), track_id_remap)
                    info = jersey_by_track.get(int(track_id))
                    jersey_val = (info or {}).get("jersey_number")
                    jersey = str(jersey_val).strip() if jersey_val is not None else "-1"

                    x1 = int(float(r.get("x1", 0)))
                    y1 = int(float(r.get("y1", 0)))
                    x2 = int(float(r.get("x2", 0)))
                    y2 = int(float(r.get("y2", 0)))
                except Exception:
                    continue

                # box
                x1 = _clamp_int(x1, 0, draw.shape[1] - 1)
                x2 = _clamp_int(x2, 0, draw.shape[1] - 1)
                y1 = _clamp_int(y1, 0, draw.shape[0] - 1)
                y2 = _clamp_int(y2, 0, draw.shape[0] - 1)
                if x2 <= x1 or y2 <= y1:
                    continue

                cv2.rectangle(draw, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # label (ID + NO)
                label = f"#{track_id}  {jersey}"
                y_text = max(14, y1 - 6)
                cv2.putText(draw, label, (x1, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
                cv2.putText(draw, label, (x1, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        writer.write(draw)
        frame_idx += 1
        try:
            pbar.update(1)
        except Exception:
            pass
        if progress_cb is not None and total_frames > 0:
            try:
                progress_cb("overlay", frame_idx, total_frames, "Overlay yazılıyor")
            except Exception:
                pass

    cap.release()
    writer.release()
    try:
        if csv_f is not None:
            csv_f.close()
    except Exception:
        pass
    try:
        pbar.close()
    except Exception:
        pass

    # Ensure browser playback: re-encode to H.264 (yuv420p) and move moov atom to start.
    ffmpeg_bin = _ffmpeg_exe()
    if ffmpeg_bin and os.path.isfile(tmp_path):
        try:
            cmd = [
                ffmpeg_bin,
                "-y",
                "-i",
                tmp_path,
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-preset",
                "fast",
                "-crf",
                "20",
                "-an",
                "-movflags",
                "+faststart",
                out_path,
            ]
            subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", check=True)
            try:
                os.remove(tmp_path)
            except Exception:
                pass
            return out_path
        except Exception:
            # If remux fails, fall back to the original file.
            pass

    # Fallback: no ffmpeg available or remux failed.
    try:
        if os.path.isfile(out_path):
            os.remove(out_path)
    except Exception:
        pass
    try:
        os.replace(tmp_path, out_path)
    except Exception:
        # last resort
        out_path = tmp_path
    return out_path


def run_full_pipeline(
    *,
    video_path: str,
    out_dir: str,
    cfg: FullPipelineConfig,
    progress_cb: Optional[Callable[[str, int, int, str], None]] = None,
) -> Dict[str, Any]:
    out_dir = str(Path(out_dir).resolve())
    os.makedirs(out_dir, exist_ok=True)

    run_id = datetime.utcnow().strftime("%Y%m%d%H%M%S") + f"_{int(time.time()*1000)%100000:05d}"

    def emit(stage: str, cur: int, total: int, msg: str) -> None:
        if progress_cb is None:
            return
        try:
            progress_cb(stage, int(cur), int(total), str(msg))
        except Exception:
            pass

    segment_path = str(Path(out_dir) / f"segment_{run_id}.mp4")
    emit("segment", 0, 1, "Video segment hazırlanıyor")
    extract_segment_to_mp4(
        src_path=video_path,
        out_path=segment_path,
        start_sec=float(cfg.start_seconds or 0.0),
        duration_sec=cfg.duration_seconds,
    )
    emit("segment", 1, 1, "Video segment hazır")

    calibration_map_video_path: Optional[str] = None
    calibration_events_json_path: Optional[str] = None
    calibration_frames_jsonl_path: Optional[str] = None
    calibration_events: List[Dict[str, Any]] = []

    if bool(getattr(cfg, "run_calibration", True)):
        emit("calibration", 0, 1, "Calibration başlıyor")
        try:
            out_map = str(Path(out_dir) / f"map_{run_id}.mp4")
            out_events = str(Path(out_dir) / f"calibration_events_{run_id}.json")
            out_frames = None
            if bool(getattr(cfg, "calibration_write_frames_jsonl", False)):
                out_frames = str(Path(out_dir) / f"calibration_frames_{run_id}.jsonl")
            calib_res = run_calibration_pipeline(
                video_path=segment_path,
                out_map=out_map,
                out_events=out_events,
                out_frames=out_frames,
                cfg=cfg,
                progress_cb=progress_cb,
            )

            calibration_map_video_path = str(calib_res.get("map_video_path") or out_map)
            calibration_events_json_path = str(calib_res.get("events_json_path") or out_events)
            try:
                calibration_frames_jsonl_path = str(calib_res.get("frames_jsonl_path") or "")
            except Exception:
                calibration_frames_jsonl_path = None
            if calibration_frames_jsonl_path and not os.path.isfile(str(calibration_frames_jsonl_path)):
                calibration_frames_jsonl_path = None

            # Import calibration events into the unified manifest event list.
            if calibration_events_json_path and os.path.isfile(calibration_events_json_path):
                try:
                    with open(calibration_events_json_path, "r", encoding="utf-8") as f:
                        ce = json.load(f) or {}
                    evs = ce.get("events") if isinstance(ce, dict) else None
                    if isinstance(evs, list):
                        for e in evs:
                            if isinstance(e, dict):
                                calibration_events.append(e)
                except Exception:
                    pass

            # Make map video browser-playable (H.264 + faststart) when possible.
            try:
                ffmpeg_bin = _ffmpeg_exe()
                if ffmpeg_bin and calibration_map_video_path and os.path.isfile(calibration_map_video_path):
                    src = str(calibration_map_video_path)
                    tmp = str(Path(out_dir) / f"map_{run_id}.tmp_h264.mp4")
                    cmd = [
                        ffmpeg_bin,
                        "-y",
                        "-i",
                        src,
                        "-c:v",
                        "libx264",
                        "-pix_fmt",
                        "yuv420p",
                        "-preset",
                        "fast",
                        "-crf",
                        "20",
                        "-an",
                        "-movflags",
                        "+faststart",
                        tmp,
                    ]
                    subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", check=True)
                    try:
                        os.replace(tmp, src)
                    except Exception:
                        calibration_map_video_path = tmp
            except Exception:
                pass
        except Exception:
            # Best-effort; keep pipeline running.
            calibration_map_video_path = None
            calibration_events_json_path = None
            calibration_frames_jsonl_path = None
            calibration_events = []
        emit("calibration", 1, 1, "Calibration tamam")

    tracking_video_path: Optional[str] = None
    tracks_csv_path: Optional[str] = None
    tracks_csv_raw_path: Optional[str] = None
    tracking_res: Optional[Dict[str, Any]] = None

    if cfg.run_tracking:
        emit("tracking", 0, 1, "Tracking başlıyor")
        tracking_res = run_tracking_reid_osnet(
            video_path=segment_path,
            out_dir=out_dir,
            device=cfg.tracking_device,
            config_path=cfg.tracking_config_path,
            detector_weights=cfg.detector_weights,
            reid_weights=cfg.reid_weights,
            cfg=cfg,
            progress_cb=progress_cb,
        )
        tracking_video_path = tracking_res["tracking_video_path"]
        tracks_csv_path = tracking_res["tracks_csv_path"]
        tracks_csv_raw_path = tracking_res.get("tracks_csv_raw_path")
        emit("tracking", 1, 1, "Tracking tamam")

    events: List[Dict[str, Any]] = []
    if calibration_events:
        events.extend(calibration_events)

    jersey_by_track: Dict[int, Dict[str, Any]] = {}
    track_id_remap: Dict[int, int] = {}

    # Derive possession events from tracking CSV (if available)
    if tracks_csv_path and os.path.isfile(tracks_csv_path):
        cap = cv2.VideoCapture(segment_path)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
        cap.release()
        try:
            events.extend(
                derive_possession_events_from_tracks(
                    tracks_csv_path=tracks_csv_path,
                    fps=fps,
                    width=w,
                    height=h,
                    cfg=cfg,
                )
            )
        except Exception:
            # possession is best-effort
            pass

        # Jersey results may already be produced during tracking.
        if isinstance(tracking_res, dict):
            try:
                jersey_by_track = tracking_res.get("jersey_by_track") or {}
                track_id_remap = tracking_res.get("track_id_remap") or {}
            except Exception:
                jersey_by_track = {}
                track_id_remap = {}

        # If not produced during tracking (or tracking was skipped), run jersey inference here.
        if bool(cfg.run_jersey_number_recognition) and (not jersey_by_track) and tracks_csv_path and os.path.isfile(tracks_csv_path):
            emit("jersey", 0, 1, "Forma numarası okunuyor")
            try:
                jersey_by_track = infer_jersey_numbers_from_tracking(
                    video_path=segment_path,
                    tracks_csv_path=tracks_csv_path,
                    out_dir=out_dir,
                    cfg=cfg,
                    progress_cb=progress_cb,
                )
            except Exception:
                jersey_by_track = {}
            emit("jersey", 1, 1, "Forma numarası hazır")

            try:
                track_id_remap = _build_track_id_remap_from_jerseys(jersey_by_track=jersey_by_track, cfg=cfg)
            except Exception:
                track_id_remap = {}

    # Action spotting events
    if cfg.run_action_spotting:
        emit("action_spotting", 0, 1, "Action spotting başlıyor")
        if not cfg.features_path:
            # Auto-extract PCA512 features (SoccerNet sn-spotting style) when not provided.
            # This produces a single .npy with shape (T, 512) at ~2 FPS.
            try:
                from pcaextractor import extract_resnet_tf2_pca512  # type: ignore
            except Exception as e:
                raise RuntimeError(f"Failed to import pcaextractor: {e}")

            auto_feat_path = str(Path(out_dir) / f"features_{run_id}_ResNET_TF2_PCA512.npy")
            cfg.features_path = extract_resnet_tf2_pca512(
                video_path=segment_path,
                out_features_npy=auto_feat_path,
                fps=2.0,
                transform="crop",
                overwrite=False,
                progress_cb=progress_cb,
            )
        emit("action_spotting", 0, 1, "Model inference çalışıyor")
        events.extend(
            run_action_spotting_spotting_v2(
                features_path=cfg.features_path,
                checkpoint_path=cfg.checkpoint_path,
                threshold=float(cfg.action_threshold),
                nms_window_sec=float(cfg.action_nms_window_sec),
            )
        )
        emit("action_spotting", 1, 1, "Action spotting tamam")

    # Sort events by time
    events.sort(key=lambda e: float(e.get("t", 0.0)))

    # Apply track-id remap to tracking-derived event fields.
    if track_id_remap:
        for e in events:
            for k in ("player_track_id", "from_player_track_id"):
                try:
                    v = e.get(k)
                    if v is None:
                        continue
                    e[k] = _resolve_track_id_remap(int(v), track_id_remap)
                except Exception:
                    continue

    # Attach jersey numbers to tracking-derived events (if available)
    if jersey_by_track:
        for e in events:
            try:
                pid = e.get("player_track_id")
                if pid is not None:
                    info = jersey_by_track.get(int(pid))
                    if info and info.get("jersey_number") is not None:
                        e["jersey_number"] = info.get("jersey_number")
                fpid = e.get("from_player_track_id")
                if fpid is not None:
                    info2 = jersey_by_track.get(int(fpid))
                    if info2 and info2.get("jersey_number") is not None:
                        e["from_jersey_number"] = info2.get("jersey_number")
            except Exception:
                continue

    # Commentary artifacts (filled after overlay generation)
    commentary_input_path: Optional[str] = None
    commentary_output_path: Optional[str] = None
    commentary_audio_manifest_path: Optional[str] = None
    commentary_video_path: Optional[str] = None
    # Product output: should be a clean (no boxes) video, optionally with commentary audio.
    product_video_path: str = segment_path

    # The user-facing overlay should focus on action spotting events.
    overlay_events = [e for e in events if str(e.get("source")) == "action_spotting"]
    if not overlay_events:
        overlay_events = events

    # For a clean overlay (no team labels baked into tracker video), draw boxes from CSV
    # on top of the original segment.
    base_video = segment_path
    final_overlay_path = str(Path(out_dir) / f"overlay_{run_id}.mp4")
    emit("overlay", 0, 1, "Overlay başlıyor")
    overlay_events_on_video(
        base_video_path=base_video,
        out_path=final_overlay_path,
        events=overlay_events,
        window_sec=float(cfg.overlay_event_window_sec),
        tracks_csv_path=tracks_csv_path,
        jersey_by_track=jersey_by_track,
        track_id_remap=track_id_remap,
        progress_cb=progress_cb,
    )
    emit("overlay", 1, 1, "Overlay tamam")

    # Commentary: ask Qwen for commentator lines, then optionally TTS+mix onto the clean segment.
    try:
        if bool(getattr(cfg, "run_commentary", True)):
            action_events = [e for e in events if str(e.get("source")) == "action_spotting"]
            possession_events = [
                e
                for e in events
                if str(e.get("source")) == "tracking" and str(e.get("type")) in ("possession_start", "possession_change")
            ]

            if action_events:
                max_events = int(getattr(cfg, "commentary_max_events", 30) or 30)
                action_events = sorted(action_events, key=lambda e: float(e.get("t", 0.0)))[: max(1, max_events)]

                items_in: List[Dict[str, Any]] = []
                for ae in action_events:
                    try:
                        t = float(ae.get("t", 0.0))
                    except Exception:
                        continue

                    actor_tid: Optional[int] = None
                    if possession_events:
                        actor_tid = _assign_actor_track_id_to_action(
                            action_t=t,
                            possession_events=possession_events,
                            max_age_sec=float(getattr(cfg, "commentary_possession_max_age_sec", 8.0)),
                        )

                    actor_info = jersey_by_track.get(int(actor_tid), {}) if actor_tid is not None else {}
                    actor_jersey = None
                    actor_team = None
                    try:
                        if actor_info:
                            actor_jersey = actor_info.get("jersey_number")
                            actor_team = actor_info.get("team_id")
                    except Exception:
                        pass

                    items_in.append(
                        {
                            "t": float(t),
                            "timecode": _timecode_mmss(float(t)),
                            "event_label": str(ae.get("label", "")),
                            "event_confidence": float(ae.get("confidence", 0.0) or 0.0),
                            "description_tr": str(ae.get("description_tr", "") or ""),
                            "actor_track_id": int(actor_tid) if actor_tid is not None else None,
                            "actor_jersey_number": actor_jersey,
                            "actor_team_id": actor_team,
                        }
                    )

                commentary_input_path = str(Path(out_dir) / f"commentary_input_{run_id}.json")
                with open(commentary_input_path, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "schema_version": "1.0",
                            "run_id": run_id,
                            "created_utc": datetime.utcnow().isoformat() + "Z",
                            "items": items_in,
                        },
                        f,
                        ensure_ascii=False,
                        indent=2,
                    )

                qwen_url = str(getattr(cfg, "qwen_vl_url", "") or "").strip()
                qwen_model = str(getattr(cfg, "qwen_vl_model", "qwen3vl8b") or "qwen3vl8b").strip()
                prompt = _build_commentary_prompt(items_in)
                raw, err = _qwen_text_openai_compatible(base_url=qwen_url, model=qwen_model, prompt=prompt)

                parsed: List[Dict[str, Any]] = []
                if raw:
                    try:
                        extracted = _extract_json_array_best_effort(raw)
                        parsed = json.loads(extracted if extracted else raw)
                    except Exception:
                        parsed = []

                by_t: Dict[float, str] = {}
                for it in parsed or []:
                    try:
                        tt = float(it.get("t"))
                        txt = str(it.get("text") or "").strip()
                        if txt:
                            by_t[tt] = txt
                    except Exception:
                        continue

                items_out: List[Dict[str, Any]] = []
                for it in items_in:
                    tt = float(it.get("t", 0.0))
                    txt = by_t.get(tt)
                    if not txt:
                        # fallback: prefer action_spotting's Turkish description
                        txt = str(it.get("description_tr") or "").strip()
                    if not txt:
                        lbl = str(it.get("event_label") or "aksiyon")
                        a = it.get("actor_track_id")
                        txt = f"{lbl}!" if a is None else f"{lbl}! Topu en son kontrol eden oyuncu #{int(a)}."
                    items_out.append({**it, "text": txt, "commentary_text": txt})

                commentary_output_path = str(Path(out_dir) / f"commentary_qwen_{run_id}.json")
                with open(commentary_output_path, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "schema_version": "1.0",
                            "run_id": run_id,
                            "created_utc": datetime.utcnow().isoformat() + "Z",
                            "qwen_url": qwen_url,
                            "qwen_model": qwen_model,
                            "qwen_error": err,
                            "items": items_out,
                        },
                        f,
                        ensure_ascii=False,
                        indent=2,
                    )

                if bool(getattr(cfg, "commentary_enable_tts", True)):
                    tts_error: Optional[str] = None
                    audio_manifest: List[Dict[str, Any]] = []
                    clips: List[Tuple[float, str]] = []
                    try:
                        from commentary_engine import CommentaryEngine  # type: ignore

                        tts_backend = str(
                            getattr(cfg, "commentary_tts_backend", None)
                            or os.getenv("COMMENTARY_TTS_BACKEND")
                            or "xttsv2"
                        )
                        speaker_wav = getattr(cfg, "commentary_speaker_wav", None) or os.getenv("COMMENTARY_SPEAKER_WAV")
                        ce = CommentaryEngine(
                            output_dir=str(Path(out_dir) / f"commentary_{run_id}"),
                            enable_llm=False,
                            tts_backend=tts_backend,
                            speaker_wav=speaker_wav,
                        )

                        for it in items_out:
                            tt = float(it.get("t", 0.0))
                            txt = str(it.get("commentary_text") or "").strip()
                            if not txt:
                                audio_manifest.append({**it, "audio_path": None, "status": "skipped", "error": "empty_text"})
                                continue
                            r = ce.synthesize_commentary(text=txt, t_seconds=tt)
                            audio_manifest.append({**it, **r})
                            ap = r.get("audio_path")
                            if ap:
                                try:
                                    ap_s = str(ap)
                                    if os.path.isfile(ap_s) and os.path.getsize(ap_s) > 1024:
                                        clips.append((tt, ap_s))
                                except Exception:
                                    pass

                        commentary_audio_manifest_path = str(Path(out_dir) / f"commentary_audio_{run_id}.json")
                        with open(commentary_audio_manifest_path, "w", encoding="utf-8") as f:
                            json.dump(
                                {
                                    "schema_version": "1.0",
                                    "run_id": run_id,
                                    "created_utc": datetime.utcnow().isoformat() + "Z",
                                    "tts_backend": tts_backend,
                                    "items": audio_manifest,
                                },
                                f,
                                ensure_ascii=False,
                                indent=2,
                            )

                        if clips:
                            mixed_path = str(Path(out_dir) / f"product_{run_id}_commentary.mp4")
                            mixed = _mix_commentary_audio_into_video(
                                base_video_path=segment_path,
                                out_path=mixed_path,
                                clips=clips,
                            )
                            if mixed:
                                commentary_video_path = mixed
                                product_video_path = mixed
                    except Exception as e:
                        tts_error = str(e)

                    # If TTS failed hard, append error info into the qwen output JSON for quick debugging.
                    if tts_error and commentary_output_path:
                        try:
                            with open(commentary_output_path, "r", encoding="utf-8") as rf:
                                cur_out = json.load(rf) or {}
                            if isinstance(cur_out, dict):
                                cur_out["tts_error"] = tts_error
                                with open(commentary_output_path, "w", encoding="utf-8") as wf:
                                    json.dump(cur_out, wf, ensure_ascii=False, indent=2)
                        except Exception:
                            pass
    except Exception:
        pass

    events_json_path = str(Path(out_dir) / f"events_{run_id}.json")
    payload = {
        "schema_version": "1.0",
        "created_utc": datetime.utcnow().isoformat() + "Z",
        "source": {
            "input_video_path": str(Path(video_path).resolve()),
            "segment_video_path": segment_path,
        },
        "artifacts": {
            "tracking_video_path": tracking_video_path,
            "tracks_csv_path": tracks_csv_path,
            **({"tracks_csv_raw_path": tracks_csv_raw_path} if tracks_csv_raw_path else {}),
            **({"calibration_map_video_path": calibration_map_video_path} if calibration_map_video_path else {}),
            **({"calibration_events_json_path": calibration_events_json_path} if calibration_events_json_path else {}),
            **({"calibration_frames_jsonl_path": calibration_frames_jsonl_path} if calibration_frames_jsonl_path else {}),
            "overlay_video_path": final_overlay_path,
            "product_video_path": product_video_path,
            **({"commentary_input_path": commentary_input_path} if commentary_input_path else {}),
            **({"commentary_output_path": commentary_output_path} if commentary_output_path else {}),
            **({"commentary_audio_manifest_path": commentary_audio_manifest_path} if commentary_audio_manifest_path else {}),
            **({"commentary_video_path": commentary_video_path} if commentary_video_path else {}),
        },
        **({"track_id_remap": track_id_remap} if track_id_remap else {}),
        "jersey_numbers": list(jersey_by_track.values()) if jersey_by_track else [],
        "events": events,
    }
    with open(events_json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    emit("done", 1, 1, "Pipeline tamam")

    return {
        "run_id": run_id,
        "segment_path": segment_path,
        **({"calibration_map_video_path": calibration_map_video_path} if calibration_map_video_path else {}),
        **({"calibration_events_json_path": calibration_events_json_path} if calibration_events_json_path else {}),
        **({"calibration_frames_jsonl_path": calibration_frames_jsonl_path} if calibration_frames_jsonl_path else {}),
        "tracking_video_path": tracking_video_path,
        "tracks_csv_path": tracks_csv_path,
        "overlay_video_path": final_overlay_path,
        "product_video_path": product_video_path,
        **({"commentary_video_path": commentary_video_path} if commentary_video_path else {}),
        **({"commentary_input_path": commentary_input_path} if commentary_input_path else {}),
        **({"commentary_output_path": commentary_output_path} if commentary_output_path else {}),
        **({"commentary_audio_manifest_path": commentary_audio_manifest_path} if commentary_audio_manifest_path else {}),
        "events_json_path": events_json_path,
        "event_count": len(events),
    }
