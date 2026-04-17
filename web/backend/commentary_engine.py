"""Sports Commentary Engine.

Used by the backend pipeline to synthesize timestamped audio commentary clips.

Goals:
- Produce the most natural Turkish commentary voice possible with XTTS v2
  (Coqui) using a cloned voice of Ertem Şener.
- Apply 2025/2026 community best-practices for XTTS v2 voice cloning:
    * Use MULTIPLE short (8–12 s) clean reference clips instead of one long
      file — this gives a much steadier speaker latent than a single clip.
    * Auto-prepare the reference WAV (resample to 24 kHz mono, trim silence,
      RMS-normalize, slice into several voiced segments).
    * Cache GPT conditioning latents + speaker embeddings once at init.
    * Tune sampling (temperature≈0.70, top_p≈0.85, top_k=50,
      repetition_penalty=5) for a lively but artifact-free delivery.
    * Split commentary into short sentences (<=25 words) before inference —
      XTTS' internal splitter is less reliable, and short segments give
      noticeably better prosody on sports commentary.
    * Post-process every generated clip: DC-removal, soft loudness
      normalization, peak limiting, and short fade in/out to kill clicks.
- Stable API for the pipeline: `synthesize_commentary(text, t_seconds)`.
- Optional ChatterboxTTS (Resemble AI, 23 languages incl. Turkish) backend
  as a drop-in alternative if the user has it installed — it beats ElevenLabs
  in blind tests and often sounds even more natural than XTTS for short lines.
- Do not crash on import if optional deps are missing.
"""

from __future__ import annotations

import json
import logging
import os
import hashlib
import importlib
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from dotenv import load_dotenv  # type: ignore
except Exception:  # pragma: no cover
    load_dotenv = None

try:
    import google.generativeai as genai  # type: ignore
except Exception:  # pragma: no cover
    genai = None

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None

try:
    from TTS.api import TTS  # type: ignore
except Exception:  # pragma: no cover
    TTS = None

# Optional: ChatterboxTTS (Resemble AI). If installed, we can use it as a
# drop-in alternative backend that natively supports Turkish voice cloning.
try:  # pragma: no cover
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS  # type: ignore
except Exception:  # pragma: no cover
    ChatterboxMultilingualTTS = None  # type: ignore

# Load environment variables from a .env file (if present)
if load_dotenv is not None:
    try:
        load_dotenv()
    except Exception:
        pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Turkish text normalization
# ------------------------------------------------------------------ #

# Common abbreviations we expand before TTS so the model reads them naturally
# instead of as letters.
_TURKISH_ABBREVIATIONS: Dict[str, str] = {
    r"\bdk\.": "dakika",
    r"\bsn\.": "saniye",
    r"\bvs\.": "ve saire",
    r"\bvb\.": "ve benzeri",
    r"\börn\.": "örneğin",
    r"\btb\.": "takım",
    r"\bProf\.": "Profesör",
    r"\bDoç\.": "Doçent",
    r"\bDr\.": "Doktor",
    r"\bSn\.": "Sayın",
    r"\bNo\.": "numara",
    r"\bm\.": "metre",
    r"\bkm\.": "kilometre",
}

# Small mapping of digits <= 99 to Turkish words (used for minute/score
# readouts that otherwise sound robotic). Above 99 we leave to the model.
_TURK_ONES = ["sıfır", "bir", "iki", "üç", "dört", "beş", "altı", "yedi", "sekiz", "dokuz"]
_TURK_TENS = ["", "on", "yirmi", "otuz", "kırk", "elli", "altmış", "yetmiş", "seksen", "doksan"]


def _turk_int_to_words(n: int) -> str:
    if n < 0 or n > 99:
        return str(n)
    if n < 10:
        return _TURK_ONES[n]
    tens, ones = divmod(n, 10)
    if ones == 0:
        return _TURK_TENS[tens]
    return f"{_TURK_TENS[tens]} {_TURK_ONES[ones]}"


class CommentaryEngine:
    """Generate (optional) LLM text and synthesize audio for commentary lines."""

    def __init__(
        self,
        output_dir: str = "commentary_output",
        voice_name: str = "tr-TR-Wavenet-D",
        language_code: str = "tr-TR",
        model_name: str = "models/gemini-2.0-flash-lite",
        *,
        enable_llm: bool = False,
        tts_backend: str = "xttsv2",
        speaker_wav: Optional[str] = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.voice_name = str(voice_name or "")
        self.language_code = str(language_code or "tr-TR")
        self.model_name = str(model_name or "")
        self.enable_llm = bool(enable_llm)
        raw_backend = str(tts_backend or "xttsv2").strip().lower()
        # Accept common aliases
        if raw_backend in ("xttsv2", "xtts_v2", "xtts-v2", "xtts2"):
            raw_backend = "xtts"
        if raw_backend in ("chatterbox_multilingual", "chatterbox-multilingual"):
            raw_backend = "chatterbox"
        if raw_backend in ("sapi5", "sapi", "pyttsx3"):
            logger.warning("Unsupported TTS backend '%s'; forcing XTTS v2 with reference speaker WAV", raw_backend)
            raw_backend = "xtts"
        self.tts_backend = raw_backend or "xtts"

        current_dir = Path(__file__).resolve().parent
        default_speaker = current_dir / "ertem_sener.wav"
        self.speaker_wav = speaker_wav or str(default_speaker)

        # Directory where we cache the prepared (resampled, trimmed, voiced)
        # reference clips. Created on first use.
        self._refs_dir = Path(self.speaker_wav).with_suffix("").parent / (
            Path(self.speaker_wav).stem + "_refs"
        )
        self._prepared_ref_paths: List[str] = []

        self.llm_model = None
        self.tts = None
        self.chatterbox = None
        self.tts_client = None
        # Cached speaker conditioning (computed once from reference clips)
        self._gpt_cond_latent = None
        self._speaker_embedding = None

        # XTTS inference defaults — tuned for Turkish sports-commentary.
        # Lower temperature than defaults (0.70) keeps the voice stable and
        # avoids the "drunken" drift XTTS sometimes produces at 0.80+.
        # rep_penalty=5 strongly discourages stutters/loops, top_p=0.85
        # adds lively variation without hallucinations.
        self._xtts_params: Dict[str, Any] = {
            "temperature": 0.70,
            "length_penalty": 1.0,
            "repetition_penalty": 5.0,
            "top_k": 50,
            "top_p": 0.85,
            "speed": 1.02,  # tiny speed-up matches Ertem Şener's pace
            "do_sample": True,
            "enable_text_splitting": False,  # we split ourselves
        }

        if self.enable_llm:
            self._initialize_genai()
        self._initialize_tts()

        self.commentary_history: list[dict[str, Any]] = []

    # ------------------------------------------------------------------ #
    # LLM init (unchanged public behavior)
    # ------------------------------------------------------------------ #
    def _initialize_genai(self) -> None:
        if genai is None:
            raise ValueError(
                "google-generativeai is not installed/available but enable_llm=True. "
                "Either install it or set enable_llm=False."
            )

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")

        genai.configure(api_key=api_key)

        try:
            self.llm_model = genai.GenerativeModel(self.model_name)
            logger.info("Initialized Gemini model: %s", self.model_name)
        except Exception as e:
            logger.warning("Model init failed (%s). Falling back to first available model.", e)
            models = genai.list_models()
            names: list[str] = []
            for m in models:
                if isinstance(m, dict):
                    n = m.get("name")
                else:
                    n = getattr(m, "name", None)
                if n:
                    names.append(str(n))
            if not names:
                raise
            self.model_name = names[0]
            self.llm_model = genai.GenerativeModel(self.model_name)
            logger.info("Switched to available model: %s", self.model_name)

    # ------------------------------------------------------------------ #
    # TTS init
    # ------------------------------------------------------------------ #
    def _initialize_tts(self) -> None:
        """Initialize TTS backend (best-effort)."""
        if self.tts_backend == "chatterbox":
            if ChatterboxMultilingualTTS is not None and torch is not None:
                try:
                    device = "cuda" if torch.cuda.is_available() else "cpu"  # type: ignore[union-attr]
                    self.chatterbox = ChatterboxMultilingualTTS.from_pretrained(device=device)
                    logger.info("Chatterbox Multilingual TTS initialized on %s", device)
                    return
                except Exception as e:
                    logger.warning("Failed to init Chatterbox TTS (%s). Falling back to XTTS v2.", e)
                    self.chatterbox = None
                    self.tts_backend = "xtts"
            else:
                logger.info("Chatterbox not installed; using XTTS v2.")
                self.tts_backend = "xtts"

        if self.tts_backend in ("xtts", "coqui", "coqui_xtts"):
            if TTS is None or torch is None:
                logger.warning("XTTS requested but Coqui TTS/torch not available")
                return
            try:
                device = "cuda" if torch.cuda.is_available() else "cpu"  # type: ignore[union-attr]
                # PyTorch >=2.6 changed torch.load default to weights_only=True, which can
                # break Coqui TTS checkpoint loading unless required classes are allowlisted.
                # We best-effort allowlist classes mentioned by the error message and retry.
                def _allowlist_from_error(msg: str) -> bool:
                    try:
                        ser = getattr(torch, "serialization", None)
                        add_safe = getattr(ser, "add_safe_globals", None) if ser is not None else None
                        if add_safe is None:
                            return False

                        m = re.search(r"Unsupported global:\s*GLOBAL\s+([A-Za-z0-9_\.]+)", msg)
                        if not m:
                            return False
                        path = m.group(1)
                        if "." not in path:
                            return False
                        mod_name, obj_name = path.rsplit(".", 1)
                        mod = importlib.import_module(mod_name)
                        obj = getattr(mod, obj_name)
                        add_safe([obj])
                        logger.info("Allowlisted torch.load global for XTTS: %s", path)
                        return True
                    except Exception:
                        return False

                last_err: Optional[Exception] = None
                self.tts = None
                for _ in range(6):
                    try:
                        self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
                        break
                    except Exception as e:
                        last_err = e
                        if not _allowlist_from_error(str(e)):
                            raise

                if self.tts is None and last_err is not None:
                    raise last_err

                # Windows note: Recent torchaudio versions route `torchaudio.load()` through
                # TorchCodec by default. If TorchCodec cannot be loaded (common on Windows
                # due to FFmpeg/DLL issues), XTTS fails when loading the speaker reference wav.
                # We patch XTTS's `load_audio()` helper to use `soundfile` + `librosa` instead.
                try:
                    import numpy as np  # type: ignore
                    import soundfile as sf  # type: ignore

                    import TTS.tts.models.xtts as xtts_mod  # type: ignore

                    if not getattr(xtts_mod, "_FOMAC_SOUNDFile_LOAD_PATCHED", False):
                        orig_load_audio = getattr(xtts_mod, "load_audio", None)

                        def _load_audio_soundfile(audiopath, sampling_rate):  # type: ignore[no-untyped-def]
                            wav, sr = sf.read(str(audiopath), always_2d=False)
                            if getattr(wav, "ndim", 1) == 2:
                                wav = wav.mean(axis=1)
                            wav = wav.astype(np.float32, copy=False)
                            if int(sr) != int(sampling_rate):
                                import librosa  # type: ignore

                                wav = librosa.resample(wav, orig_sr=int(sr), target_sr=int(sampling_rate))
                            wav = np.clip(wav, -1.0, 1.0)
                            return torch.from_numpy(wav).unsqueeze(0)

                        if callable(orig_load_audio):
                            setattr(xtts_mod, "_FOMAC_ORIG_load_audio", orig_load_audio)
                        setattr(xtts_mod, "load_audio", _load_audio_soundfile)
                        setattr(xtts_mod, "_FOMAC_SOUNDFile_LOAD_PATCHED", True)
                        logger.info("Patched XTTS load_audio() to bypass torchaudio/torchcodec")
                except Exception as e:
                    logger.warning("XTTS audio-load patch skipped/failed: %s", e)

                # Prepare optimized speaker reference clips (once) and cache
                # conditioning. This is the single biggest quality lever for
                # XTTS voice cloning.
                try:
                    self._prepared_ref_paths = self._prepare_speaker_references()
                except Exception as e:
                    logger.warning("Speaker reference preparation failed: %s", e)
                    self._prepared_ref_paths = [self.speaker_wav]

                self._cache_speaker_conditioning()

                logger.info("TTS initialized (XTTS v2) on %s", device)
                return
            except Exception as e:
                logger.warning("Failed to init XTTS v2: %s", e)

        logger.info("TTS disabled/unavailable")

    # ------------------------------------------------------------------ #
    # Speaker reference preparation
    # ------------------------------------------------------------------ #
    def _prepare_speaker_references(self) -> List[str]:
        """Slice the raw reference WAV into several clean voiced segments.

        Strategy (per XTTS v2 community best-practice, 2025/2026):
        - Convert to mono 24 kHz float32.
        - Trim leading/trailing silence.
        - Use an RMS-based voice-activity detector to find continuous voiced
          regions and pick the N longest.
        - Each segment is normalized to -3 dBFS peak and written as
          `<speaker>_refs/seg_k.wav`.

        The prepared clips live right next to the original WAV so they are
        re-usable across runs. If preparation fails for any reason, we fall
        back to passing the raw WAV (the original behavior).
        """
        src = Path(self.speaker_wav)
        if not src.is_file():
            logger.warning("Speaker WAV not found for prep: %s", src)
            return []

        out_dir = self._refs_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        # If we already prepared them in a previous run, reuse.
        existing = sorted(out_dir.glob("seg_*.wav"))
        if existing:
            logger.info("Using %d cached reference segments from %s", len(existing), out_dir)
            return [str(p) for p in existing]

        try:
            import numpy as np  # type: ignore
            import soundfile as sf  # type: ignore
        except Exception:
            logger.warning("numpy/soundfile not available; using raw reference WAV")
            return [str(src)]

        try:
            wav, sr = sf.read(str(src), always_2d=False)
        except Exception as e:
            logger.warning("Could not read reference WAV: %s", e)
            return [str(src)]

        if wav.ndim == 2:
            wav = wav.mean(axis=1)
        wav = wav.astype(np.float32, copy=False)

        # Resample to 24 kHz (XTTS native)
        target_sr = 24000
        if int(sr) != target_sr:
            try:
                import librosa  # type: ignore
                wav = librosa.resample(wav, orig_sr=int(sr), target_sr=target_sr)
                sr = target_sr
            except Exception:
                logger.info("librosa resample unavailable; keeping sr=%d", sr)

        # Peak-normalize lightly so RMS thresholds are stable
        peak = float(np.max(np.abs(wav))) if wav.size else 1.0
        if peak > 1e-6:
            wav = wav * (0.95 / peak)

        # Frame-based RMS VAD
        frame_ms = 30
        frame = max(1, int(sr * frame_ms / 1000))
        if wav.size < frame * 4:
            logger.info("Reference too short for segmentation; using raw WAV")
            return [str(src)]

        n_frames = int(np.floor(wav.size / frame))
        trimmed = wav[: n_frames * frame]
        windows = trimmed.reshape(n_frames, frame)
        rms = np.sqrt(np.mean(windows * windows, axis=1) + 1e-12)

        # Adaptive threshold: anything above 35% of the median of the top
        # half of frames counts as voiced. This is robust to crowd noise.
        high_half = np.sort(rms)[n_frames // 2:]
        ref_rms = float(np.median(high_half)) if high_half.size else float(np.median(rms))
        thresh = max(0.01, 0.35 * ref_rms)
        voiced = rms > thresh

        # Collapse natural breath/inter-word gaps (<=400 ms) so spoken
        # segments stay contiguous. Anything longer than that is a real
        # pause between clauses and we want to cut there.
        gap_frames = int(400 / frame_ms)
        i = 0
        while i < len(voiced):
            if not voiced[i]:
                j = i
                while j < len(voiced) and not voiced[j]:
                    j += 1
                if (j - i) <= gap_frames and i > 0 and j < len(voiced):
                    voiced[i:j] = True
                i = j
            else:
                i += 1

        # Extract voiced spans, keep those >= 6 s, pick best N by length
        spans: List[Tuple[int, int]] = []
        i = 0
        min_len_frames = int(6000 / frame_ms)
        max_len_frames = int(12000 / frame_ms)
        while i < len(voiced):
            if voiced[i]:
                j = i
                while j < len(voiced) and voiced[j]:
                    j += 1
                if (j - i) >= min_len_frames:
                    # Cap at max length (centered)
                    if (j - i) > max_len_frames:
                        mid = (i + j) // 2
                        half = max_len_frames // 2
                        spans.append((mid - half, mid + half))
                    else:
                        spans.append((i, j))
                i = j
            else:
                i += 1

        if not spans:
            logger.info("No good voiced spans found; using raw reference WAV")
            return [str(src)]

        spans.sort(key=lambda s: s[1] - s[0], reverse=True)
        spans = spans[:4]  # up to 4 segments — plenty for stable conditioning
        spans.sort(key=lambda s: s[0])  # write in time order

        ref_paths: List[str] = []
        fade = int(sr * 0.015)  # 15ms fade
        for k, (a, b) in enumerate(spans):
            seg = trimmed[a * frame : b * frame].copy()
            # Fade in/out to avoid clicks
            if seg.size > 2 * fade:
                ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
                seg[:fade] *= ramp
                seg[-fade:] *= ramp[::-1]
            # Per-segment peak normalize to -3 dBFS
            pk = float(np.max(np.abs(seg))) or 1.0
            seg = seg * (0.707 / pk)
            out_path = out_dir / f"seg_{k:02d}.wav"
            sf.write(str(out_path), seg.astype(np.float32), sr, subtype="PCM_16")
            ref_paths.append(str(out_path))
            logger.info("Prepared ref segment: %s (%.2fs)", out_path.name, seg.size / sr)

        return ref_paths or [str(src)]

    def _cache_speaker_conditioning(self) -> None:
        """Pre-compute and cache speaker embeddings from reference clips."""
        refs: List[str] = list(self._prepared_ref_paths) or [str(self.speaker_wav)]
        refs = [r for r in refs if os.path.isfile(r)]
        if not refs:
            logger.warning("No usable reference audio for speaker conditioning")
            return

        xtts_model = self.tts.synthesizer.tts_model  # type: ignore[union-attr]
        try:
            self._gpt_cond_latent, self._speaker_embedding = (
                xtts_model.get_conditioning_latents(
                    audio_path=refs,
                    # Wider conditioning window with multiple short clips is
                    # the biggest known quality lever for XTTS cloning.
                    gpt_cond_len=30,
                    gpt_cond_chunk_len=6,
                    max_ref_length=30,
                    sound_norm_refs=True,
                )
            )
            logger.info("Speaker conditioning cached from %d reference(s)", len(refs))
        except Exception as e:
            logger.warning("Failed to cache speaker conditioning: %s", e)
            self._gpt_cond_latent = None
            self._speaker_embedding = None

    # ------------------------------------------------------------------ #
    # Text normalization / splitting
    # ------------------------------------------------------------------ #
    @staticmethod
    def _expand_turkish_numbers(text: str) -> str:
        """Replace bare 0–99 integers with Turkish words (keeps years/scores
        like '2026' or '3-1' alone — those are read correctly by XTTS)."""

        def repl(m: re.Match) -> str:
            try:
                n = int(m.group(0))
            except Exception:
                return m.group(0)
            if 0 <= n <= 99:
                return _turk_int_to_words(n)
            return m.group(0)

        # Only replace standalone 1–2 digit integers (avoid breaking scores,
        # years, or phone-style numbers).
        return re.sub(r"(?<![\w-])\d{1,2}(?![\w-])", repl, text)

    @staticmethod
    def _clean_text_for_tts(text: str) -> str:
        """Turkish text normalization for XTTS v2.

        - Expand common abbreviations (dk., vs., Dr., Sn., …).
        - Convert small integers to Turkish words for more natural prosody.
        - Collapse double punctuation ("..", "!!", "??") — double dots cause
          very long noisy output in XTTS.
        - Normalize whitespace around punctuation.
        - Nudge palatalized L on "gol" → "gôl" (XTTS otherwise uses kalın L).
        - Ensure there is terminal punctuation so XTTS has a clean end-of-
          sentence cue — for excited lines, prefer "!" over ".".
        """
        text = text.strip()
        if not text:
            return text

        # Expand abbreviations
        for pat, repl in _TURKISH_ABBREVIATIONS.items():
            text = re.sub(pat, repl, text, flags=re.IGNORECASE)

        # Small integers → words (more natural prosody on "3. dakika" etc.)
        text = CommentaryEngine._expand_turkish_numbers(text)

        # Double dots ("..") cause extremely long generation + noise in XTTS v2
        text = re.sub(r"\.{2,}", ".", text)
        text = re.sub(r"!{2,}", "!", text)
        text = re.sub(r"\?{2,}", "?", text)

        # Normalize whitespace around punctuation
        text = re.sub(r"\s+([,;:!?.])", r"\1", text)
        text = re.sub(r"([,;:])\s{2,}", r"\1 ", text)
        text = re.sub(r"\s{2,}", " ", text)

        # Turkish pronunciation nudge: "gol" → "gôl" (ince L).
        text = re.sub(r"\b([Gg])ol\b", r"\1ôl", text)
        text = re.sub(r"\b([Gg])oller\b", r"\1ôller", text)

        # Ensure final punctuation
        if not re.search(r"[.!?…]$", text):
            text = text + "."

        return text

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Split into short sentences for individual synthesis.

        XTTS v2 produces noticeably better prosody on short chunks.
        We:
        1. Split on sentence boundaries (. ! ? …).
        2. Further split any sentence longer than ~22 words on comma/semicolon
           boundaries, so each piece is well under the 400-token XTTS cap.
        """
        parts = re.split(r"(?<=[.!?…])\s+", text)
        sentences = [s.strip() for s in parts if s.strip()]
        if not sentences:
            return [text]

        MAX_WORDS = 22
        refined: list[str] = []
        for s in sentences:
            words = s.split()
            if len(words) <= MAX_WORDS:
                refined.append(s)
                continue
            # Long sentence: try to split on , ; — otherwise chunk by words.
            sub_parts = re.split(r"(?<=[,;])\s+", s)
            buf: list[str] = []
            count = 0
            for sp in sub_parts:
                w = sp.split()
                if count + len(w) > MAX_WORDS and buf:
                    refined.append(" ".join(buf).strip())
                    buf, count = [], 0
                buf.append(sp)
                count += len(w)
            if buf:
                refined.append(" ".join(buf).strip())

        return [r for r in refined if r]

    # ------------------------------------------------------------------ #
    # Misc helpers
    # ------------------------------------------------------------------ #
    def _unique_clip_id(self, *, text: str, timestamp: str) -> str:
        h = hashlib.sha1((str(timestamp) + "|" + str(text)).encode("utf-8", errors="ignore")).hexdigest()[:10]
        return f"{timestamp}_{h}"

    def _validate_audio_file(self, audio_path: Path) -> bool:
        try:
            if not audio_path.exists():
                return False
            size = int(audio_path.stat().st_size)
            # A valid WAV with actual audio should be much larger than a header.
            return size >= 2048
        except Exception:
            return False

    @staticmethod
    def _postprocess_wav(wav):  # type: ignore[no-untyped-def]
        """Simple per-clip audio polish.

        - Remove DC offset.
        - Apply soft RMS-based loudness targeting (-20 dBFS RMS).
        - Peak-limit to -0.3 dBFS to stop clipping.
        - 10ms linear fade in/out to kill edge clicks.
        """
        try:
            import numpy as np  # type: ignore
        except Exception:
            return wav

        try:
            arr = np.asarray(wav, dtype=np.float32)
            if arr.size == 0:
                return arr

            # DC removal
            arr = arr - float(np.mean(arr))

            # Loudness target
            rms = float(np.sqrt(np.mean(arr * arr) + 1e-12))
            target_rms = 10 ** (-20.0 / 20.0)  # -20 dBFS RMS
            if rms > 1e-6:
                gain = min(6.0, target_rms / rms)  # cap +15.6 dB gain
                arr = arr * gain

            # Peak-limit
            peak = float(np.max(np.abs(arr)))
            limit = 10 ** (-0.3 / 20.0)  # -0.3 dBFS
            if peak > limit:
                arr = arr * (limit / peak)

            # Fade in/out (10ms @ 24kHz ≈ 240 samples)
            fade = min(240, arr.size // 20)
            if fade > 0:
                ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
                arr[:fade] *= ramp
                arr[-fade:] *= ramp[::-1]

            return arr
        except Exception:
            return wav

    # ------------------------------------------------------------------ #
    # Prompt / LLM generation (unchanged)
    # ------------------------------------------------------------------ #
    def _create_commentary_prompt(self, match_data: Dict[str, Any]) -> str:
        team_a = match_data.get("team_a", "Team A")
        team_b = match_data.get("team_b", "Team B")
        active_player = match_data.get("active_player", "a player")
        action_type = match_data.get("action_type", "action")
        emotion = match_data.get("emotion", "excited")
        referee_side = match_data.get("referee_side", "")

        return (
            "You are an enthusiastic football commentator providing live match commentary.\n\n"
            "Match Context:\n"
            f"- Teams: {team_a} vs {team_b}\n"
            f"- Active Player: {active_player}\n"
            f"- Action: {action_type}\n"
            f"- Emotion Level: {emotion}\n"
            + (f"- Referee Decision: {referee_side}\n" if referee_side else "")
            + "\nGenerate a short, engaging commentary (max 2 sentences)."
        )

    def _generate_commentary_text(self, match_data: Dict[str, Any]) -> Optional[str]:
        if self.llm_model is None:
            return None
        try:
            prompt = self._create_commentary_prompt(match_data)
            response = self.llm_model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(  # type: ignore[attr-defined]
                    max_output_tokens=150,
                    temperature=0.9,
                ),
            )
            return str(getattr(response, "text", "") or "").strip() or None
        except Exception as e:
            logger.error("Error generating commentary text: %s", e)
            return None

    # ------------------------------------------------------------------ #
    # Synthesis
    # ------------------------------------------------------------------ #
    def _synthesize_xtts(self, sentences: List[str]):  # type: ignore[no-untyped-def]
        """Run XTTS v2 on each sentence and return concatenated float32 wav."""
        import numpy as np  # type: ignore

        xtts_model = self.tts.synthesizer.tts_model  # type: ignore[union-attr]

        # Use cached speaker conditioning, or compute on the fly from the
        # prepared reference segments (or the raw WAV as a last resort).
        if self._gpt_cond_latent is not None and self._speaker_embedding is not None:
            gpt_cond_latent = self._gpt_cond_latent
            speaker_embedding = self._speaker_embedding
        else:
            refs = list(self._prepared_ref_paths) or [str(self.speaker_wav)]
            refs = [r for r in refs if os.path.isfile(r)]
            if not refs:
                raise FileNotFoundError(
                    "XTTS v2 requires a reference speaker WAV. "
                    f"Not found: {self.speaker_wav}"
                )
            gpt_cond_latent, speaker_embedding = xtts_model.get_conditioning_latents(
                audio_path=refs,
                gpt_cond_len=30,
                gpt_cond_chunk_len=6,
                max_ref_length=30,
                sound_norm_refs=True,
            )

        all_wav_chunks: list = []
        for sentence in sentences:
            if not sentence.strip():
                continue
            out = xtts_model.inference(
                text=sentence,
                language="tr",
                gpt_cond_latent=gpt_cond_latent,
                speaker_embedding=speaker_embedding,
                **self._xtts_params,
            )
            wav_chunk = out["wav"]
            if not isinstance(wav_chunk, np.ndarray):
                wav_chunk = np.array(wav_chunk, dtype=np.float32)
            all_wav_chunks.append(wav_chunk.astype(np.float32, copy=False))

        if not all_wav_chunks:
            return None

        # ~120 ms natural pause between sentences sounds more like a real
        # commentator than XTTS' default sub-50ms cutoff.
        silence = np.zeros(int(24000 * 0.12), dtype=np.float32)
        combined: list = []
        for i, chunk in enumerate(all_wav_chunks):
            combined.append(chunk)
            if i < len(all_wav_chunks) - 1:
                combined.append(silence)
        return np.concatenate(combined)

    def _synthesize_chatterbox(self, text: str):  # type: ignore[no-untyped-def]
        """Run Chatterbox Multilingual on the full text (it handles splitting
        internally and supports Turkish)."""
        speaker = str(self.speaker_wav)
        if not os.path.isfile(speaker):
            raise FileNotFoundError(
                f"Chatterbox requires a reference speaker WAV: {speaker}"
            )
        # Chatterbox returns a torch tensor at 24 kHz.
        audio = self.chatterbox.generate(  # type: ignore[union-attr]
            text,
            language_id="tr",
            audio_prompt_path=speaker,
            exaggeration=0.55,
            cfg_weight=0.55,
        )
        try:
            import numpy as np  # type: ignore
            if hasattr(audio, "detach"):
                audio = audio.detach().cpu().numpy().astype("float32")
            if audio.ndim == 2:
                audio = audio.squeeze()
            return audio
        except Exception:
            return audio

    def _synthesize_audio(self, *, text: str, timestamp: str) -> Optional[str]:
        safe_timestamp = str(timestamp or "").replace(":", "-").replace(" ", "_")
        clip_id = self._unique_clip_id(text=str(text), timestamp=safe_timestamp)
        audio_path = self.output_dir / f"commentary_{clip_id}.wav"

        cleaned = self._clean_text_for_tts(str(text))
        if not cleaned:
            return None
        sentences = self._split_sentences(cleaned)

        try:
            import numpy as np  # type: ignore
            import soundfile as sf  # type: ignore
        except Exception as e:
            logger.error("soundfile/numpy not available: %s", e)
            raise

        try:
            if self.tts_backend == "chatterbox" and self.chatterbox is not None:
                # Chatterbox handles multi-sentence input well; pass the whole
                # cleaned string and let it do its thing.
                full_wav = self._synthesize_chatterbox(" ".join(sentences))
            else:
                if self.tts is None:
                    raise RuntimeError(
                        "XTTS v2 backend requested but not initialized. "
                        "Install Coqui TTS + torch to synthesize with ertem_sener.wav."
                    )
                full_wav = self._synthesize_xtts(sentences)

            if full_wav is None or getattr(full_wav, "size", 0) == 0:
                return None

            # Post-process for consistent loudness + click-free edges
            full_wav = self._postprocess_wav(full_wav)
            sf.write(str(audio_path), full_wav.astype(np.float32, copy=False), 24000)

        except Exception as e:
            logger.error("TTS synthesis failed: %s", e)
            try:
                if audio_path.exists():
                    audio_path.unlink()
            except Exception:
                pass
            raise

        if self._validate_audio_file(audio_path):
            return str(audio_path)
        try:
            if audio_path.exists():
                audio_path.unlink()
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def synthesize_commentary(self, *, text: str, t_seconds: float) -> Dict[str, Any]:
        ts = f"{float(t_seconds):09.3f}s"
        err: Optional[str] = None
        t0 = time.time()
        audio_path = None
        try:
            audio_path = self._synthesize_audio(text=str(text), timestamp=ts)
        except Exception as e:
            err = str(e)
            audio_path = None
        dt_ms = int(round((time.time() - t0) * 1000.0))
        return {
            "t": float(t_seconds),
            "text": str(text),
            "audio_path": audio_path,
            "status": "success" if audio_path else "error",
            "error": err,
            "synth_ms": dt_ms,
        }

    def process_match_action(self, match_data: Dict[str, Any]) -> Dict[str, Any]:
        timestamp = match_data.get("timestamp", datetime.now().isoformat())
        commentary_text = self._generate_commentary_text(match_data)
        if not commentary_text:
            return {
                "text": None,
                "audio_path": None,
                "timestamp": timestamp,
                "status": "error",
                "error": "Failed to generate commentary text",
            }

        audio_path = self._synthesize_audio(text=commentary_text, timestamp=str(timestamp))
        result = {
            "text": commentary_text,
            "audio_path": audio_path,
            "timestamp": timestamp,
            "status": "success" if audio_path else "partial_success",
            "match_data": match_data,
        }
        self.commentary_history.append(result)
        self._save_metadata(result)
        return result

    def _save_metadata(self, result: Dict[str, Any]) -> None:
        try:
            metadata_file = self.output_dir / "commentary_metadata.json"
            if metadata_file.exists():
                metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
            else:
                metadata = []
            metadata.append(result)
            metadata_file.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            logger.error("Error saving metadata: %s", e)

    def get_all_commentary(self) -> list[dict[str, Any]]:
        return list(self.commentary_history)

    def clear_history(self) -> None:
        self.commentary_history = []


if __name__ == "__main__":
    import sys

    output_dir = "_commentary_test_output"
    print(f"=== FoMAC Commentary Engine - TTS Test ===")
    print(f"Output directory: {output_dir}\n")

    ce = CommentaryEngine(output_dir=output_dir, enable_llm=False, tts_backend="xttsv2")

    # Test commentary lines with varying excitement levels
    test_lines = [
        (0.0,  "Ve maç başlıyor! İki takım da sahada yerini aldı."),
        (5.0,  "Topu kaptı, hızla ileri çıkıyor, rakiplerini tek tek geçiyor!"),
        (12.5, "Ceza sahasına giriyor, şut çekiyor... Gol! Muhteşem bir gol! Tribünler ayağa kalktı!"),
        (20.0, "Hakem ofsayt bayrağını kaldırdı, gol geçersiz! Tartışmalı bir karar."),
        (35.0, "Son dakika, penaltı! İnanılmaz bir an, kaleci hazırlanıyor!"),
        (42.0, "Vuruyor ve gol! Maçın skorunu değiştiren harika bir penaltı!"),
        (60.0, "Tehlikeli bir atak daha, kurtarış! Kaleci bugün formda!"),
        (88.0, "Uzatma dakikalarındayız, son bir şans! Orta geliyor, kafa vuruşu ve gol! İnanılmaz!"),
    ]

    results = []
    for t, text in test_lines:
        print(f"[{t:06.1f}s] Synthesizing: {text[:70]}...")
        r = ce.synthesize_commentary(text=text, t_seconds=t)
        results.append(r)
        status_icon = "OK" if r["status"] == "success" else "FAIL"
        print(f"          [{status_icon}] {r.get('synth_ms', '?')}ms -> {r.get('audio_path', 'N/A')}\n")

    success = sum(1 for r in results if r["status"] == "success")
    print(f"=== Results: {success}/{len(results)} clips generated ===")

    if success > 0:
        print(f"\nAudio files saved to: {Path(output_dir).resolve()}")
        print("Play them to hear the excited commentator voice!")

    sys.exit(0 if success == len(results) else 1)
