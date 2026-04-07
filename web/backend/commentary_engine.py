"""Sports Commentary Engine.

Used by the backend pipeline to synthesize timestamped audio commentary clips.

Goals:
- Do not crash on import if optional deps are missing.
- Stable API for pipeline: `synthesize_commentary(text, t_seconds)`.
- Support lightweight/offline Windows TTS via `pyttsx3`.
- Optionally support Coqui XTTS v2 if installed.
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
from typing import Any, Dict, Optional

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

# Load environment variables from a .env file (if present)
if load_dotenv is not None:
    try:
        load_dotenv()
    except Exception:
        pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


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
        tts_backend: str = "pyttsx3",
        speaker_wav: Optional[str] = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.voice_name = str(voice_name or "")
        self.language_code = str(language_code or "tr-TR")
        self.model_name = str(model_name or "")
        self.enable_llm = bool(enable_llm)
        raw_backend = str(tts_backend or "pyttsx3").strip().lower()
        # Accept common aliases
        if raw_backend in ("xttsv2", "xtts_v2", "xtts-v2", "xtts2"):
            raw_backend = "xtts"
        if raw_backend in ("sapi5", "sapi"):
            raw_backend = "sapi"
        self.tts_backend = raw_backend or "pyttsx3"
        self.speaker_wav = speaker_wav

        self.llm_model = None
        self.tts = None
        self.tts_client = None

        if self.enable_llm:
            self._initialize_genai()
        self._initialize_tts()

        self.commentary_history: list[dict[str, Any]] = []

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

    def _initialize_tts(self) -> None:
        """Initialize TTS backend (best-effort)."""
        if self.tts_backend == "pyttsx3":
            try:
                import pyttsx3  # type: ignore

                self.tts_client = pyttsx3.init()
                logger.info("TTS initialized (pyttsx3)")
                return
            except Exception as e:
                logger.warning("Failed to init pyttsx3: %s", e)

        if self.tts_backend == "sapi":
            # We don't keep a long-lived COM object by default; we'll create per-call.
            try:
                import win32com.client  # type: ignore

                _ = win32com.client.Dispatch("SAPI.SpVoice")
                logger.info("TTS initialized (SAPI)")
                self.tts_client = "sapi"
                return
            except Exception as e:
                logger.warning("Failed to init SAPI: %s", e)

        if self.tts_backend in ("xtts", "coqui", "coqui_xtts"):
            if TTS is None or torch is None:
                logger.warning("XTTS requested but Coqui TTS/torch not available")
                return
            try:
                device = "cuda" if torch.cuda.is_available() else "cpu"  # type: ignore[union-attr]
                # PyTorch >=2.6 changed torch.load default to weights_only=True, which can
                # break Coqui TTS checkpoint loading unless required classes are allowlisted.
                # We'll best-effort allowlist classes mentioned by the error message and retry.
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

                logger.info("TTS initialized (XTTS v2) on %s", device)
                return
            except Exception as e:
                logger.warning("Failed to init XTTS v2: %s", e)

        logger.info("TTS disabled/unavailable")

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

    def _synthesize_audio_sapi(self, *, text: str, audio_path: Path) -> Optional[str]:
        """Windows-only: synthesize WAV via SAPI5 COM (more reliable than pyttsx3 save_to_file)."""
        try:
            import win32com.client  # type: ignore

            voice = win32com.client.Dispatch("SAPI.SpVoice")
            stream = win32com.client.Dispatch("SAPI.SpFileStream")
            # 3 == SSFMCreateForWrite
            stream.Open(str(audio_path), 3, False)
            voice.AudioOutputStream = stream
            voice.Speak(str(text))
            stream.Close()
        except Exception as e:
            logger.error("SAPI synthesis failed: %s", e)
            try:
                if audio_path.exists():
                    audio_path.unlink()
            except Exception:
                pass
            return None

        if self._validate_audio_file(audio_path):
            return str(audio_path)
        try:
            if audio_path.exists():
                audio_path.unlink()
        except Exception:
            pass
        return None

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

    def _synthesize_audio(self, *, text: str, timestamp: str) -> Optional[str]:
        safe_timestamp = str(timestamp or "").replace(":", "-").replace(" ", "_")
        clip_id = self._unique_clip_id(text=str(text), timestamp=safe_timestamp)
        audio_path = self.output_dir / f"commentary_{clip_id}.wav"

        if self.tts_backend == "pyttsx3" and self.tts_client is not None:
            try:
                self.tts_client.save_to_file(str(text), str(audio_path))
                self.tts_client.runAndWait()
                # pyttsx3 sometimes produces an empty WAV header on Windows.
                if self._validate_audio_file(audio_path):
                    return str(audio_path)

                # Best-effort fallback to SAPI if available.
                try:
                    if audio_path.exists():
                        audio_path.unlink()
                except Exception:
                    pass
                return self._synthesize_audio_sapi(text=str(text), audio_path=audio_path)
            except Exception as e:
                logger.error("pyttsx3 synthesis failed: %s", e)
                try:
                    if audio_path.exists():
                        audio_path.unlink()
                except Exception:
                    pass
                return self._synthesize_audio_sapi(text=str(text), audio_path=audio_path)

        if self.tts_backend == "sapi":
            return self._synthesize_audio_sapi(text=str(text), audio_path=audio_path)

        if self.tts_backend in ("xtts", "coqui", "coqui_xtts"):
            if self.tts is None:
                raise RuntimeError(
                    "XTTS v2 backend requested but not initialized. "
                    "Install Coqui TTS + torch, or choose tts_backend='sapi'."
                )

            current_dir = Path(__file__).parent
            speaker = self.speaker_wav or str(current_dir / "ertem_sener.wav")
            if not os.path.isfile(speaker):
                raise FileNotFoundError(
                    "XTTS v2 requires a reference speaker WAV. "
                    "Provide speaker_wav or set COMMENTARY_SPEAKER_WAV env var. "
                    f"Not found: {speaker}"
                )
            try:
                self.tts.tts_to_file(
                    text=str(text),
                    file_path=str(audio_path),
                    speaker_wav=str(speaker),
                    language="tr",
                )
            except Exception as e:
                logger.error("XTTS synthesis failed: %s", e)
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

        return None

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
    # Minimal smoke-test (no LLM)
    ce = CommentaryEngine(output_dir="_commentary_smoke", enable_llm=False, tts_backend="pyttsx3")
    r = ce.synthesize_commentary(text="Deneme spiker yorumu", t_seconds=1.23)
    print(r)
