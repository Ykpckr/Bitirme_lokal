
from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import os
import cv2
import base64
import time
import numpy as np
from typing import Optional, List, Dict, Any
import logging
import json
from contextlib import asynccontextmanager

from fastapi.responses import FileResponse
import mimetypes
import subprocess
import hashlib
import shutil
from fastapi.responses import StreamingResponse, Response
from datetime import datetime
from urllib.parse import quote as requests_quote
import random
import threading
import asyncio

def _ffmpeg_exe() -> Optional[str]:
    """Return a usable ffmpeg executable path.

    Tries system PATH first, then `imageio-ffmpeg` (which can ship/download ffmpeg
    on Windows without a global install).
    """
    p = shutil.which("ffmpeg")
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

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)


# --- Pipeline progress jobs (in-memory) ---
_jobs_lock = threading.Lock()
_jobs: Dict[str, Dict[str, Any]] = {}


def _new_job_id() -> str:
    return datetime.utcnow().strftime("%Y%m%d%H%M%S") + f"_{random.getrandbits(32):08x}"


def _set_job(job_id: str, patch: Dict[str, Any]) -> None:
    with _jobs_lock:
        cur = _jobs.get(job_id) or {}
        cur.update(patch)
        _jobs[job_id] = cur


def _get_job(job_id: str) -> Optional[Dict[str, Any]]:
    with _jobs_lock:
        j = _jobs.get(job_id)
        return dict(j) if isinstance(j, dict) else None

# --- Configuration ---
BACKEND_DIR = os.path.abspath(os.path.dirname(__file__))
REPO_ROOT = os.path.abspath(os.path.join(BACKEND_DIR, "..", ".."))

# Allow overriding via env var; default to repo-local yolo weights if present.
_default_yolo = os.path.join(REPO_ROOT, "yolo11n.pt")
YOLO_MODEL_PATH = os.getenv("FOMAC_YOLO_MODEL_PATH", _default_yolo)

# This dictionary will hold the loaded model
ml_models = {}

# --- Lifespan Management (for loading models on startup) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the YOLO model
    try:
        from ultralytics import YOLO
        if os.path.exists(YOLO_MODEL_PATH):
            logger.info(f"Loading YOLO model from {YOLO_MODEL_PATH}")
            ml_models["yolo"] = YOLO(YOLO_MODEL_PATH)
            logger.info("YOLO model loaded successfully")
        else:
            logger.warning(f"YOLO model not found at {YOLO_MODEL_PATH}. Inference will not be available.")
            ml_models["yolo"] = None
    except ImportError:
        logger.warning("ultralytics not installed. YOLO inference will be unavailable.")
        ml_models["yolo"] = None
    except Exception as e:
        logger.error(f"Error loading YOLO model: {e}")
        ml_models["yolo"] = None
    
    yield
    
    # Clean up the ML models and release the resources
    ml_models.clear()
    logger.info("ML models cleaned up.")


app = FastAPI(title="FOMAC Backend", version="1.1.0", lifespan=lifespan)

# --- CORS Configuration ---
app.add_middleware(
    CORSMiddleware,
    # include both 3000 and 3001 (Next.js may pick the next free port)
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
    ],
    # Also allow any localhost/127.0.0.1 port (dev environments often shift ports).
    allow_origin_regex=r"^http://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    # Helpful for players/debug UIs that inspect Range-related headers.
    expose_headers=["Accept-Ranges", "Content-Range", "Content-Length", "Content-Type"],
)


# --- Pydantic Models for Request/Response ---
class FrameRequest(BaseModel):
    frame_id: float
    frame_data: str # base64 encoded image string

class Detection(BaseModel):
    x: float
    y: float
    width: float
    height: float
    confidence: float
    class_name: str

class FrameResponse(BaseModel):
    frame_id: float
    inference_time: float
    detections: List[Detection]


# --- API Endpoints ---

# --- Debug Video Generation ---
from typing import Dict
class ProcessVideoRequest(BaseModel):
    path: str
    minutes: Optional[float] = None
    start_seconds: Optional[float] = 0.0

@app.post("/api/process_video")
async def process_video(req: ProcessVideoRequest):
    """
    For each request, generate two videos:
    - Normal video (just extract/copy the segment)
    - YOLO-debug video (bounding boxes drawn)
    Returns URLs for both.
    """
    requested_path = os.path.abspath(req.path)
    if not _is_allowed_media_path(requested_path):
        raise HTTPException(status_code=404, detail="Video not found or access denied")

    duration = None
    if req.minutes is not None:
        duration = float(req.minutes) * 60.0
    start = float(req.start_seconds or 0.0)

    base_name = os.path.splitext(os.path.basename(requested_path))[0]
    # Generate a 32-bit unique id for this processed video
    video_id = random.getrandbits(32)
    id_hex = f"{video_id:08x}"
    normal_name = f"{base_name}_{id_hex}_normal.mp4"
    debug_name = f"{base_name}_{id_hex}_debug.mp4"
    normal_path = os.path.join(UPLOADS_DIR, normal_name)
    debug_path = os.path.join(UPLOADS_DIR, debug_name)

    # --- Normal video: extract/copy segment ---
    ffmpeg_bin = _ffmpeg_exe()
    src_fps = None
    src_width = None
    src_height = None
    if ffmpeg_bin:
        cmd = [
            ffmpeg_bin,
            "-y",
            "-ss", str(start),
            "-i", requested_path,
            "-t", str(duration) if duration else "",
            "-c:v", "libx264",
            "-preset", "fast",
            "-c:a", "aac",
            "-movflags", "+faststart",
            normal_path,
        ]
        # Remove empty args
        cmd = [c for c in cmd if c != ""]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"ffmpeg failed for normal video: {e.stderr}")
            raise HTTPException(status_code=500, detail=f"ffmpeg failed for normal video: {e.stderr}")
    else:
        # Fallback: OpenCV extraction
        cap = cv2.VideoCapture(requested_path)
        if not cap.isOpened():
            raise HTTPException(status_code=500, detail="Failed to open video for extraction")
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        src_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
        src_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
        start_frame = int(start * src_fps)
        total_frames = int(duration * src_fps) if duration else int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(normal_path, fourcc, src_fps, (src_width, src_height))
        if not writer.isOpened():
            cap.release()
            raise HTTPException(status_code=500, detail="Failed to create video writer for extraction")
        written = 0
        for _ in range(total_frames):
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            if frame.shape[1] != src_width or frame.shape[0] != src_height:
                frame = cv2.resize(frame, (src_width, src_height))
            writer.write(frame)
            written += 1
        writer.release()
        cap.release()
        if written == 0:
            try:
                os.remove(normal_path)
            except Exception:
                pass
            raise HTTPException(status_code=500, detail="No frames written during extraction")

    # We will not create a separate debug video file here. The frontend treats
    # debug vs normal: create a separate debug video with YOLO boxes if model available.
    # Ensure the normal video exists and is non-empty, then optionally produce debug outputs.
    if not os.path.isfile(normal_path) or os.path.getsize(normal_path) == 0:
        logger.error(f"Normal video missing after processing: {normal_path}")
        raise HTTPException(status_code=500, detail="Normal video not produced")

    # Prepare debug outputs
    debug_json_path = os.path.join(UPLOADS_DIR, f"{base_name}_{id_hex}_debug.json")
    try:
        yolo_model = ml_models.get("yolo")
    except Exception:
        yolo_model = None

    if yolo_model is not None:
        try:
            cap = cv2.VideoCapture(normal_path)
            if not cap.isOpened():
                raise Exception("Failed to open normal video for debug generation")

            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)

            temp_debug = normal_path + ".tmp_debug.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(temp_debug, fourcc, fps, (w, h))
            frame_idx = 0
            detections_list = []
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                # run inference on RGB frame
                try:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                except Exception:
                    frame_rgb = frame
                try:
                    results = yolo_model(frame_rgb, verbose=False, imgsz=640, conf=0.25)
                except TypeError:
                    results = yolo_model.predict(frame_rgb, imgsz=640, conf=0.25)

                img_h, img_w = frame.shape[:2]
                frame_dets = []
                inference_time_ms = 0.0
                if results and results[0].boxes is not None:
                    boxes = results[0].boxes
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        class_name = yolo_model.names.get(class_id, "unknown").lower()
                        if class_name in ["player", "ball", "referee", "person"]:
                            det = {
                                "x": float(x1 / img_w),
                                "y": float(y1 / img_h),
                                "width": float((x2 - x1) / img_w),
                                "height": float((y2 - y1) / img_h),
                                "confidence": float(box.conf[0].cpu().numpy()),
                                "class_name": "player" if class_name == "person" else class_name,
                            }
                            frame_dets.append(det)
                            # draw on frame (BGR)
                            x = int(det["x"] * img_w)
                            y = int(det["y"] * img_h)
                            w_box = int(det["width"] * img_w)
                            h_box = int(det["height"] * img_h)
                            cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
                            label = f"{det['class_name']} {int(det['confidence']*100)}%"
                            cv2.putText(frame, label, (x, max(10, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)

                # timestamp in seconds
                timestamp = frame_idx / fps
                detections_list.append({"frame_id": timestamp, "inference_time": inference_time_ms, "detections": frame_dets, "received_time": time.time()})
                writer.write(frame)
                frame_idx += 1

            writer.release()
            cap.release()

            # Re-encode temp_debug to ensure libx264 compatibility if ffmpeg exists
            if _ffmpeg_exe():
                ffmpeg_bin = _ffmpeg_exe()
                cmd = [
                    ffmpeg_bin,
                    "-y",
                    "-i",
                    temp_debug,
                    "-c:v",
                    "libx264",
                    "-preset",
                    "fast",
                    "-c:a",
                    "aac",
                    "-movflags",
                    "+faststart",
                    debug_path,
                ]
                try:
                    subprocess.run(cmd, capture_output=True, text=True, check=True)
                except subprocess.CalledProcessError as e:
                    logger.error(f"ffmpeg re-encode failed for debug video: {e.stderr}")
                    # Fallback: move temp_debug to debug_path
                    try:
                        shutil.move(temp_debug, debug_path)
                    except Exception:
                        pass
                finally:
                    try:
                        if os.path.exists(temp_debug):
                            os.remove(temp_debug)
                    except Exception:
                        pass
            else:
                try:
                    shutil.move(temp_debug, debug_path)
                except Exception:
                    pass

            # write detections JSON
            try:
                with open(debug_json_path, "w") as jf:
                    json.dump(detections_list, jf)
            except Exception as e:
                logger.error(f"Failed to write debug json: {e}")

        except Exception as e:
            logger.error(f"Failed to create debug video: {e}")
            # If debug generation fails, fall back to using normal as debug
            debug_path = normal_path
            debug_json_path = ""
    else:
        # No model available, point debug to normal
        debug_path = normal_path
        debug_json_path = ""

    debug_json_url = f"http://localhost:8000/api/file?path={requests_quote(debug_json_path)}" if debug_json_path else ""

    return {
        "video_id": video_id,
        "normal_path": normal_path,
        "normal_url": f"http://localhost:8000/api/video?path={requests_quote(normal_path)}",
        "debug_path": debug_path,
        "debug_url": f"http://localhost:8000/api/video?path={requests_quote(debug_path)}",
        "debug_json_path": debug_json_path,
        "debug_json_url": debug_json_url,
    }
@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "ok", "service": "FOMAC Backend"}

@app.get("/health")
async def health():
    """Health check with model status"""
    model_loaded = ml_models.get("yolo") is not None
    return {
        "status": "ok",
        "model_loaded": model_loaded,
        "model_path": YOLO_MODEL_PATH if model_loaded else None,
    }


@app.get("/api/model_info")
async def model_info():
    yolo = ml_models.get("yolo")
    if yolo is None:
        raise HTTPException(status_code=404, detail="Model not loaded")
    try:
        names = getattr(yolo, 'names', None)
        return {"names": names}
    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# --- Video Discovery ---
SUPPORTED_VIDEO_FORMATS = {".mp4", ".avi", ".mov", ".mkv"}
UPLOADS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "uploads"))

# Ensure uploads directory exists and is writable
os.makedirs(UPLOADS_DIR, exist_ok=True)

# Use env var if set; otherwise default to uploads so endpoints work on Windows.
VIDEO_DIR = os.getenv("FOMAC_VIDEO_DIR", UPLOADS_DIR)


def _is_allowed_media_path(path: str) -> bool:
    try:
        requested_path = os.path.normcase(os.path.realpath(os.path.abspath(path)))
        if not os.path.isfile(requested_path):
            return False

        roots = []
        if VIDEO_DIR:
            roots.append(os.path.normcase(os.path.realpath(os.path.abspath(VIDEO_DIR))))
        roots.append(os.path.normcase(os.path.realpath(os.path.abspath(UPLOADS_DIR))))

        for root in roots:
            try:
                if os.path.commonpath([requested_path, root]) == root:
                    return True
            except Exception:
                # e.g., different drives
                continue
        return False
    except Exception:
        return False


def _extract_segment_to_uploads(src_path: str, start_sec: float, duration_sec: Optional[float]) -> str:
    os.makedirs(UPLOADS_DIR, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(src_path))[0]
    stamp = datetime.utcnow().strftime('%Y%m%d%H%M%S')
    out_name = f"{base_name}_seg_{stamp}.mp4"
    out_path = os.path.join(UPLOADS_DIR, out_name)

    ffmpeg_bin = _ffmpeg_exe()
    if ffmpeg_bin:
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
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"ffmpeg failed: {e.stderr}")
            raise HTTPException(status_code=500, detail=f"ffmpeg failed: {e.stderr}")
    else:
        cap = cv2.VideoCapture(src_path)
        if not cap.isOpened():
            raise HTTPException(status_code=500, detail="Failed to open video for extraction")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
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
            raise HTTPException(status_code=500, detail="Failed to create video writer for extraction")

        written = 0
        for _ in range(frames_to_write):
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            if frame.shape[1] != w or frame.shape[0] != h:
                frame = cv2.resize(frame, (w, h))
            writer.write(frame)
            written += 1

        writer.release()
        cap.release()
        if written == 0:
            try:
                os.remove(out_path)
            except Exception:
                pass
            raise HTTPException(status_code=500, detail="No frames written during extraction")

    if not os.path.isfile(out_path) or os.path.getsize(out_path) == 0:
        raise HTTPException(status_code=500, detail="Extraction failed to produce output file")

    return out_path


class RunFullPipelineRequest(BaseModel):
    path: str
    minutes: Optional[float] = None
    start_seconds: Optional[float] = 0.0

    run_action_spotting: bool = True

    # action spotting inputs
    features_path: Optional[str] = None
    checkpoint_path: Optional[str] = None
    action_threshold: float = 0.50
    action_nms_window_sec: float = 10.0

    # tracking inputs
    run_tracking: bool = True
    tracking_device: Optional[str] = None
    tracking_config_path: Optional[str] = None
    detector_weights: Optional[str] = None
    reid_weights: Optional[str] = None

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
    jersey_max_samples_per_track: int = 30
    jersey_min_det_conf: float = 0.55
    jersey_min_box_area: int = 1600
    jersey_min_frame_gap: int = 10
    jersey_frame_topk: int = 20
    jersey_crops_dir: Optional[str] = None
    jersey_vis_filter: bool = True
    jersey_vis_min_score: float = 0.12
    jersey_in_tracking: bool = True
    jersey_merge_same_number: bool = True
    jersey_merge_min_confidence: float = 0.60
    jersey_merge_max_overlap_frames: int = 5

    # heuristics
    possession_dist_norm: float = 0.08
    possession_stable_frames: int = 6
    possession_stride_frames: int = 5
    ball_cls_id: int = 1
    player_cls_id: int = 0


@app.post("/api/run_full_pipeline")
async def run_full_pipeline(req: RunFullPipelineRequest):
    requested_path = os.path.abspath(req.path)
    if not _is_allowed_media_path(requested_path):
        raise HTTPException(status_code=404, detail="Video not found or access denied")

    duration = float(req.minutes) * 60.0 if req.minutes is not None else None
    start = float(req.start_seconds or 0.0)
    segment_path = _extract_segment_to_uploads(requested_path, start, duration)

    from pipeline import FullPipelineConfig, run_full_pipeline as _run

    cfg = FullPipelineConfig(
        start_seconds=0.0,
        duration_seconds=None,
        run_tracking=bool(req.run_tracking),
        tracking_device=req.tracking_device,
        tracking_config_path=req.tracking_config_path,
        detector_weights=req.detector_weights,
        reid_weights=req.reid_weights,
        run_action_spotting=bool(req.run_action_spotting),
        features_path=req.features_path,
        checkpoint_path=req.checkpoint_path,
        action_threshold=float(req.action_threshold),
        action_nms_window_sec=float(req.action_nms_window_sec),
        run_jersey_number_recognition=bool(req.run_jersey_number_recognition),
        qwen_vl_url=str(req.qwen_vl_url),
        qwen_vl_model=str(req.qwen_vl_model),
        jersey_prompt=str(req.jersey_prompt),
        jersey_max_tracks=int(req.jersey_max_tracks),
        jersey_max_samples_per_track=int(req.jersey_max_samples_per_track),
        jersey_min_det_conf=float(req.jersey_min_det_conf),
        jersey_min_box_area=int(req.jersey_min_box_area),
        jersey_min_frame_gap=int(req.jersey_min_frame_gap),
        jersey_frame_topk=int(req.jersey_frame_topk),
        jersey_crops_dir=req.jersey_crops_dir,
        jersey_vis_filter=bool(req.jersey_vis_filter),
        jersey_vis_min_score=float(req.jersey_vis_min_score),
        jersey_in_tracking=bool(req.jersey_in_tracking),
        jersey_merge_same_number=bool(req.jersey_merge_same_number),
        jersey_merge_min_confidence=float(req.jersey_merge_min_confidence),
        jersey_merge_max_overlap_frames=int(req.jersey_merge_max_overlap_frames),
        possession_dist_norm=float(req.possession_dist_norm),
        possession_stable_frames=int(req.possession_stable_frames),
        possession_stride_frames=int(req.possession_stride_frames),
        ball_cls_id=int(req.ball_cls_id),
        player_cls_id=int(req.player_cls_id),
    )

    try:
        result = _run(video_path=segment_path, out_dir=UPLOADS_DIR, cfg=cfg)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Full pipeline failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Full pipeline failed: {str(e)}")

    return {
        **result,
        "segment_url": f"http://localhost:8000/api/video?path={requests_quote(result['segment_path'])}",
        "overlay_video_url": f"http://localhost:8000/api/video?path={requests_quote(result['overlay_video_path'])}",
        "events_json_url": f"http://localhost:8000/api/file?path={requests_quote(result['events_json_path'])}",
        **(
            {"tracking_video_url": f"http://localhost:8000/api/video?path={requests_quote(result['tracking_video_path'])}"}
            if result.get("tracking_video_path")
            else {}
        ),
        **(
            {"tracks_csv_url": f"http://localhost:8000/api/file?path={requests_quote(result['tracks_csv_path'])}"}
            if result.get("tracks_csv_path")
            else {}
        ),
    }


@app.post("/api/run_full_pipeline_async")
async def run_full_pipeline_async(req: RunFullPipelineRequest):
    """Starts the full pipeline in a background thread and returns a job_id.

    Progress can be tracked via /api/pipeline_progress?job_id=...
    """
    requested_path = os.path.abspath(req.path)
    if not _is_allowed_media_path(requested_path):
        raise HTTPException(status_code=404, detail="Video not found or access denied")

    job_id = _new_job_id()
    _set_job(
        job_id,
        {
            "status": "running",
            "stage": "queued",
            "current": 0,
            "total": 1,
            "message": "Kuyruğa alındı",
            "result": None,
            "error": None,
            "updated_utc": datetime.utcnow().isoformat() + "Z",
        },
    )

    def worker() -> None:
        try:
            from pipeline import FullPipelineConfig, run_full_pipeline as _run

            def progress_cb(stage: str, current: int, total: int, message: str) -> None:
                _set_job(
                    job_id,
                    {
                        "status": "running",
                        "stage": stage,
                        "current": int(current),
                        "total": int(total),
                        "message": str(message),
                        "updated_utc": datetime.utcnow().isoformat() + "Z",
                    },
                )

            # Make sure the UI leaves the initial "queued" state quickly
            progress_cb("segment", 0, 1, "Segment çıkarılıyor")

            duration = float(req.minutes) * 60.0 if req.minutes is not None else None
            start = float(req.start_seconds or 0.0)

            cfg = FullPipelineConfig(
                start_seconds=float(start),
                duration_seconds=float(duration) if duration is not None else None,
                run_tracking=bool(req.run_tracking),
                tracking_device=req.tracking_device,
                tracking_config_path=req.tracking_config_path,
                detector_weights=req.detector_weights,
                reid_weights=req.reid_weights,
                run_action_spotting=bool(req.run_action_spotting),
                features_path=req.features_path,
                checkpoint_path=req.checkpoint_path,
                action_threshold=float(req.action_threshold),
                action_nms_window_sec=float(req.action_nms_window_sec),
                run_jersey_number_recognition=bool(req.run_jersey_number_recognition),
                qwen_vl_url=str(req.qwen_vl_url),
                qwen_vl_model=str(req.qwen_vl_model),
                jersey_prompt=str(req.jersey_prompt),
                jersey_max_tracks=int(req.jersey_max_tracks),
                jersey_max_samples_per_track=int(req.jersey_max_samples_per_track),
                jersey_min_det_conf=float(req.jersey_min_det_conf),
                jersey_min_box_area=int(req.jersey_min_box_area),
                jersey_min_frame_gap=int(req.jersey_min_frame_gap),
                jersey_in_tracking=bool(req.jersey_in_tracking),
                jersey_merge_same_number=bool(req.jersey_merge_same_number),
                jersey_merge_min_confidence=float(req.jersey_merge_min_confidence),
                jersey_merge_max_overlap_frames=int(req.jersey_merge_max_overlap_frames),
                possession_dist_norm=float(req.possession_dist_norm),
                possession_stable_frames=int(req.possession_stable_frames),
                possession_stride_frames=int(req.possession_stride_frames),
                ball_cls_id=int(req.ball_cls_id),
                player_cls_id=int(req.player_cls_id),
            )

            result = _run(video_path=requested_path, out_dir=UPLOADS_DIR, cfg=cfg, progress_cb=progress_cb)

            payload = {
                **result,
                "segment_url": f"http://localhost:8000/api/video?path={requests_quote(result['segment_path'])}",
                "overlay_video_url": f"http://localhost:8000/api/video?path={requests_quote(result['overlay_video_path'])}",
                "events_json_url": f"http://localhost:8000/api/file?path={requests_quote(result['events_json_path'])}",
                **(
                    {"tracking_video_url": f"http://localhost:8000/api/video?path={requests_quote(result['tracking_video_path'])}"}
                    if result.get("tracking_video_path")
                    else {}
                ),
                **(
                    {"tracks_csv_url": f"http://localhost:8000/api/file?path={requests_quote(result['tracks_csv_path'])}"}
                    if result.get("tracks_csv_path")
                    else {}
                ),
            }

            _set_job(
                job_id,
                {
                    "status": "done",
                    "stage": "done",
                    "current": 1,
                    "total": 1,
                    "message": "Tamamlandı",
                    "result": payload,
                    "updated_utc": datetime.utcnow().isoformat() + "Z",
                },
            )
        except Exception as e:
            _set_job(
                job_id,
                {
                    "status": "error",
                    "stage": "error",
                    "message": f"Hata: {e}",
                    "error": str(e),
                    "updated_utc": datetime.utcnow().isoformat() + "Z",
                },
            )

    t = threading.Thread(target=worker, daemon=True)
    t.start()

    return {
        "job_id": job_id,
        "progress_url": f"http://localhost:8000/api/pipeline_progress?job_id={requests_quote(job_id)}",
    }


@app.get("/api/pipeline_progress")
async def pipeline_progress(job_id: str):
    """Server-Sent Events progress stream for a running job."""

    async def event_stream():
        last_payload = None
        while True:
            job = _get_job(job_id)
            if not job:
                data = {"status": "error", "message": "Unknown job_id"}
                yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
                return

            # Only send when changed
            payload = {
                "status": job.get("status"),
                "stage": job.get("stage"),
                "current": job.get("current"),
                "total": job.get("total"),
                "message": job.get("message"),
                "updated_utc": job.get("updated_utc"),
            }

            # Always stream (keeps frontend responsive even if stage doesn't change)
            last_payload = payload
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

            if job.get("status") in ("done", "error"):
                if job.get("status") == "done":
                    yield f"data: {json.dumps({'type':'result','result': job.get('result')}, ensure_ascii=False)}\n\n"
                else:
                    yield f"data: {json.dumps({'type':'error','error': job.get('error')}, ensure_ascii=False)}\n\n"
                return

            # heartbeat
            yield ": ping\n\n"
            await asyncio.sleep(0.5)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/api/upload_video")
async def upload_video(file: UploadFile = File(...)):
    """Upload a video file into backend uploads directory."""
    os.makedirs(UPLOADS_DIR, exist_ok=True)

    filename = (file.filename or "upload.mp4").strip().replace("\\", "/")
    base = os.path.basename(filename)
    base_name, ext = os.path.splitext(base)
    if not ext:
        ext = ".mp4"
    ext_l = ext.lower()
    # Accept any extension (best-effort). ffmpeg/OpenCV may still fail later.

    # unique name to avoid collisions
    stamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    rand_hex = f"{random.getrandbits(32):08x}"
    safe_base = "".join([c for c in base_name if c.isalnum() or c in (" ", "-", "_", ".")]).strip() or "video"
    out_name = f"{safe_base}_{stamp}_{rand_hex}{ext_l}"
    out_path = os.path.join(UPLOADS_DIR, out_name)

    try:
        with open(out_path, "wb") as f:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
    except Exception as e:
        try:
            if os.path.exists(out_path):
                os.remove(out_path)
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=f"Upload failed: {e}")

    return {
        "path": out_path,
        "name": out_name,
        "url": f"http://localhost:8000/api/video?path={requests_quote(out_path)}",
    }

class VideoFile(BaseModel):
    path: str
    name: str

@app.get("/api/videos", response_model=List[VideoFile])
async def list_videos():
    """
    Recursively scans the video directory and returns a list of supported video files.
    """
    video_files = []
    if not os.path.isdir(VIDEO_DIR):
        logger.warning(f"Video directory not found: {VIDEO_DIR}")
        return []
        
    for root, _, files in os.walk(VIDEO_DIR):
        for file in sorted(files):
            if any(file.lower().endswith(ext) for ext in SUPPORTED_VIDEO_FORMATS):
                full_path = os.path.join(root, file)
                video_files.append(VideoFile(path=full_path, name=file))
    
    logger.info(f"Found {len(video_files)} videos in {VIDEO_DIR}")
    return video_files


@app.get("/api/video")
async def get_video_stream(request: Request, path: str):
    """
    Streams a video file from the filesystem.
    Takes a 'path' query parameter, which should be a valid path to a video file.
    """
    # Security: Ensure the requested path is within the allowed video directory
    # and is a real file, to prevent directory traversal attacks.
    requested_path = os.path.abspath(path)
    if not _is_allowed_media_path(requested_path):
        raise HTTPException(status_code=404, detail="Video not found or access denied")
    
    # Determine media type (ensure mp4 is recognized)
    media_type, _ = mimetypes.guess_type(requested_path)
    if not media_type:
        if requested_path.lower().endswith('.mp4'):
            media_type = 'video/mp4'
        else:
            media_type = 'application/octet-stream'

    def _parse_http_range(range_value: str, size: int) -> Optional[tuple[int, int]]:
        """Parse a single HTTP Range header value.

        Supports:
        - bytes=start-end
        - bytes=start-
        - bytes=-suffix_len

        Returns (start, end) inclusive, or None if malformed/unsupported.
        """
        if not range_value:
            return None
        try:
            units, ranges = range_value.split("=", 1)
        except ValueError:
            return None
        if units.strip().lower() != "bytes":
            return None
        # We only support a single range.
        if "," in ranges:
            return None
        part = ranges.strip()
        if "-" not in part:
            return None
        start_str, end_str = part.split("-", 1)
        start_str = start_str.strip()
        end_str = end_str.strip()

        if start_str == "":
            # suffix-byte-range-spec: "-<length>" => last <length> bytes
            if end_str == "":
                return None
            try:
                suffix_len = int(end_str)
            except ValueError:
                return None
            if suffix_len <= 0:
                return None
            if suffix_len >= size:
                return (0, max(0, size - 1))
            return (size - suffix_len, size - 1)

        try:
            start = int(start_str)
        except ValueError:
            return None
        if start < 0:
            return None

        if end_str == "":
            end = size - 1
        else:
            try:
                end = int(end_str)
            except ValueError:
                return None
            if end < start:
                return None

        if start >= size:
            return None
        end = min(end, size - 1)
        return (start, end)

    # Support HTTP Range requests for video playback in browsers
    size = os.path.getsize(requested_path)
    range_header = request.headers.get("range")
    if not range_header:
        resp = FileResponse(requested_path, media_type=media_type)
        resp.headers["Accept-Ranges"] = "bytes"
        return resp

    parsed = _parse_http_range(range_header, size)
    if not parsed:
        # Malformed/unsupported Range; return whole file (still advertise byte ranges).
        resp = FileResponse(requested_path, media_type=media_type)
        resp.headers["Accept-Ranges"] = "bytes"
        return resp

    start, end = parsed
    length = end - start + 1

    def stream_range(path, start, length):
        with open(path, 'rb') as f:
            f.seek(start)
            remaining = length
            chunk_size = 1024 * 1024
            while remaining > 0:
                read_size = min(chunk_size, remaining)
                data = f.read(read_size)
                if not data:
                    break
                remaining -= len(data)
                yield data

    headers = {
        'Content-Range': f'bytes {start}-{end}/{size}',
        'Accept-Ranges': 'bytes',
        'Content-Length': str(length),
        'Content-Type': media_type,
    }
    return StreamingResponse(stream_range(requested_path, start, length), status_code=206, headers=headers)


@app.head("/api/video")
async def head_video_stream(request: Request, path: str):
    """HEAD variant of `/api/video`.

    Some browsers/players issue a HEAD request before starting playback.
    We return the same headers as GET would, but without a response body.
    """
    requested_path = os.path.abspath(path)
    if not _is_allowed_media_path(requested_path):
        raise HTTPException(status_code=404, detail="Video not found or access denied")

    media_type, _ = mimetypes.guess_type(requested_path)
    if not media_type:
        if requested_path.lower().endswith('.mp4'):
            media_type = 'video/mp4'
        else:
            media_type = 'application/octet-stream'

    def _parse_http_range(range_value: str, size: int) -> Optional[tuple[int, int]]:
        if not range_value:
            return None
        try:
            units, ranges = range_value.split("=", 1)
        except ValueError:
            return None
        if units.strip().lower() != "bytes":
            return None
        if "," in ranges:
            return None
        part = ranges.strip()
        if "-" not in part:
            return None
        start_str, end_str = part.split("-", 1)
        start_str = start_str.strip()
        end_str = end_str.strip()

        if start_str == "":
            if end_str == "":
                return None
            try:
                suffix_len = int(end_str)
            except ValueError:
                return None
            if suffix_len <= 0:
                return None
            if suffix_len >= size:
                return (0, max(0, size - 1))
            return (size - suffix_len, size - 1)

        try:
            start = int(start_str)
        except ValueError:
            return None
        if start < 0:
            return None

        if end_str == "":
            end = size - 1
        else:
            try:
                end = int(end_str)
            except ValueError:
                return None
            if end < start:
                return None

        if start >= size:
            return None
        end = min(end, size - 1)
        return (start, end)

    size = os.path.getsize(requested_path)
    range_header = request.headers.get("range")
    if not range_header:
        return Response(
            status_code=200,
            headers={
                "Accept-Ranges": "bytes",
                "Content-Length": str(size),
                "Content-Type": media_type,
            },
        )

    parsed = _parse_http_range(range_header, size)
    if not parsed:
        return Response(
            status_code=200,
            headers={
                "Accept-Ranges": "bytes",
                "Content-Length": str(size),
                "Content-Type": media_type,
            },
        )

    start, end = parsed
    length = end - start + 1
    return Response(
        status_code=206,
        headers={
            "Content-Range": f"bytes {start}-{end}/{size}",
            "Accept-Ranges": "bytes",
            "Content-Length": str(length),
            "Content-Type": media_type,
        },
    )


@app.get("/api/uploads")
async def list_uploaded_videos():
    files = []
    if not os.path.isdir(UPLOADS_DIR):
        return files
    for f in sorted(os.listdir(UPLOADS_DIR)):
        full = os.path.join(UPLOADS_DIR, f)
        if not (os.path.isfile(full) and any(f.lower().endswith(ext) for ext in SUPPORTED_VIDEO_FORMATS)):
            continue
        # Try to extract video_id from filename pattern: base_<hex8>_normal.mp4
        video_id = None
        try:
            name_parts = f.rsplit(".", 1)[0].split("_")
            # look for 8-hex chunk in parts
            for part in name_parts:
                if len(part) == 8:
                    try:
                        video_id = int(part, 16)
                        break
                    except ValueError:
                        continue
        except Exception:
            video_id = None
        item = VideoFile(path=full, name=f)
        url = f"http://localhost:8000/api/video?path={requests_quote(full)}"
        kind = "video"
        lf = f.lower()
        if lf.startswith("overlay_"):
            kind = "overlay"
        elif lf.startswith("tracking_"):
            kind = "tracking"
        elif lf.startswith("segment_"):
            kind = "segment"
        # attach video_id if found (pydantic model will ignore extras on response)
        files.append(
            {
                "path": item.path,
                "name": item.name,
                "url": url,
                "kind": kind,
                **({"video_id": video_id} if video_id is not None else {}),
            }
        )
    return files


@app.get("/api/uploaded_video")
async def get_uploaded_video(path: str):
    requested_path = os.path.normcase(os.path.realpath(os.path.abspath(path)))
    uploads_root = os.path.normcase(os.path.realpath(os.path.abspath(UPLOADS_DIR)))
    try:
        in_uploads = os.path.commonpath([requested_path, uploads_root]) == uploads_root
    except Exception:
        in_uploads = False
    if not in_uploads or not os.path.isfile(requested_path):
        raise HTTPException(status_code=404, detail="Uploaded video not found or access denied")
    media_type, _ = mimetypes.guess_type(requested_path)
    if not media_type or not media_type.startswith("video/"):
        media_type = "application/octet-stream"
    return FileResponse(requested_path, media_type=media_type)


@app.get("/api/file")
async def get_file(path: str):
    requested_path = os.path.abspath(path)
    if not _is_allowed_media_path(requested_path):
        raise HTTPException(status_code=404, detail="File not found or access denied")
    media_type, _ = mimetypes.guess_type(requested_path)
    if not media_type:
        media_type = "application/octet-stream"
    return FileResponse(requested_path, media_type=media_type)


@app.get("/api/thumbnail")
async def get_thumbnail(path: str):
    requested_path = os.path.abspath(path)
    if not _is_allowed_media_path(requested_path):
        raise HTTPException(status_code=404, detail="Video not found or access denied")
    try:
        cap = cv2.VideoCapture(requested_path)
        success, frame = cap.read()
        cap.release()
        if not success or frame is None:
            raise HTTPException(status_code=500, detail="Failed to read frame for thumbnail")
        # encode as jpeg
        ret, buf = cv2.imencode('.jpg', frame)
        if not ret:
            raise HTTPException(status_code=500, detail="Failed to encode thumbnail")
        return Response(content=buf.tobytes(), media_type='image/jpeg')
    except Exception as e:
        logger.error(f"Error generating thumbnail for {requested_path}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class ExtractRequest(BaseModel):
    path: str
    minutes: float
    start_seconds: Optional[float] = 0.0


@app.post("/api/extract")
async def extract_clip(req: ExtractRequest):
    requested_path = os.path.abspath(req.path)
    if not _is_allowed_media_path(requested_path):
        raise HTTPException(status_code=404, detail="Video not found or access denied")

    # ensure uploads dir exists
    os.makedirs(UPLOADS_DIR, exist_ok=True)

    duration = float(req.minutes) * 60.0
    start = float(req.start_seconds or 0.0)

    base_name = os.path.splitext(os.path.basename(requested_path))[0]
    timestamp = datetime.utcnow().strftime('%Y%m%d%H%M%S')
    out_name = f"{base_name}_clip_{timestamp}.mp4"
    out_path = os.path.join(UPLOADS_DIR, out_name)

    # Try to use ffmpeg if available, otherwise fall back to OpenCV-based extraction
    ffmpeg_bin = _ffmpeg_exe()
    if ffmpeg_bin:
        # Build ffmpeg command (re-encode to ensure compatibility)
        cmd = [
            ffmpeg_bin,
            "-y",
            "-ss",
            str(start),
            "-i",
            requested_path,
            "-t",
            str(duration),
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

        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"ffmpeg failed: {e.stderr}")
            raise HTTPException(status_code=500, detail=f"ffmpeg failed: {e.stderr}")
    else:
        # Fallback: use OpenCV to read frames and write a new MP4
        try:
            cap = cv2.VideoCapture(requested_path)
            if not cap.isOpened():
                raise HTTPException(status_code=500, detail="Failed to open video for extraction")

            src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            src_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
            src_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)

            start_frame = int(start * src_fps)
            total_frames = int(duration * src_fps)

            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(out_path, fourcc, src_fps, (src_width, src_height))
            if not writer.isOpened():
                cap.release()
                raise HTTPException(status_code=500, detail="Failed to create video writer for extraction")

            written = 0
            for _ in range(total_frames):
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                # ensure correct size
                if frame.shape[1] != src_width or frame.shape[0] != src_height:
                    frame = cv2.resize(frame, (src_width, src_height))
                writer.write(frame)
                written += 1

            writer.release()
            cap.release()

            if written == 0:
                # remove zero-length file if created
                try:
                    os.remove(out_path)
                except Exception:
                    pass
                raise HTTPException(status_code=500, detail="No frames written during extraction")
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"OpenCV extraction failed: {e}")
            raise HTTPException(status_code=500, detail=f"OpenCV extraction failed: {e}")

    # verify output exists and is non-empty
    if not os.path.isfile(out_path) or os.path.getsize(out_path) == 0:
        logger.error(f"Extraction produced no output file: {out_path}")
        raise HTTPException(status_code=500, detail="Extraction failed to produce output file")

    return {"path": out_path, "url": f"http://localhost:8000/api/video?path={requests_quote(out_path)}"}


class DeleteRequest(BaseModel):
    path: str


@app.post("/api/delete_upload")
async def delete_upload(req: DeleteRequest):
    requested_path = os.path.abspath(req.path)
    if not requested_path.startswith(UPLOADS_DIR) or not os.path.isfile(requested_path):
        raise HTTPException(status_code=404, detail="Uploaded file not found or access denied")
    try:
        os.remove(requested_path)
        return {"deleted": True}
    except Exception as e:
        logger.error(f"Failed to delete uploaded file {requested_path}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/process_frame", response_model=FrameResponse)
async def process_frame(request: FrameRequest):
    """
    Receives a single frame, runs YOLO inference, and returns detections.
    """
    yolo_model = ml_models.get("yolo")
    if yolo_model is None:
        raise HTTPException(status_code=503, detail="YOLO model is not available.")

    try:
        # Decode base64 string
        img_str = request.frame_data.split(",")[1]
        img_bytes = base64.b64decode(img_str)
        
        # Convert to numpy array for OpenCV
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            raise HTTPException(status_code=400, detail="Failed to decode frame from base64 string.")

        # Run YOLO inference (convert BGR->RGB for the model)
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except Exception:
            frame_rgb = frame
        inference_start_time = time.time()
        # Use reasonable defaults for confidence and image size; avoid half precision forcing
        try:
            results = yolo_model(frame_rgb, verbose=False, imgsz=640, conf=0.25)
        except TypeError:
            # Fallback if signature differs
            results = yolo_model.predict(frame_rgb, imgsz=640, conf=0.25)
        inference_time = (time.time() - inference_start_time) * 1000  # ms

        # Parse detections
        detections = []
        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            img_height, img_width = frame.shape[:2]
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                class_id = int(box.cls[0].cpu().numpy())
                class_name = yolo_model.names.get(class_id, "unknown").lower()
                
                if class_name in ["player", "ball", "referee", "person"]:
                    detections.append(Detection(
                        x=float(x1 / img_width),
                        y=float(y1 / img_height),
                        width=float((x2 - x1) / img_width),
                        height=float((y2 - y1) / img_height),
                        confidence=float(box.conf[0].cpu().numpy()),
                        class_name="player" if class_name == "person" else class_name
                    ))

        return FrameResponse(
            frame_id=request.frame_id,
            inference_time=inference_time,
            detections=detections
        )

    except Exception as e:
        logger.error(f"Error processing frame {request.frame_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
