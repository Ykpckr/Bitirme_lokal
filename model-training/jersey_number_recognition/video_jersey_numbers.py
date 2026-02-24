import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np

try:
    import cv2
except Exception as exc:
    raise ImportError(
        "OpenCV is required. Install with: pip install opencv-python"
    ) from exc

try:
    from ultralytics import YOLO
except Exception as exc:
    raise ImportError(
        "Ultralytics is required. Install with: pip install ultralytics"
    ) from exc

import torch
from PIL import Image
from torchvision import transforms

from model import create_model


def build_transform(use_original_size: bool, image_size: int) -> transforms.Compose:
    if use_original_size:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def load_jersey_model(model_path: Path, device: str) -> Tuple[torch.nn.Module, dict, bool]:
    if not model_path.exists():
        raise FileNotFoundError(f"Jersey model not found: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)
    idx_to_label = {int(k): v for k, v in checkpoint["idx_to_label"].items()}
    num_classes = len(idx_to_label)
    model_name = checkpoint["config"]["MODEL_NAME"]
    use_original_size = checkpoint["config"].get("USE_ORIGINAL_SIZE", False)

    model = create_model(num_classes, model_name, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model, idx_to_label, use_original_size


def classify_crop(
    crop_bgr: np.ndarray,
    model: torch.nn.Module,
    transform: transforms.Compose,
    idx_to_label: dict,
    device: str
) -> Tuple[str, float]:
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(crop_rgb)
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0)
        conf, pred_idx = torch.max(probs, dim=0)

    label = idx_to_label.get(int(pred_idx.item()), "?")
    if label == -1:
        label = "no-number"
    return str(label), float(conf.item())


def draw_labels(
    frame: np.ndarray,
    detections: List[Tuple[int, int, int, int, str, float]],
    color: Tuple[int, int, int] = (0, 255, 0)
) -> np.ndarray:
    for x1, y1, x2, y2, label, conf in detections:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text = f"{label} ({conf:.2f})"
        cv2.putText(
            frame,
            text,
            (x1, max(0, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA
        )
    return frame


def run_video(
    video_path: Path,
    player_model_path: Path,
    jersey_model_path: Path,
    output_path: Path,
    det_conf: float,
    jersey_conf: float,
    frame_stride: int,
    image_size: int,
    show: bool,
    device: str
):
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not player_model_path.exists():
        raise FileNotFoundError(f"Player detector not found: {player_model_path}")

    jersey_model, idx_to_label, use_original_size = load_jersey_model(jersey_model_path, device)
    transform = build_transform(use_original_size, image_size)

    yolo = YOLO(str(player_model_path))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    last_detections: List[Tuple[int, int, int, int, str, float]] = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_stride == 0:
            results = yolo.predict(
                source=frame,
                conf=det_conf,
                device=device,
                verbose=False
            )
            detections: List[Tuple[int, int, int, int, str, float]] = []

            if results and len(results) > 0:
                result = results[0]
                if result.boxes is not None and result.boxes.xyxy is not None:
                    boxes = result.boxes.xyxy.cpu().numpy().astype(int)
                    confs = result.boxes.conf.cpu().numpy()

                    for (x1, y1, x2, y2), box_conf in zip(boxes, confs):
                        if box_conf < det_conf:
                            continue

                        x1 = max(0, x1)
                        y1 = max(0, y1)
                        x2 = min(width - 1, x2)
                        y2 = min(height - 1, y2)
                        if x2 <= x1 or y2 <= y1:
                            continue

                        crop = frame[y1:y2, x1:x2]
                        if crop.size == 0:
                            continue

                        label, cls_conf = classify_crop(crop, jersey_model, transform, idx_to_label, device)
                        if cls_conf < jersey_conf:
                            label = "?"

                        detections.append((x1, y1, x2, y2, label, cls_conf))

            last_detections = detections

        annotated = frame.copy()
        annotated = draw_labels(annotated, last_detections)

        writer.write(annotated)

        if show:
            cv2.imshow("Jersey Number Recognition", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_idx += 1

    cap.release()
    writer.release()
    if show:
        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    default_video = Path(
        r"H:\SoccerNetVideos\england_epl\2014-2015\2015-02-21 - 18-00 Chelsea 1 - 1 Burnley\1_720p.mkv"
    )

    parser = argparse.ArgumentParser(description="Jersey number recognition on video")
    parser.add_argument("--video", type=Path, default=default_video, help="Input video path")
    parser.add_argument(
        "--player-model",
        type=Path,
        default=project_root / "model-training" / "player-detection" / "models" / "football_detector_optimized" / "weights" / "best.pt",
        help="YOLO player detector weights"
    )
    parser.add_argument(
        "--jersey-model",
        type=Path,
        default=project_root / "output" / "best_model.pth",
        help="Jersey number classifier weights"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "output" / "jersey_video_predictions.mp4",
        help="Output video path"
    )
    parser.add_argument("--det-conf", type=float, default=0.35, help="Player detection confidence")
    parser.add_argument("--jersey-conf", type=float, default=0.45, help="Jersey classification confidence")
    parser.add_argument("--frame-stride", type=int, default=2, help="Process every Nth frame")
    parser.add_argument("--image-size", type=int, default=512, help="Resize size for jersey model")
    parser.add_argument("--show", action="store_true", help="Show live window")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for inference"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    run_video(
        video_path=args.video,
        player_model_path=args.player_model,
        jersey_model_path=args.jersey_model,
        output_path=args.output,
        det_conf=args.det_conf,
        jersey_conf=args.jersey_conf,
        frame_stride=max(1, args.frame_stride),
        image_size=args.image_size,
        show=args.show,
        device=args.device
    )

    print(f"Saved output video to: {args.output}")


if __name__ == "__main__":
    main()