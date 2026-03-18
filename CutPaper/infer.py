from pathlib import Path
import argparse

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLO inference on image(s)")
    parser.add_argument("source", help="Image path, folder path, video path, or URL")
    parser.add_argument(
        "--weights",
        default="runs/detect/runs/train/cutpaper/weights/best.pt",
        help="Path to model weights (.pt)",
    )
    parser.add_argument("--device", default="0", help="CUDA device id, cpu, or auto")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.7, help="IoU threshold for NMS")
    parser.add_argument("--project", default="runs/detect", help="Output directory")
    parser.add_argument("--name", default="predict", help="Run name")
    parser.add_argument(
        "--nosave",
        action="store_true",
        help="Do not save visualized prediction images",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent

    weights_path = (root / args.weights).resolve()
    source_value = args.source

    source_path = Path(source_value)
    if source_path.exists():
        source_value = str(source_path.resolve())

    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    # Force project path to be workspace-relative to avoid duplicated runs/detect nesting.
    project_path = Path(args.project)
    if not project_path.is_absolute():
        project_path = (root / project_path).resolve()

    model = YOLO(str(weights_path))
    model.predict(
        source=source_value,
        device=args.device,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        project=str(project_path),
        name=args.name,
        save=not args.nosave,
    )


if __name__ == "__main__":
    main()
