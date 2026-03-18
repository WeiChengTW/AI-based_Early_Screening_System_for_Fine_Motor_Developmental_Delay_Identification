from pathlib import Path
import argparse

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO model on CutPaper dataset")
    parser.add_argument("--data", default="data.yaml", help="Path to dataset yaml file")
    parser.add_argument(
        "--model", default="yolov8n.pt", help="Pretrained model checkpoint"
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--device", default="0", help="CUDA device id, cpu, or auto")
    parser.add_argument(
        "--project", default="runs/train", help="Directory for training outputs"
    )
    parser.add_argument("--name", default="cutpaper", help="Run name")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    data_path = (root / args.data).resolve()

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset yaml not found: {data_path}")

    # Force project path to be workspace-relative to avoid duplicated runs/train nesting.
    project_path = Path(args.project)
    if not project_path.is_absolute():
        project_path = (root / project_path).resolve()

    model = YOLO(args.model)
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(project_path),
        name=args.name,
    )


if __name__ == "__main__":
    main()
