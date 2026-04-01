#!/usr/bin/env python3
import argparse
import csv
import importlib.util
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


def load_detector_class(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.PaperDetector_yolo


def contour_to_xyxy(contour) -> Optional[Tuple[float, float, float, float]]:
    if contour is None:
        return None
    points = contour.reshape(-1, 2)
    x1, y1 = points.min(axis=0)
    x2, y2 = points.max(axis=0)
    return float(x1), float(y1), float(x2), float(y2)


def iou_xyxy(
    a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]
) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter

    return inter / union if union > 0.0 else 0.0


def collect_images(img_dir: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted(
        [p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    )


def get_class_id_map(names) -> Dict[str, int]:
    class_map: Dict[str, int] = {}
    if isinstance(names, dict):
        items = names.items()
    else:
        items = enumerate(names)

    for cls_id, cls_name in items:
        name = str(cls_name).lower()
        if name in {"object", "paper"}:
            class_map[name] = int(cls_id)

    return class_map


def get_top_box_by_class(
    detector, image, conf: float, device: str
) -> Dict[str, Optional[Tuple[float, float, float, float]]]:
    result_map = {"object": None, "paper": None}

    names = detector.model.names if detector.model is not None else {}
    class_map = get_class_id_map(names)
    target_ids = [class_map[k] for k in ["object", "paper"] if k in class_map]
    if not target_ids:
        return result_map

    results = detector.model.predict(
        source=image,
        conf=conf,
        device=device,
        verbose=False,
        classes=target_ids,
    )
    if not results:
        return result_map

    r = results[0]
    if r.boxes is None or len(r.boxes) == 0:
        return result_map

    boxes_cls = r.boxes.cls.cpu().numpy().astype(int)
    boxes_conf = r.boxes.conf.cpu().numpy()
    boxes_xyxy = r.boxes.xyxy.cpu().numpy()

    for label in ["object", "paper"]:
        cls_id = class_map.get(label)
        if cls_id is None:
            continue
        idxs = [i for i, c in enumerate(boxes_cls) if c == cls_id]
        if not idxs:
            continue
        best_idx = max(idxs, key=lambda i: boxes_conf[i])
        x1, y1, x2, y2 = boxes_xyxy[best_idx]
        result_map[label] = (float(x1), float(y1), float(x2), float(y2))

    return result_map


def draw_overlay(
    image_path: Path,
    t1_object_box: Optional[Tuple[float, float, float, float]],
    t2_object_box: Optional[Tuple[float, float, float, float]],
    t1_paper_box: Optional[Tuple[float, float, float, float]],
    t2_paper_box: Optional[Tuple[float, float, float, float]],
    object_iou: Optional[float],
    paper_iou: Optional[float],
    object_status: str,
    paper_status: str,
    save_path: Path,
):
    image = cv2.imread(str(image_path))
    if image is None:
        return

    if t1_object_box is not None:
        x1, y1, x2, y2 = map(int, t1_object_box)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(
            image,
            "t1_object",
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 200, 0),
            2,
        )

    if t2_object_box is not None:
        x1, y1, x2, y2 = map(int, t2_object_box)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 220), 2)
        cv2.putText(
            image,
            "t2_object",
            (x1, max(40, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 220),
            2,
        )

    if t1_paper_box is not None:
        x1, y1, x2, y2 = map(int, t1_paper_box)
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 200, 0), 2)
        cv2.putText(
            image,
            "t1_paper",
            (x1, max(60, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 200, 0),
            2,
        )

    if t2_paper_box is not None:
        x1, y1, x2, y2 = map(int, t2_paper_box)
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 255), 2)
        cv2.putText(
            image,
            "t2_paper",
            (x1, max(80, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 255),
            2,
        )

    obj_label = (
        f"object IoU={object_iou:.4f}"
        if object_iou is not None
        else f"object: {object_status}"
    )
    paper_label = (
        f"paper IoU={paper_iou:.4f}"
        if paper_iou is not None
        else f"paper: {paper_status}"
    )
    cv2.putText(
        image,
        obj_label,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        image,
        paper_label,
        (10, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), image)


def main():
    parser = argparse.ArgumentParser(
        description="Compute IoU between ch3-t1 and ch3-t2 for object and paper classes."
    )
    parser.add_argument(
        "--img-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "img",
        help="Folder containing test images",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "ch3-t1" / "models" / "best.pt",
        help="YOLO weights path",
    )
    parser.add_argument("--conf", type=float, default=0.85, help="Confidence threshold")
    parser.add_argument(
        "--device", type=str, default="cpu", help="Inference device, e.g. cpu or 0"
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path(__file__).resolve().parent / "iou_results.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--overlay-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "overlay",
        help="Folder to save overlay visualization images",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    det1_path = root / "ch3-t1" / "PaperDetector_yolo.py"
    det2_path = root / "ch3-t2" / "PaperDetector_yolo.py"

    if not args.img_dir.exists():
        raise FileNotFoundError(f"Image folder not found: {args.img_dir}")
    if not args.weights.exists():
        raise FileNotFoundError(f"Weights not found: {args.weights}")

    Detector1 = load_detector_class(det1_path, "paper_detector_t1")
    Detector2 = load_detector_class(det2_path, "paper_detector_t2")

    images = collect_images(args.img_dir)
    if not images:
        print(f"No images found in: {args.img_dir}")
        return

    rows = []
    object_ious = []
    paper_ious = []
    args.overlay_dir.mkdir(parents=True, exist_ok=True)

    for image_path in images:
        det1 = Detector1(str(image_path), weights_path=str(args.weights))
        det2 = Detector2(str(image_path), weights_path=str(args.weights))
        image = cv2.imread(str(image_path))
        if image is None:
            rows.append(
                {
                    "image": image_path.name,
                    "object_iou": "",
                    "paper_iou": "",
                    "object_status": "image_read_failed",
                    "paper_status": "image_read_failed",
                    "t1_object_box": "",
                    "t2_object_box": "",
                    "t1_paper_box": "",
                    "t2_paper_box": "",
                }
            )
            print(f"{image_path.name}: image_read_failed")
            continue

        t1_boxes = get_top_box_by_class(det1, image, conf=args.conf, device=args.device)
        t2_boxes = get_top_box_by_class(det2, image, conf=args.conf, device=args.device)

        t1_object_box = t1_boxes["object"]
        t2_object_box = t2_boxes["object"]
        t1_paper_box = t1_boxes["paper"]
        t2_paper_box = t2_boxes["paper"]

        if t1_object_box is not None and t2_object_box is not None:
            object_iou = iou_xyxy(t1_object_box, t2_object_box)
            object_status = "ok"
            object_ious.append(object_iou)
        else:
            object_iou = None
            object_status = "missing_detection"

        if t1_paper_box is not None and t2_paper_box is not None:
            paper_iou = iou_xyxy(t1_paper_box, t2_paper_box)
            paper_status = "ok"
            paper_ious.append(paper_iou)
        else:
            paper_iou = None
            paper_status = "missing_detection"

        draw_overlay(
            image_path=image_path,
            t1_object_box=t1_object_box,
            t2_object_box=t2_object_box,
            t1_paper_box=t1_paper_box,
            t2_paper_box=t2_paper_box,
            object_iou=object_iou,
            paper_iou=paper_iou,
            object_status=object_status,
            paper_status=paper_status,
            save_path=args.overlay_dir / image_path.name,
        )

        rows.append(
            {
                "image": image_path.name,
                "object_iou": f"{object_iou:.6f}" if object_iou is not None else "",
                "paper_iou": f"{paper_iou:.6f}" if paper_iou is not None else "",
                "object_status": object_status,
                "paper_status": paper_status,
                "t1_object_box": (
                    f"{t1_object_box}" if t1_object_box is not None else ""
                ),
                "t2_object_box": (
                    f"{t2_object_box}" if t2_object_box is not None else ""
                ),
                "t1_paper_box": f"{t1_paper_box}" if t1_paper_box is not None else "",
                "t2_paper_box": f"{t2_paper_box}" if t2_paper_box is not None else "",
            }
        )
        object_msg = (
            f"object IoU={object_iou:.6f}"
            if object_iou is not None
            else "object missing_detection"
        )
        paper_msg = (
            f"paper IoU={paper_iou:.6f}"
            if paper_iou is not None
            else "paper missing_detection"
        )
        print(f"{image_path.name}: {object_msg}, {paper_msg}")

    object_mean_iou = float(np.mean(object_ious)) if object_ious else None
    object_min_iou = float(np.min(object_ious)) if object_ious else None
    object_max_iou = float(np.max(object_ious)) if object_ious else None
    paper_mean_iou = float(np.mean(paper_ious)) if paper_ious else None
    paper_min_iou = float(np.min(paper_ious)) if paper_ious else None
    paper_max_iou = float(np.max(paper_ious)) if paper_ious else None

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image",
                "object_iou",
                "paper_iou",
                "object_status",
                "paper_status",
                "t1_object_box",
                "t2_object_box",
                "t1_paper_box",
                "t2_paper_box",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\nSummary")
    print(f"images_total: {len(images)}")
    print(f"object_images_with_both_boxes: {len(object_ious)}")
    print(
        f"object_mean_iou: {object_mean_iou if object_mean_iou is not None else 'N/A'}"
    )
    print(f"object_min_iou: {object_min_iou if object_min_iou is not None else 'N/A'}")
    print(f"object_max_iou: {object_max_iou if object_max_iou is not None else 'N/A'}")
    print(f"paper_images_with_both_boxes: {len(paper_ious)}")
    print(f"paper_mean_iou: {paper_mean_iou if paper_mean_iou is not None else 'N/A'}")
    print(f"paper_min_iou: {paper_min_iou if paper_min_iou is not None else 'N/A'}")
    print(f"paper_max_iou: {paper_max_iou if paper_max_iou is not None else 'N/A'}")
    print(f"csv_saved: {args.csv}")
    print(f"overlay_saved_dir: {args.overlay_dir}")


if __name__ == "__main__":
    main()
