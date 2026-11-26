"""Quick check that the weights load and detect people. Run this BEFORE committing / integrating.

    pip install ultralytics==8.4.190
    python smoke_test.py --image path/to/crowd.jpg
    python smoke_test.py --image path/to/crowd.jpg --weights weights/best_v2.pt --conf 0.2

Prints the person count and writes an annotated copy next to the input (<name>_pred.jpg).
Open the output and check that the boxes sit on people.
"""
import argparse
import sys
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--weights", default=None, help="default: ./weights/best_v2.pt or YOLO_WEIGHTS env var")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=960)
    a = ap.parse_args()

    img_path = Path(a.image)
    if not img_path.is_file():
        sys.exit(f"Image not found: {img_path}")

    import cv2
    from person_detector import PersonDetector

    frame = cv2.imread(str(img_path))
    if frame is None:
        sys.exit(f"Could not read image: {img_path}")

    det = PersonDetector(weights=a.weights, imgsz=a.imgsz, conf=a.conf)
    det.detect(frame)                                  # warm-up
    t0 = time.perf_counter()
    people = det.detect(frame)
    ms = (time.perf_counter() - t0) * 1000

    out = img_path.with_name(img_path.stem + "_pred.jpg")
    cv2.imwrite(str(out), det.annotate(frame))
    print(f"weights : {det.weights}")
    print(f"persons : {len(people)} (conf >= {a.conf}, imgsz {a.imgsz}, half={det.half})")
    print(f"latency : {ms:.0f} ms for one frame (after warm-up)")
    print(f"saved   : {out}")


if __name__ == "__main__":
    main()
