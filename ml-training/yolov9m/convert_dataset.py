
from pathlib import Path
import shutil

SOURCE = Path.home() / "Downloads" / "tiny-person.v1i.yolov9"
OUTPUT = Path("ml-training/yolov9m/data")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
KEEP_CLASS_ID = 0  # person_on_earth
NEW_CLASS_ID = 0   # single class: person


def convert_polygon_to_box(values):
    points = values[1:]

    if len(points) < 6 or len(points) % 2 != 0:
        raise ValueError("Invalid polygon coordinates")

    coords = [float(v) for v in points]

    tolerance = 1e-6

    if not all(-tolerance <= v <= 1.0 + tolerance for v in coords):
        raise ValueError("Coordinates outside normalized range")

    coords = [min(1.0, max(0.0, v)) for v in coords]

    xs = coords[0::2]
    ys = coords[1::2]

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    width = x_max - x_min
    height = y_max - y_min

    if width <= 0 or height <= 0:
        raise ValueError("Invalid bounding box dimensions")

    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2

    return (
        f"{NEW_CLASS_ID} "
        f"{x_center:.6f} {y_center:.6f} "
        f"{width:.6f} {height:.6f}"
    )


def main():
    if not SOURCE.exists():
        raise FileNotFoundError(f"Dataset not found: {SOURCE}")

    if OUTPUT.exists():
        raise FileExistsError(
            f"{OUTPUT} already exists. Rename or inspect it first; "
            "this script will not overwrite existing data."
        )

    total_images = 0
    kept_boxes = 0
    skipped_boxes = 0

    try:
        for split in ("train", "valid"):
            image_dir = SOURCE / split / "images"
            label_dir = SOURCE / split / "labels"

            if not image_dir.is_dir() or not label_dir.is_dir():
                raise FileNotFoundError(
                    f"Missing images or labels for split: {split}"
                )

            out_images = OUTPUT / "images" / split
            out_labels = OUTPUT / "labels" / split
            out_images.mkdir(parents=True)
            out_labels.mkdir(parents=True)

            images = sorted(
                p for p in image_dir.iterdir()
                if p.suffix.lower() in IMAGE_EXTS
            )

            for image_path in images:
                label_path = label_dir / f"{image_path.stem}.txt"

                if not label_path.exists():
                    raise FileNotFoundError(
                        f"Missing label for image: {image_path.name}"
                    )

                converted = []

                for line_number, line in enumerate(
                    label_path.read_text().splitlines(), start=1
                ):
                    if not line.strip():
                        continue

                    parts = line.split()

                    try:
                        class_id = int(parts[0])

                        if class_id != KEEP_CLASS_ID:
                            skipped_boxes += 1
                            continue

                        converted.append(convert_polygon_to_box(parts))
                        kept_boxes += 1

                    except (ValueError, IndexError) as exc:
                        raise ValueError(
                            f"Invalid annotation in {label_path.name}, "
                            f"line {line_number}: {exc}"
                        ) from exc

                shutil.copy2(image_path, out_images / image_path.name)
                (out_labels / f"{image_path.stem}.txt").write_text(
                    "\n".join(converted) + ("\n" if converted else "")
                )
                total_images += 1

        yaml_text = """path: .
train: images/train
val: images/valid

nc: 1
names: ['person']
"""
        (OUTPUT / "data.yaml").write_text(yaml_text)

    except Exception:
        shutil.rmtree(OUTPUT, ignore_errors=True)
        raise

    print("Conversion completed successfully.")
    print(f"Images copied: {total_images}")
    print(f"Bounding boxes kept: {kept_boxes}")
    print(f"Other-class annotations skipped: {skipped_boxes}")
    print(f"Output folder: {OUTPUT}")
    print(f"Config file: {OUTPUT / 'data.yaml'}")


if __name__ == "__main__":
    main()
