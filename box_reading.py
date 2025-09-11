import argparse
import json
from pathlib import Path
import pandas as pd
from PIL import Image
import shutil
import yaml


def _is_normalized_bbox(bbox):
    """Return True if bbox values appear to be normalized (all <= 1)."""
    try:
        return all(float(v) <= 1.0 for v in bbox)
    except Exception:
        return False


def export_yolo_labels(image_path: Path, bboxes: list, label_output_dir: Path, catid_to_classidx: dict):
    """Export bounding boxes to YOLO v11 label format.

    Args:
        image_path: Path to the image (used to read size).
        bboxes: list of annotation dicts with keys: bbox (x,y,w,h), category_id.
        label_output_dir: directory where .txt label file will be saved.
        catid_to_classidx: mapping from original category_id -> contiguous class index.
    """
    img = Image.open(image_path)
    W, H = img.size

    label_output_dir.mkdir(parents=True, exist_ok=True)
    label_path = label_output_dir / (Path(image_path).stem + ".txt")

    with open(label_path, "w") as f:
        for b in bboxes:
            bbox = b["bbox"]
            # handle normalized bboxes (0..1) or absolute pixels
            if _is_normalized_bbox(bbox):
                x = float(bbox[0]) * W
                y = float(bbox[1]) * H
                w = float(bbox[2]) * W
                h = float(bbox[3]) * H
            else:
                x, y, w, h = map(float, bbox)

            # Normalize to YOLO format
            x_center = (x + w / 2) / W
            y_center = (y + h / 2) / H
            w_norm = w / W
            h_norm = h / H

            orig_cat_id = int(b["category_id"])
            class_id = catid_to_classidx[orig_cat_id]
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")


def convert_dataset_existing_splits(dataroot: Path, out_root: Path):
    """
    Use the existing train/val/test folders in ChessReD to create YOLO format dataset.

    Output layout:
        out_root/
            classes.txt
            data.yaml
            train/
                images/
                labels/
            val/
                images/
                labels/
            test/
                images/
                labels/
    """
    with open(dataroot / "annotations.json", "r") as f:
        data = json.load(f)

    pieces = pd.DataFrame(data["annotations"]["pieces"])
    categories = pd.DataFrame(data["categories"])
    images = pd.DataFrame(data["images"])

    # Build category id -> class idx mapping
    categories_sorted = categories.sort_values("id").reset_index(drop=True)
    cat_id_list = [int(r["id"]) for _, r in categories_sorted.iterrows()]
    catid_to_classidx = {cid: idx for idx, cid in enumerate(cat_id_list)}
    class_names = [r["name"] for _, r in categories_sorted.iterrows()]

    # Write classes.txt
    out_root.mkdir(parents=True, exist_ok=True)
    classes_txt_path = out_root / "classes.txt"
    classes_txt_path.write_text("\n".join(class_names), encoding="utf-8")

    # Build quick lookup maps for images
    # Map by file name and by stem for robustness
    path_to_id = {}
    stem_to_id = {}
    for _, r in images.iterrows():
        img_rel = Path(r["path"])  # path in annotations.json
        path_to_id[img_rel.name] = int(r["id"])
        stem_to_id[img_rel.stem] = int(r["id"])

    splits = ["train", "val", "test"]
    for split in splits:
        img_dir = dataroot / split / "images"
        lbl_dir = out_root / split / "labels"
        lbl_dir.mkdir(parents=True, exist_ok=True)

        if not img_dir.exists():
            print(f"Warning: expected directory {img_dir} does not exist — skipping split '{split}'.")
            continue

        for img_path in img_dir.glob("*"):
            if not img_path.is_file():
                continue

            fname = img_path.name
            stem = img_path.stem

            image_id = None
            # try exact filename match
            if fname in path_to_id:
                image_id = path_to_id[fname]
            # try stem match
            elif stem in stem_to_id:
                image_id = stem_to_id[stem]
            else:
                # try to find by matching end of stored path (in case of subfolders)
                matches = images[images["path"].apply(lambda p: Path(p).name == fname)]
                if len(matches) == 1:
                    image_id = int(matches.iloc[0]["id"])

            if image_id is None:
                print(f"Warning: could not determine image_id for file {img_path}. Skipping.")
                continue

            ann_for_image = pieces[pieces["image_id"] == image_id]

            boxes = []
            for _, row in ann_for_image.iterrows():
                boxes.append({
                    "piece": row.get("name"),
                    "bbox": row["bbox"],
                    "category_id": row["category_id"],
                })

            export_yolo_labels(img_path, boxes, lbl_dir, catid_to_classidx)

    # Create data.yaml with relative paths
    data_yaml = {
        "path": "./",
        "train": str(Path("train/images")),
        "val": str(Path("val/images")),
        "test": str(Path("test/images")),
        "nc": len(class_names),
        "names": class_names,
    }

    data_yaml_path = out_root / "data.yaml"
    with open(data_yaml_path, "w") as f:
        yaml.dump(data_yaml, f, sort_keys=False)

    print(f"YOLO dataset created under {out_root.resolve()}")
    print(f"classes.txt written to {classes_txt_path.resolve()}")
    print(f"data.yaml written to {data_yaml_path.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="Convert ChessReD existing train/val/test splits to YOLO format")
    parser.add_argument("--dataroot", default="", help="Path to ChessReD data root")
    parser.add_argument("--out", default="yolo_dataset", help="Output root directory")
    args = parser.parse_args()

    dataroot = Path(args.dataroot)
    out_root = Path(args.out)

    convert_dataset_existing_splits(dataroot, out_root)


if __name__ == "__main__":
    main()