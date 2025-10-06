"""This module implements the ChessRecognitionDataset class and a YOLO-v11 export utility."""
import json
import os
import shutil
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple, Union

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.io import read_image


class ChessRecognitionDataset(Dataset):


    def __init__(
        self,
        dataroot: Union[str, Path],
        split: str,
        transform: Union[Callable, None] = None,
    ) -> None:

        super(ChessRecognitionDataset, self).__init__()

        self.dataroot = Path(dataroot)
        self.split = split
        self.transform = transform

        # Load annotations
        data_path = Path(dataroot, "annotations.json")
        if not data_path.is_file():
            raise FileNotFoundError(f"File '{data_path}' doesn't exist.")

        with open(data_path, "r") as f:
            annotations_file = json.load(f)

        # Load tables
        self.annotations = pd.DataFrame(
            annotations_file["annotations"]["pieces"], index=None
        )
        self.categories = pd.DataFrame(annotations_file["categories"], index=None)
        self.images = pd.DataFrame(annotations_file["images"], index=None)

        # Get split info
        self.length = annotations_file["splits"][split]["n_samples"]
        self.split_img_ids = annotations_file["splits"][split]["image_ids"]

        # Keep only the split's data
        self.annotations = self.annotations[
            self.annotations["image_id"].isin(self.split_img_ids)
        ]
        self.images = self.images[self.images["id"].isin(self.split_img_ids)]

        assert (self.length == len(self.split_img_ids) and self.length == len(self.images)), (
            f"The numeber of images in the dataset ({len(self.images)}) for split:{self.split}, does "
            f"not match neither the length specified in the annotations ({self.length}) or the length of "
            f"the list of ids for the split {len(self.split_img_ids)}"
        )

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return self.length

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:

        # LOAD IMAGE
        img_id = self.split_img_ids[index]
        img_path = Path(
            self.dataroot,
            self.images[self.images["id"] == img_id].path.values[0],
        )

        img = read_image(str(img_path)).float()

        if self.transform is not None:
            img = self.transform(img)

        # GET ANNOTATIONS
        cols = "abcdefgh"
        rows = "87654321"

        empty_cat_id = int(
            self.categories[self.categories["name"] == "empty"].id.values[0]
        )

        img_anns = self.annotations[self.annotations["image_id"] == img_id].copy()

        # Convert chessboard positions to 64x1 array indexes
        img_anns["array_pos"] = img_anns["chessboard_position"].map(
            lambda x: 8 * rows.index(x[1]) + cols.index(x[0])
        )

        # Keep columns of interest
        img_anns = pd.DataFrame(img_anns["category_id"]).set_index(
            img_anns["array_pos"]
        )

        # Add category_id for 'empty' in missing row indexes and create tensor
        img_anns = torch.tensor(
            list(
                img_anns.reindex(range(64), fill_value=empty_cat_id)[
                    "category_id"
                ].values
            )
        )

        img_anns = F.one_hot(img_anns)
        img_anns = img_anns.flatten().float()

        return (img, img_anns)

    @staticmethod
    def export_yolo_v11(
        dataroot: Union[str, Path],
        out_root: Union[str, Path],
        corners_size: float,
        splits: Iterable[str] = ("train", "valid", "test"),
        use_symlinks: bool = False,
        include_empty_class: bool = False,
        keep_images_without_boxes: bool = False,  # <-- NEW
        only_corners: bool = False,
    ) -> None:

        import math
        dataroot = Path(dataroot)
        out_root = Path(out_root)
        out_root.mkdir(parents=True, exist_ok=True)

        # --- Load annotations file
        ann_path = dataroot / "annotations.json"
        if not ann_path.is_file():
            raise FileNotFoundError(f"annotations.json not found at: {ann_path}")

        with open(ann_path, "r") as f:
            data = json.load(f)

        # --- Build DataFrames
        df_images = pd.DataFrame(data["images"])
        if not {"id", "path"}.issubset(df_images.columns):
            raise ValueError("`images` entries must include at least 'id' and 'path'")

        df_anns = pd.DataFrame(data["annotations"]["pieces"])
        if not {"image_id", "category_id", "bbox"}.issubset(df_anns.columns):
            raise ValueError("`annotations['pieces']` must include 'image_id', 'category_id', and 'bbox'")

        df_categories = pd.DataFrame(data["categories"])

        df_cat = pd.DataFrame(data["annotations"]["corners"])
        if not {"id", "name"}.issubset(df_categories.columns):
            raise ValueError("`categories` must include 'id' and 'name'")
        
        # --- Category mapping (YOLO ids 0..C-1); optionally drop 'empty'
        if not include_empty_class and "empty" in set(df_categories["name"]):
            df_categories = df_categories[df_categories["name"] != "empty"].copy()

        df_categories = df_categories.reset_index(drop=True)
        yolo_names: List[str] = df_categories["name"].tolist()
        catid_to_yolo: Dict[int, int] = {int(row.id): int(i) for i, row in df_categories.iterrows()}

        # Keep only annotations for included categories
        df_anns = df_anns[df_anns["category_id"].isin(catid_to_yolo.keys())].copy()

        # --- Helpers
        def parse_bbox(b) -> Union[List[float], None]:
            """Normalize bbox to [x, y, w, h] floats. Return None if invalid."""
            if b is None:
                return None
            if isinstance(b, float):
                if math.isnan(b) or not math.isfinite(b):
                    return None
                return None  # bare float is not a bbox
            if isinstance(b, (list, tuple)):
                if len(b) != 4:
                    return None
                try:
                    return [float(b[0]), float(b[1]), float(b[2]), float(b[3])]
                except Exception:
                    return None
            if isinstance(b, str):
                s = b.strip()
                try:
                    j = json.loads(s)
                    return parse_bbox(j)
                except Exception:
                    s = s.strip("[]")
                    parts = [p.strip() for p in s.split(",")]
                    if len(parts) != 4:
                        return None
                    try:
                        return [float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])]
                    except Exception:
                        return None
            if isinstance(b, dict):
                if all(k in b for k in ("x", "y", "w", "h")):
                    try:
                        return [float(b["x"]), float(b["y"]), float(b["w"]), float(b["h"])]
                    except Exception:
                        return None
                if all(k in b for k in ("left", "top", "width", "height")):
                    try:
                        return [float(b["left"]), float(b["top"]), float(b["width"]), float(b["height"])]
                    except Exception:
                        return None
                return None
            return None

        def coco_bbox_to_yolo(bbox: List[float], img_w: int, img_h: int) -> Tuple[float, float, float, float]:
            """Convert [x,y,w,h] (pixels) to normalized YOLO (xc,yc,w,h)."""
            x, y, w, h = bbox
            xc = (x + w / 2.0) / float(img_w)
            yc = (y + h / 2.0) / float(img_h)
            ww = w / float(img_w)
            hh = h / float(img_h)

            def cl(v: float) -> float:
                return max(0.0, min(1.0, v))

            return cl(xc), cl(yc), cl(ww), cl(hh)

        # --- Resolve split keys (accept "valid" or "val")
        splits_available = set(data.get("splits", {}).keys())

        def resolve_split_key(s: str) -> str:
            if s in splits_available:
                return s
            if s == "valid" and "val" in splits_available:
                return "val"
            if s == "val" and "valid" in splits_available:
                return "valid"
            raise KeyError(f"Split '{s}' not found in annotations. Available: {sorted(splits_available)}")
        
        def point_to_bbox(point: list, diam: float):
            return [point[0]-diam/2, point[1]-diam/2, diam, diam]
        # --- Counters
        total_images_seen = 0
        kept_images = 0
        skipped_images_no_boxes = 0
        total_boxes = 0
        skipped_boxes = 0

        # --- Export
        for s in splits:
            s_resolved = resolve_split_key(s)
            (out_root / s / "images").mkdir(parents=True, exist_ok=True)
            (out_root / s / "labels").mkdir(parents=True, exist_ok=True)

            image_ids: List[int] = list(data["splits"][s_resolved]["image_ids"])
            df_split_imgs = df_images[df_images["id"].isin(image_ids)].copy()

            # Pre-group ONLY valid-parsed bboxes by image
            anns_by_img: Dict[int, List[Tuple[int, List[float]]]] = {img_id: [] for img_id in image_ids}
            subset = df_anns[df_anns["image_id"].isin(image_ids)]
            corners_by_img : Dict[int, List[List[float]]] = {img_id: [] for img_id in image_ids}
            subset_corners = df_cat[df_cat["image_id"].isin(image_ids)]
            for _, row in subset.iterrows():
                parsed = parse_bbox(row.bbox)
                if parsed is None:
                    skipped_boxes += 1
                    continue
                anns_by_img[int(row.image_id)].append((int(row.category_id), parsed))
                total_boxes += 1
            
            for _, row in subset_corners.iterrows():
                corners = [row.corners['bottom_right'], row.corners['bottom_left'], row.corners['top_left'], row.corners['top_right']]
                for corner in corners:  
                    corners_by_img[int(row.image_id)].append(corner)

            for _, img_row in df_split_imgs.iterrows():
                total_images_seen += 1
                img_id = int(img_row.id)
                src_rel = Path(str(img_row.path))
                src_abs = dataroot / src_rel
                if not src_abs.is_file():
                    raise FileNotFoundError(f"Image not found: {src_abs}")

                ann_list = anns_by_img.get(img_id, [])
                corners_list = corners_by_img.get(img_id, [])
                # Fast-path: no annotations at all
                if len(ann_list) == 0 and not keep_images_without_boxes:
                    skipped_images_no_boxes += 1
                    continue
                    
                # Determine image size (only if needed beyond here)
                if {"width", "height"}.issubset(df_split_imgs.columns):
                    try:
                        img_w = int(img_row.width)
                        img_h = int(img_row.height)
                    except Exception:
                        tensor = read_image(str(src_abs))
                        img_h, img_w = int(tensor.shape[-2]), int(tensor.shape[-1])
                else:
                    tensor = read_image(str(src_abs))
                    img_h, img_w = int(tensor.shape[-2]), int(tensor.shape[-1])

                # Build YOLO lines from parsed boxes
                lines: List[str] = []
                if not only_corners:
                    for cat_id, bbox in ann_list:
                        yolo_cls = catid_to_yolo[int(cat_id)]
                        xc, yc, ww, hh = coco_bbox_to_yolo(bbox, img_w, img_h)
                        if ww <= 0.0 or hh <= 0.0:
                            skipped_boxes += 1
                            continue
                        lines.append(f"{yolo_cls} {xc:.6f} {yc:.6f} {ww:.6f} {hh:.6f}")
            
                for corner in corners_list:
                    bbox = point_to_bbox(corner, corners_size)
                    xc, yc, ww, hh = coco_bbox_to_yolo(bbox, img_w, img_h)
                    lines.append(f"12 {xc:.6f} {yc:.6f} {ww:.6f} {hh:.6f}")

                # If no VALID boxes after filtering
                if len(lines) == 0 and not keep_images_without_boxes:
                    skipped_images_no_boxes += 1
                    continue

                # Unique filenames to avoid collisions
                stem = src_rel.stem
                dst_img_name = f"{stem}_{img_id}{src_rel.suffix}"
                dst_lbl_name = f"{stem}_{img_id}.txt"

                dst_img = out_root / s / "images" / dst_img_name
                dst_lbl = out_root / s / "labels" / dst_lbl_name

                # Copy or symlink image
                if use_symlinks:
                    try:
                        if dst_img.exists() or dst_img.is_symlink():
                            dst_img.unlink()
                        os.symlink(src_abs, dst_img)
                    except (NotImplementedError, OSError):
                        shutil.copy2(src_abs, dst_img)
                else:
                    shutil.copy2(src_abs, dst_img)

                # Write label file (empty if keep_images_without_boxes=True and no boxes)
                with open(dst_lbl, "w") as f:
                    if lines:
                        f.write("\n".join(lines))

                kept_images += 1

        # --- Write data.yaml
        yaml_text = (
            f"path: {out_root.resolve()}\n"
            f"train: train/images\n"
            f"val: valid/images\n"
            f"test: test/images\n"
            f"names:\n"
        )
        for i, name in enumerate(yolo_names):
            yaml_text += f"  {i}: {name}\n"

        with open(out_root / "data.yaml", "w") as f:
            f.write(yaml_text)

        print(
            f"YOLO v11 export complete at: {out_root.resolve()}\n"
            f"Classes ({len(yolo_names)}): {yolo_names}\n"
            f"Images seen: {total_images_seen}, kept: {kept_images}, "
            f"skipped (no boxes): {skipped_images_no_boxes}\n"
            f"Boxes kept: {total_boxes - skipped_boxes}, skipped (invalid/degenerate): {skipped_boxes}"
        )
    @staticmethod
    def visualize_yolo_labels(
        yolo_root: Union[str, Path],
        out_root: Union[str, Path] = None,
        splits: Iterable[str] = ("train", "valid", "test"),
        include_images_without_boxes: bool = False,
        line_width: int = 2,
        save_as_png: bool = True,
    ) -> None:
        # Visualize YOLO label files by drawing boxes on images and saving them
        # to a separate folder.

        import re
        from torchvision.io import read_image, write_png
        from torchvision.utils import draw_bounding_boxes

        yolo_root = Path(yolo_root)
        if out_root is None:
            out_root = yolo_root / "visualizations"
        out_root = Path(out_root)
        out_root.mkdir(parents=True, exist_ok=True)

        # Try to read class names from data.yaml (optional).
        names_map = {}  # id -> name
        data_yaml = yolo_root / "data.yaml"
        if data_yaml.is_file():
            try:
                # Minimal parser for the "names:" block: lines like "  0: pawn"
                in_names = False
                for line in data_yaml.read_text(encoding="utf-8").splitlines():
                    if re.match(r"^\s*names\s*:\s*$", line):
                        in_names = True
                        continue
                    if in_names:
                        if re.match(r"^\s*\w+\s*:", line) and not re.match(r"^\s*\d+\s*:", line):
                            # Another top-level-ish key (path/train/val/test/etc.) → stop names block
                            break
                        m = re.match(r"^\s*(\d+)\s*:\s*(.+?)\s*$", line)
                        if m:
                            idx = int(m.group(1))
                            names_map[idx] = m.group(2)
            except Exception:
                names_map = {}

        # Simple color palette (cycled by class id)
        palette = [
            "red", "green", "blue", "yellow", "magenta", "cyan",
            "white", "orange", "pink", "purple", "brown", "olive",
        ]

        # Helpers
        def find_image_for_label(images_dir: Path, stem: str) -> Union[Path, None]:
            # Try exact matches with any extension in images_dir
            candidates = list(images_dir.glob(stem + ".*"))
            return candidates[0] if candidates else None

        def yolo_to_xyxy(xc: float, yc: float, w: float, h: float, W: int, H: int) -> Tuple[float, float, float, float]:
            x1 = (xc - w / 2.0) * W
            y1 = (yc - h / 2.0) * H
            x2 = (xc + w / 2.0) * W
            y2 = (yc + h / 2.0) * H
            # clamp
            x1 = max(0.0, min(float(W - 1), x1))
            y1 = max(0.0, min(float(H - 1), y1))
            x2 = max(0.0, min(float(W - 1), x2))
            y2 = max(0.0, min(float(H - 1), y2))
            return x1, y1, x2, y2

        # Stats
        images_processed = 0
        images_saved = 0
        boxes_drawn = 0
        label_files_missing_images = 0

        for split in splits:
            labels_dir = yolo_root / split / "labels"
            images_dir = yolo_root / split / "images"
            if not labels_dir.is_dir() or not images_dir.is_dir():
                print(f"[WARN] Missing split folders for '{split}' — skipping.")
                continue

            out_split = out_root / split
            out_split.mkdir(parents=True, exist_ok=True)

            for lbl_path in labels_dir.glob("*.txt"):
                images_processed += 1
                stem = lbl_path.stem
                img_path = find_image_for_label(images_dir, stem)
                if img_path is None:
                    label_files_missing_images += 1
                    continue

                # Read label lines
                lines = [ln.strip() for ln in lbl_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
                if len(lines) == 0 and not include_images_without_boxes:
                    # Nothing to draw; skip unless explicitly keeping empty ones
                    continue

                # Load image tensor (C,H,W) uint8
                img = read_image(str(img_path))
                H, W = int(img.shape[1]), int(img.shape[2])

                # Parse boxes
                xyxy = []
                labels = []
                colors = []
                for ln in lines:
                    parts = ln.split()
                    if len(parts) < 5:
                        continue
                    try:
                        cls = int(float(parts[0]))
                        xc, yc, w, h = map(float, parts[1:5])
                    except Exception:
                        continue
                    # skip non-positive
                    if w <= 0.0 or h <= 0.0:
                        continue
                    x1, y1, x2, y2 = yolo_to_xyxy(xc, yc, w, h, W, H)
                    if x2 <= x1 or y2 <= y1:
                        continue
                    xyxy.append([x1, y1, x2, y2])
                    # Build label string
                    if cls in names_map:
                        labels.append(f"{names_map[cls]} ({cls})")
                    else:
                        labels.append(f"class {cls}")
                    colors.append(palette[cls % len(palette)])

                if len(xyxy) == 0 and not include_images_without_boxes:
                    continue

                if len(xyxy) > 0:
                    boxes_tensor = torch.tensor(xyxy, dtype=torch.float32)
                    img_drawn = draw_bounding_boxes(
                        img,
                        boxes=boxes_tensor,
                        labels=labels if len(labels) == len(xyxy) else None,
                        colors=colors if len(colors) == len(xyxy) else None,
                        width=int(line_width),
                        font_size=2,  # auto
                    )
                    boxes_drawn += len(xyxy)
                else:
                    # No boxes, but requested to include the image as-is
                    img_drawn = img

                # Save
                if save_as_png:
                    out_file = out_split / f"{stem}.png"
                    write_png(img_drawn, str(out_file))
                else:
                    out_ext = img_path.suffix if img_path.suffix else ".png"
                    out_file = out_split / f"{stem}{out_ext}"
                    if out_file.suffix.lower() == ".png":
                        write_png(img_drawn, str(out_file))
                    else:
                        # Fallback: if non-png, convert via PIL to preserve ext
                        from PIL import Image
                        pil = Image.fromarray(img_drawn.permute(1, 2, 0).cpu().numpy())
                        pil.save(out_file)

                images_saved += 1

        print(
            f"Visualization complete at: {out_root.resolve()}\n"
            f"Images processed: {images_processed}, saved: {images_saved}, "
            f"boxes drawn: {boxes_drawn}, labels missing images: {label_files_missing_images}"
        )
