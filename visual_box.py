import os
from pathlib import Path
import cv2

# --- CONFIG ---
images_dir = Path("test\images")        # folder with images (e.g., .jpg/.png)
labels_dir = Path("test\labels")        # folder with YOLO .txt label files (same base names as images)
out_dir    = Path("viz")           # where to save visualizations
out_dir.mkdir(parents=True, exist_ok=True)

# If you want human-readable names per class id, fill this list:
CLASS_NAMES = ["corners", "eights", "ones"]  # change to your classes
# If you'd like consistent colors per class:
PALETTE = [(255, 0, 0), (0, 200, 0), (0, 165, 255), (255, 0, 255), (0, 255, 255)]

def yolo_to_bbox(xc, yc, w, h, img_w, img_h):
    """
    Convert normalized YOLO (xc, yc, w, h) to pixel (x1, y1, x2, y2).
    Clips to image bounds.
    """
    x1 = int((xc - w/2) * img_w)
    y1 = int((yc - h/2) * img_h)
    x2 = int((xc + w/2) * img_w)
    y2 = int((yc + h/2) * img_h)

    # clip
    x1 = max(0, min(x1, img_w - 1))
    y1 = max(0, min(y1, img_h - 1))
    x2 = max(0, min(x2, img_w - 1))
    y2 = max(0, min(y2, img_h - 1))
    return x1, y1, x2, y2

def draw_boxes_on_image(img_path, label_path, out_path):
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"⚠️ Could not read image: {img_path}")
        return

    H, W = img.shape[:2]

    if not label_path.exists():
        print(f"ℹ️ No label file for {img_path.name}, saving image as-is.")
        cv2.imwrite(str(out_path), img)
        return

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls_id = int(parts[0])
            xc, yc, w, h = map(float, parts[1:])

            x1, y1, x2, y2 = yolo_to_bbox(xc, yc, w, h, W, H)

            color = PALETTE[cls_id % len(PALETTE)]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            # label text
            label = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id)
            txt = f"{label} {w*W:.0f}x{h*H:.0f}"
            # text background
            (tw, th), baseline = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            ty = max(0, y1 - th - 4)
            cv2.rectangle(img, (x1, ty), (x1 + tw + 4, ty + th + baseline + 4), color, -1)
            cv2.putText(img, txt, (x1 + 2, ty + th + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

    cv2.imwrite(str(out_path), img)

def find_label_for_image(img_path):
    # match same basename but .txt in labels_dir
    return labels_dir / (img_path.stem + ".txt")

# Process all images
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
for img_path in images_dir.iterdir():
    if img_path.suffix.lower() in SUPPORTED_EXTS:
        label_path = find_label_for_image(img_path)
        out_path = out_dir / img_path.name
        draw_boxes_on_image(img_path, label_path, out_path)
        print(f"✅ Saved: {out_path.name}")
