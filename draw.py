import cv2
from detect import *
from torchvision.utils import draw_bounding_boxes
import torch
def detect_and_crop(path_to_img):
    ones, eights = detect_corners_and_orientation(path_to_img)
    _, _, corners = detect_pieces_and_corners(cv2.imread(path_to_img), 0.2, 0.5)
    warped = warp_quad(path_to_img, corners, ones, eights, False)
    cv2.namedWindow("Board", cv2.WINDOW_NORMAL)  # or: cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO
    cv2.imshow("Board", warped)

    cv2.resizeWindow("Board", 800, 800)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def draw_corners(path_to_img, corners, ones, eights):
    img = cv2.imread(path_to_img)
    for corner in corners:
        #cv2.putText(img, str(i), (int(corners[i][0]), int(corners[i][1])), 
        #cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 0, 0), 5)
        cv2.rectangle(img, (int(corner[0]), int(corner[3])), (int(corner[2]), int(corner[1])) ,color=(255, 0, 0), thickness= 5)

    for eight in eights:
        cv2.circle(img, eight, 15, (0, 255, 0), -1)
    for one in ones:    
        cv2.circle(img, one, 15, (0, 0, 255), -1)
    return img


def draw_corners_from_path(path_to_img):
    ones, eights = detect_corners_and_orientation(path_to_img)
    _, _, _, corners = detect_pieces_and_corners(cv2.imread(path_to_img), 0.2, 0.3)
    #corners = order_corners(corners, ones, eights)
    img = draw_corners(path_to_img, corners, ones, eights)
    cv2.namedWindow("Board", cv2.WINDOW_NORMAL)  
    cv2.imshow("Board", img)

    cv2.resizeWindow("Board", 800, 800)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def draw_pieces_boxes(path_to_img):
    corners, ones, eights = detect_corners_and_orientation(path_to_img)
    corners = order_corners(corners, ones, eights)
    img, (bl_y, bl_x), cropped_shape = crop_rectangle(path_to_img, corners)
    pieces_coords = detect_pieces(img)
    for i in range(12):
        for piece_coord in pieces_coords[i]:
            x1, y1, x2, y2 = piece_coord
            cv2.rectangle(img=img, pt1=(int(x1), int(y2)), pt2=(int(x2), int(y1)), color = (255, 0, 0), thickness=-1)
    cv2.namedWindow("Board", cv2.WINDOW_NORMAL)  
    cv2.imshow("Board", img)

    cv2.resizeWindow("Board", 800, 800)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def draw_origin_point(path_to_img):
    img = cv2.imread(path_to_img)
    corners, ones, eights = detect_corners_and_orientation(path_to_img)
    corners = order_corners(corners, ones, eights)
    cropped_img, (bl_y, bl_x), cropped_shape = crop_rectangle(path_to_img, corners)
    pieces_coords = detect_pieces(cropped_img)
    for i in range(12):
        for piece_coord in pieces_coords[i]:
            x1, y1, x2, y2 = piece_coord
            base_midpoint = [y2, (x1+x2) // 2]
            origin_point = get_origin_point(base_midpoint, img.shape, bl_y, bl_x, cropped_shape)
            cv2.circle(img=img, center= (int(origin_point[0]), int(origin_point[1])), radius = 15, color = (255, 0, 0), thickness=-1)
    cv2.namedWindow("Board", cv2.WINDOW_NORMAL)  
    cv2.imshow("Board", img)

    cv2.resizeWindow("Board", 800, 800)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

import cv2

def draw_grid(img, rows=8, cols=8, color=(0, 255, 0), thickness=3, line_type=cv2.LINE_AA):
    """
    Draws grid lines on a copy of `img` that divide it into `rows` x `cols` equal parts.
    By default, makes an 8x8 grid.

    Parameters:
        img:        BGR image (numpy array)
        rows:       number of horizontal parts
        cols:       number of vertical parts
        color:      BGR tuple for line color
        thickness:  line thickness in pixels
        line_type:  cv2 line type (e.g., cv2.LINE_AA)

    Returns:
        A new image with grid lines drawn.
    """
    out = img.copy()
    h, w = out.shape[:2]

    # Draw horizontal divider lines (exclude borders at y=0 and y=h)
    for r in range(1, rows):
        y = round(r * h / rows)
        cv2.line(out, (0, y), (w - 1, y), color, thickness, line_type)

    # Draw vertical divider lines (exclude borders at x=0 and x=w)
    for c in range(1, cols):
        x = round(c * w / cols)
        cv2.line(out, (x, 0), (x, h - 1), color, thickness, line_type)

    return out

# Example usage:
# img = cv2.imread("input.jpg")
# grid_img = draw_grid(img, rows=8, cols=8, color=(0, 255, 0), thickness=2)
# cv2.imwrite("with_grid.jpg", grid_img)
# cv2.imshow("grid", grid_img); cv2.waitKey(0); cv2.destroyAllWindows()

def draw_warped_point(path_to_img):
    img = cv2.imread(path_to_img)
    ones, eights = detect_corners_and_orientation(path_to_img)
    pieces_coords, _ , corners = detect_pieces_and_corners(img, 0.2, 0.3)
    corners = order_corners(corners, ones, eights)
    warped = warp_quad(path_to_img, corners, ones, eights, True)
    out = draw_grid(warped)
    for i in range(12):
        for piece_coord in pieces_coords[i]:
            base_midpoint = get_base_midpoint(piece_coord)
            warped_point = get_point_in_board(base_midpoint, img.shape, corners)
            cv2.circle(img=out, center= (int(warped_point[0]), int(warped_point[1])), radius = 15, color = (255, 0, 0), thickness=-1)
    cv2.namedWindow("Board", cv2.WINDOW_NORMAL)  
    cv2.imshow("Board", out)

    cv2.resizeWindow("Board", 800, 800)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def save_warped_files(folder_num, folder_name):
    os.makedirs(folder_name, exist_ok = True)
    folder_path = create_folder_path(folder_num)
    files = os.listdir(folder_path)
    for i in range(len(files)):
        path_to_img = create_file_path(i, folder_num)
        ones, eights = detect_corners_and_orientation(path_to_img)
        _, _, corners = detect_pieces_and_corners(cv2.imread(path_to_img), 0.2, 0.3)
        if len(corners)>=4:
            warped = warp_quad(path_to_img, corners, ones, eights, flag = False)
            save_dir = os.path.join(folder_name, str(folder_num))
            os.makedirs(save_dir, exist_ok = True)
            saved_img_path = os.path.join(save_dir, f"{i}.png")
            cv2.imwrite(saved_img_path, warped)
            

def save_cropped_files(folder_num, folder_name):
    os.makedirs(folder_name, exist_ok = True)
    folder_path = create_folder_path(folder_num)
    files = os.listdir(folder_path)
    for i in range(len(files)):
        path_to_img = create_file_path(i, folder_num)
        corners, _, _ = detect_corners_and_orientation(path_to_img)
        if len(corners)>=4:
            cropped, _, _ = crop_rectangle(path_to_img, corners, margin = 220)
            save_dir = os.path.join(folder_name, str(folder_num))
            os.makedirs(save_dir, exist_ok = True)
            saved_img_path = os.path.join(save_dir, f"{folder_num}_{i}.png")
            cv2.imwrite(saved_img_path, cropped)
        

def warp_quad(path_to_img, quad_pts, ones, eights, flag, scale=1.0, margin_ratio=0, margin=0):
    """
    Perspective-warp the quad to a top-down rectangle.
    Output size inferred from the quad side lengths, with optional margin.
    Returns warped
    """
    if(flag): 
        (tl, tr, br, bl) = quad_pts
    else:
        ordered = order_corners(quad_pts, ones, eights)
        (tl, tr, br, bl) = ordered

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)

    maxW = int(math.ceil(max(widthA, widthB) * scale * (1.0 + margin_ratio)))
    maxH = int(math.ceil(max(heightA, heightB) * scale * (1.0 + margin_ratio)))
    maxW = max(maxW, 10)
    maxH = max(maxH, 10)
    
    dst = np.array([
        [margin, margin],
        [maxW - margin - 1, margin],
        [maxW - margin - 1, maxH - margin - 1],
        [margin, maxH - margin - 1]
    ], dtype=np.float32)   

    img_bgr = cv2.imread(path_to_img)
    if(flag):
        M = cv2.getPerspectiveTransform(quad_pts, dst)
    else:
        M = cv2.getPerspectiveTransform(ordered, dst)
    warped = cv2.warpPerspective(img_bgr, M, (maxW, maxH))
    h, w = warped.shape[:2]
    size = max(h, w)
    #square_img = cv2.resize(warped, (size, size))
    return warped

def draw_detected_boxes_and_classes(img, boxes, boxes_names):
    for box in boxes:
        coords = box.xyxy.cpu().numpy()
        cls = box.cls.cpu().numpy()
        box_name = boxes_names[cls]
        cv2.rectangle(img, (int(coords[0]), int(coords[3])), (int(coords[2]), int(coords[1])) ,color=(255, 0, 0), thickness= -1)
        cv2.putText(img, box_name, (int(coords[2]), int(coords[1])), color= (0, 0, 0), thickness=-1)
    cv2.namedWindow("Board", cv2.WINDOW_NORMAL)  
    cv2.imshow("Board", img)

    cv2.resizeWindow("Board", 800, 800)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def draw_corners_with_names(path_to_img):
    img = cv2.imread(path_to_img)
    results = model.predict(source=path_to_img, conf = 0, iou = 0.0, agnostic_nms = False)
    r = results[0]
    boxes = r.boxes
    # Sorting the values of the boxes by confidence using torch
    conf_vals = boxes.conf.view(-1)
    order = torch.argsort(conf_vals, descending = True)
    boxes_sorted = boxes[order]
    class_ids = boxes_sorted.cls.cpu().numpy()
    coords = boxes_sorted.xyxy.cpu().numpy()
    conf = boxes_sorted.conf.cpu().numpy()
    print(conf[class_ids == 0][:2])
    print(conf[class_ids == 1][:2])
    # get the boxes coordinates of the corners, the 1s and the 8s rows
    eights_coords = coords[class_ids == 0]
    ones_coords = coords[class_ids == 1]
    if len(eights_coords) >= 2:
        eights_coords = eights_coords[:2]
    if len(ones_coords) >= 2:
        ones_coords = ones_coords[:2]
    for eight in eights_coords:
        x1, y1, x2, y2 = eight
        cv2.rectangle(img=img, pt1=(int(x1), int(y2)), pt2=(int(x2), int(y1)), color = (255, 0, 0), thickness=5)
    for one in ones_coords:
        x1, y1, x2, y2 = one
        cv2.rectangle(img=img, pt1=(int(x1), int(y2)), pt2=(int(x2), int(y1)), color = (0, 255, 0), thickness=5)
    cv2.namedWindow("Board", cv2.WINDOW_NORMAL)  
    cv2.imshow("Board", img)

    cv2.resizeWindow("Board", 800, 800)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def draw_piece_detections(
    img_input,
    conf_score=0.25,
    iou_score=0.45,
    display_size=800,          # final shown board size (pixels)
    text_scale=0.30,           # much smaller labels
    text_thickness=1,          # thin text
    box_thickness=2,           # thin boxes
    save_path=None,
    show=False,
):
    cls_to_piece_type = {
        0:"white-pawn", 1:"white-rook", 2:"white-knight", 3:"white-bishop",
        4:"white-queen",5:"white-king", 6:"black-pawn",   7:"black-rook",
        8:"black-knight",9:"black-bishop",10:"black-queen",11:"black-king", 12: "corner"
    }

    # Load image
    if isinstance(img_input, str):
        img = cv2.imread(img_input)
        if img is None:
            raise ValueError(f"Could not read image from path: {img_input}")
    else:
        img = img_input.copy()

    H, W = img.shape[:2]

    # Run detection on original image
    results = pieces_model.predict(
        source=img, conf=conf_score, iou=iou_score, agnostic_nms=False, verbose=False
    )
    r = results[0]
    boxes = r.boxes

    # Prepare 800x800 canvas for drawing
    disp = cv2.resize(img, (display_size, display_size), interpolation=cv2.INTER_LINEAR)
    sx = display_size / float(W)
    sy = display_size / float(H)

    detections = []
    if boxes is not None and len(boxes) > 0:
        xyxy = boxes.xyxy.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy().astype(int)
        confs  = boxes.conf.cpu().numpy()

        font = cv2.FONT_HERSHEY_SIMPLEX

        def color_for(cid: int):
            class_to_color = {0: (0, 0, 255), 1: (255, 128, 0), 2: (255, 255, 0), 3: (128, 255, 0), 
                              4: (0, 255, 255), 5: (0, 0, 255), 6: (255, 255, 255), 7: (255, 0, 255), 8: (128, 128, 128), 
                              9: (0, 0, 0), 10: (127, 0, 255), 11: (255, 0, 127), 12: (153, 0, 0)}
            return class_to_color[cid]

        for (x1, y1, x2, y2), cid, conf in zip(xyxy, cls_ids, confs):
            if(cid == 12): 
                continue
            # scale box to 800x800 canvas
            X1 = int(round(x1 * sx)); Y1 = int(round(y1 * sy))
            X2 = int(round(x2 * sx)); Y2 = int(round(y2 * sy))

            name = cls_to_piece_type.get(cid, f"class-{cid}")
            label = name

            # box
            col = color_for(cid)
            cv2.rectangle(disp, (X1, Y1), (X2, Y2), col, box_thickness)

            # label background (compact)
            (tw, th), base = cv2.getTextSize(label, font, text_scale, text_thickness)
            th_total = th + base
            y1_txt = Y1 - th_total - 2
            y2_txt = Y1 - 2
            if y1_txt < 0:
                y1_txt = Y1 + 2
                y2_txt = Y1 + th_total + 2
            x1_txt = X1
            x2_txt = X1 + tw + 6

            # clamp to canvas
            x1_txt = max(0, x1_txt); y1_txt = max(0, y1_txt)
            x2_txt = min(display_size - 1, x2_txt); y2_txt = min(display_size - 1, y2_txt)

            detections.append({
                "name": name,
                "conf": float(conf),
                "bbox": [float(x1), float(y1), float(x2), float(y2)],  # original coords
                "class_id": int(cid),
            })

    if save_path:
        cv2.imwrite(save_path, disp)
    if show:
        cv2.imshow("board (800x800)", disp); cv2.waitKey(0); cv2.destroyAllWindows()

    return disp, detections
draw_corners_with_names(create_file_path(13, 19))