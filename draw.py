import cv2
from detect import *
def detect_and_crop(path_to_img):
    corners, ones, eights = detect_corners_and_orientation(path_to_img)
    warped = warp_quad(path_to_img, corners, ones, eights)
    cv2.namedWindow("Board", cv2.WINDOW_NORMAL)  # or: cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO
    cv2.imshow("Board", warped)

    cv2.resizeWindow("Board", 800, 800)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def draw_corners(path_to_img, corners, ones, eights):
    img = cv2.imread(path_to_img)
    for i in range(4):
        cv2.putText(img, str(i), (int(corners[i][0]), int(corners[i][1])), 
        cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 0, 0), 2)   
    for eight in eights:
        cv2.circle(img, eight, 15, (0, 255, 0), -1)
    for one in ones:    
        cv2.circle(img, one, 15, (0, 0, 255), -1)
    return img


def draw_corners_from_path(path_to_img):
    corners, ones, eights = detect_corners_and_orientation(path_to_img)
    corners = order_corners(corners, ones, eights)
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

def draw_grid(img, rows=8, cols=8, color=(0, 255, 0), thickness=1, line_type=cv2.LINE_AA):
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
    corners, ones, eights = detect_corners_and_orientation(path_to_img)
    corners = order_corners(corners, ones, eights)
    cropped_img, (bl_y, bl_x), cropped_shape = crop_rectangle(path_to_img, corners)
    pieces_coords, _ = detect_pieces(cropped_img, 0.2, 0.5)
    warped = warp_quad(path_to_img, corners, ones, eights)
    out = draw_grid(warped)
    for i in range(12):
        for piece_coord in pieces_coords[i]:
            x1, y1, x2, y2 = piece_coord
            base_midpoint = [y2, (x1+x2) // 2]
            origin_point = get_origin_point(base_midpoint, img.shape, bl_y, bl_x, cropped_shape)
            warped_point = get_point_in_board(origin_point, img.shape, corners)
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
        corners, ones, eights = detect_corners_and_orientation(path_to_img)
        if len(corners)>=4:
            warped = warp_quad(path_to_img, corners, ones, eights)
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
        

def warp_quad(path_to_img, quad_pts, ones, eights, scale=1.0, margin_ratio=0, margin=0):
    """
    Perspective-warp the quad to a top-down rectangle.
    Output size inferred from the quad side lengths, with optional margin.
    Returns warped
    """
    #ordered = order_corners(quad_pts, ones, eights)
    (tl, tr, br, bl) = quad_pts

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
    M = cv2.getPerspectiveTransform(quad_pts, dst)
    warped = cv2.warpPerspective(img_bgr, M, (maxW, maxH))
    h, w = warped.shape[:2]
    size = max(h, w)
    #square_img = cv2.resize(warped, (size, size))
    return warped
save_cropped_files(1, "cropped imgs")