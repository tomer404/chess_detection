from ultralytics import YOLO
import cv2
import os
import numpy as np
import math
import torch
from collections import defaultdict
model = YOLO(r"runs/detect/trainNano/weights/best.pt")
pieces_model = YOLO(r"runs2/detect/train21/weights/best.pt")

def create_file_path(img_num, folder_num):
    if img_num<10:
        str_i = "00"+str(img_num)
    elif 9<img_num<100:
        str_i = "0"+str(img_num)
    else:
        str_i = str(img_num)
    if folder_num<10:
        folder_str = "00"+str(folder_num)
    elif 9<folder_num<100:
        folder_str = "0"+str(folder_num)
    else:
        folder_str = str(folder_num)
    img_name = r"images/"+str(folder_num)+"/G"+folder_str+"_IMG"+str_i+".jpg"
    return img_name


def create_folder_path(folder_num):
    return r"images\\"+str(folder_num)

def mid_point_of_boxes(coords):
    new_coords = []
    for i in range(len(coords)):
        x1, y1, x2, y2 = coords[i]
        new_coords.append([int((x1+x2)//2), int((y1+y2)//2)])
    return new_coords

def detect_corners_and_orientation(path_to_img):
    results = model.predict(source=path_to_img, conf = 0, iou = 0.0, agnostic_nms = False)
    r = results[0]
    boxes = r.boxes
    # Sorting the values of the boxes by confidence using torch
    conf_vals = boxes.conf.view(-1)
    order = torch.argsort(conf_vals, descending = True)
    boxes_sorted = boxes[order]
    class_ids = boxes_sorted.cls.cpu().numpy()
    coords = boxes_sorted.xyxy.cpu().numpy()
    # get the boxes coordinates of the corners, the 1s and the 8s rows
    eights_coords = coords[class_ids == 0]
    ones_coords = coords[class_ids == 1]
    if len(eights_coords) >= 2:
        eights_coords = eights_coords[:2]
    if len(ones_coords) >= 2:
        ones_coords = ones_coords[:2]
    if len(ones_coords) == 2 and len(eights_coords) == 2:
        conf = boxes_sorted.conf.cpu().numpy()
        eights_conf = conf[class_ids == 0]
        eights_conf = eights_conf[:2]
        ones_conf = conf[class_ids == 1]
        ones_conf = ones_conf[:2]
        if(ones_conf[1] > eights_conf[1]):
            eights_coords = []
        else:
            ones_coords = []
    
    # get the coordinates of the middle point of the boxes
    #corners_coords = mid_point_of_boxes(corners_coords)
    ones_coords = mid_point_of_boxes(ones_coords)
    eights_coords = mid_point_of_boxes(eights_coords)
    return [ones_coords, eights_coords]

def distance(pt1, pt2):
    return (pt1[0]-pt2[0])**2+(pt1[1]-pt2[1])**2

def closest_corner(corners, pt):
    min_distance = distance(pt, corners[0])
    min_distance_ind = 0
    for i in range(1, len(corners)):
        if distance(pt, corners[i])<min_distance:
            min_distance = distance(pt, corners[i])
            min_distance_ind = i
    return min_distance_ind


def order_corners_clockwise(corners):
    # This method returns the corners in a clockwise order, regardless of the 1st and 8th row
    corners.sort(key=lambda corner: corner[1])
    top = corners[:2]
    bottom = corners[2:4]
    top.sort(key = lambda corner: corner[0])
    bottom.sort(key = lambda corner: corner[0], reverse = True)
    return top + bottom


def order_corners(corners, ones, eights):
    # This method fixes the top row to be the 8th row
    if(len(corners) == 4):
        corners = order_corners_clockwise(corners)
        if(len(eights) == 2):
            tl_ind = closest_corner(corners, eights[0])
            tr_ind = closest_corner(corners, eights[1])
            if {tl_ind, tr_ind} == {0, 3}:
                first = 3
            else:
                first = min(tl_ind, tr_ind)
            # Cyclic shifting the sorted array so that first is in the beginning
            corners = corners[first:4] + corners[:first]
            return np.array(corners, dtype=np.float32).reshape(4, 2)
        elif(len(ones) == 2):
            wh_0 = closest_corner(corners, ones[0])
            wh_1 = closest_corner(corners, ones[1])
            if wh_0!=wh_1:
                tr_ind, tl_ind = list({0, 1, 2, 3}-{wh_0, wh_1})
                if {tl_ind, tr_ind} == {0, 3}:
                    first = 3
                else:
                    first = min(tl_ind, tr_ind)
            else:
                first = wh_0
            # Cyclic shifting the sorted array so that first is in the beginning
            corners = corners[first:4] + corners[:first]
            return np.array(corners, dtype=np.float32).reshape(4, 2)
        return np.array(corners, dtype = np.float32).reshape(4, 2)
    return None


def remove_intersecting_boxes(coords):
    new_coords = []
    for (x1, y1, x2, y2) in coords:
        if x1 > x2: x1, x2 = x2, x1
        if y1 > y2: y1, y2 = y2, y1

        valid = True
        for (X1, Y1, X2, Y2) in new_coords:
            non_overlap = (x2 <= X1) or (X2 <= x1) or (y2 <= Y1) or (Y2 <= y1)
            if not non_overlap:  
                valid = False
                break
        if valid:
            new_coords.append((x1, y1, x2, y2))
    return new_coords



def detect_pieces_and_corners(img, conf_score, iou_score):
    results = pieces_model.predict(source = img, conf = 0, iou = iou_score, agnostic_nms = False)
    r = results[0]
    boxes = r.boxes
    # Sorting the values of the boxes by confidence using torch
    conf_vals = boxes.conf.view(-1)
    order = torch.argsort(conf_vals, descending = True)
    boxes_sorted = boxes[order]
    class_ids = boxes_sorted.cls.cpu().numpy()
    corner_boxes = boxes_sorted[class_ids == 12]
    corners = corner_boxes.xyxy.cpu().numpy()
    corners = remove_intersecting_boxes(corners)    
    if len(corners) > 4:
        corners = corners[:4]
    corner_points = mid_point_of_boxes(corners)
    mask = boxes_sorted.conf.view(-1) >= conf_score
    boxes_sorted = boxes_sorted[mask]
    class_ids = boxes_sorted.cls.cpu().numpy()
    conf = boxes_sorted.conf.cpu().numpy()
    coords = boxes_sorted.xyxy.cpu().numpy()
    coords_split_cls = [coords[class_ids == i] for i in range(12)]
    conf_split_cls = [conf[class_ids == i] for i in range(12)]
    return coords_split_cls, conf_split_cls, corner_points


def get_origin_point(point, original_shape, bl_y, bl_x, cropped_shape):
    '''get the coordinates of the point in the original image, before cropping and streching'''
    h, w = original_shape[:2]
    cropped_h, cropped_w = cropped_shape[:2]
    size = max(cropped_h, cropped_w)
    y, x = point
    if(size == cropped_h):
        x = int(cropped_w/size*x)
    else:
        y = int(cropped_h/size*y)
    y, x = y + bl_y, x + bl_x
    return x, y


def get_dst_size(corners):
    (tl, tr, br, bl) = corners

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)

    maxW = int(math.ceil(max(widthA, widthB)))
    maxH = int(math.ceil(max(heightA, heightB)))
    maxW = max(maxW, 10)
    maxH = max(maxH, 10)
    return maxW, maxH


def get_point_in_board(point, shape, corners):
    h, w = shape[:2]
    maxW, maxH = get_dst_size(corners)
    dst = np.array([
        [0, 0],
        [maxW - 1, 0],
        [maxW - 1, maxH - 1],
        [0, maxH - 1]
    ], dtype=np.float32)
    M = cv2.getPerspectiveTransform(corners, dst)
    pt_src = np.array([[[point[0], point[1]]]], dtype=np.float32)
    warped_point = cv2.perspectiveTransform(pt_src, M)
    return warped_point[0, 0]


def check_if_inside(y, x, h, w):
    return not (x > w or x < 0 or y > h or y < 0)

def get_base_midpoint(coords):
    x1, y1, x2, y2 = coords
    base_midpoint = [(x1+x2) // 2, y2*9/10 + y1/10]
    return base_midpoint
def get_square_of_piece(coords, shape, corners):
    base_midpoint = get_base_midpoint(coords)
    #origin_point = get_origin_point(base_midpoint, shape, bl_y, bl_x, cropped_shape)
    warped_point = get_point_in_board(base_midpoint, shape, corners)
    maxW, maxH = get_dst_size(corners)
    x, y = warped_point
    if check_if_inside(y, x, maxH-1, maxW-1):
        x_ind, y_ind = int(8*x/(maxW-1)), int(8*y/(maxH-1))
        if x_ind == 8:
            x_ind = 7
        if y_ind == 8:
            y_ind = 7
        return x_ind, y_ind
    else:
        return None
    

def crop_rectangle(path_to_img, quad_pts, margin = 0):
    '''This method crops a bounding rectangle of the board'''
    img = cv2.imread(path_to_img)
    h_img, w_img = img.shape[:2]
    min_x = max(0, int(min(quad_pts[0][0], quad_pts[1][0], quad_pts[2][0], quad_pts[3][0]) - margin))
    max_x = min(w_img, int(max(quad_pts[0][0], quad_pts[1][0], quad_pts[2][0], quad_pts[3][0]) + margin))
    min_y = max(0, int(min(quad_pts[0][1], quad_pts[1][1], quad_pts[2][1], quad_pts[3][1]) - margin))
    max_y = min(h_img, int(max(quad_pts[0][1], quad_pts[1][1], quad_pts[2][1], quad_pts[3][1]) + margin))
    cropped_img = img[min_y:max_y, min_x:max_x]
    h, w = cropped_img.shape[:2]
    size = max(h, w)
    square_img = cv2.resize(cropped_img, (size, size))
    return square_img, (min_y, min_x), cropped_img.shape
    

def main(path_to_img, conf_score, iou_score):
    ones, eights = detect_corners_and_orientation(path_to_img)
    img = cv2.imread(path_to_img)
    #cropped_img, (bl_y, bl_x), cropped_shape = crop_rectangle(path_to_img, corners, margin=220)
    pieces_coords, pieces_conf, corners = detect_pieces_and_corners(img, conf_score, iou_score)
    corners = order_corners(corners, ones, eights)
    cls_to_piece_type = {0: "white-pawn",  1: "white-rook",  2: "white-knight", 3: "white-bishop", 4: "white-queen", 5: "white-king",
    6: "black-pawn", 7: "black-rook", 8: "black-knight", 9: "black-bishop", 10: "black-queen", 11: "black-king"}
    num_to_file = {0:"a", 1:"b", 2:"c", 3: "d", 4: "e", 5: "f", 6: "g",7: "h"}
    square_to_piece = defaultdict(list) # multi values dictionary
    square_to_piece_final = {}
    for i in range(12):
        # build (coord, orig_idx) list while filtering
        kept = []
        for orig_j, piece_coord in enumerate(pieces_coords[i]):
            square = get_square_of_piece(piece_coord, img.shape, corners)
            if square is not None:
                kept.append((piece_coord, orig_j))

        # cap
        piece_caps = {
            0:8, 1:2, 2:2, 3:2, 4:1, 5:1,   # pawns, rooks, knights, bishops, queen, king
            6:8, 7:2, 8:2, 9:2, 10:1, 11:1  
        }
        limit = piece_caps.get(i, 8)
        kept = kept[:limit]

        # add using original indices
        for piece_coord, orig_j in kept:
            square = get_square_of_piece(piece_coord, img.shape, corners)
            if square is not None:
                square_to_piece[square].append((i, orig_j))

    # handling collisions - if there are multiple pieces detected on the same square,
    # we're taking only the one with the highest confidence value
    for square in square_to_piece:
        max_conf = 0
        for piece in square_to_piece[square]:
            current_conf = pieces_conf[piece[0]][piece[1]]
            if current_conf > max_conf:
                max_conf = current_conf
                piece_with_max_conf = piece
        square_str = num_to_file[square[0]]+str(8-square[1])
        square_to_piece_final[square_str] = cls_to_piece_type[piece_with_max_conf[0]]
    return square_to_piece_final

def pos_to_fen(pos_map, side_to_move):
    fen = ""
    piece_to_notation = {"black-bishop": "b","black-king": "k","black-knight": "n","black-pawn": "p","black-queen": "q","black-rook": "r",
                    "white-bishop": "B","white-king": "K","white-knight": "N","white-pawn": "P", "white-queen": "Q", "white-rook": "R"}
    for row in range(8, 0, -1):
        cnt_empty = 0
        for col in ["a", "b", "c", "d", "e", "f", "g", "h"]:
            piece = pos_map.get(col+str(row))
            if piece is None:
                cnt_empty += 1
            else:
                if(cnt_empty != 0):
                    fen += str(cnt_empty)
                    cnt_empty = 0
                fen += piece_to_notation[piece]
        if cnt_empty != 0:
            fen += str(cnt_empty)
        if row != 1:
            fen += "/"
    fen += f" {side_to_move} KQkq - 0 0"
    return fen
