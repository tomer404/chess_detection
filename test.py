import json
from pathlib import Path
from typing import Dict, Union
import pandas as pd
from detect import *


def read_chess_positions(
    dataroot: Union[str, Path],
    image: Union[int, str, Path],
    include_empty: bool = False
) -> Dict[str, str]:
    """
    Read the chess-piece positions for a given image.

    Args:
        dataroot: Folder containing annotations.json and the dataset files.
        image: Either the image_id (int) OR the image 'path' string
               as listed in annotations.json.
        include_empty: If True, include empty squares in the dict with value 'empty'.

    Returns:
        positions_dict: mapping like {"a8": "black rook", "e1": "white king", ...}
                        (empty squares omitted unless include_empty=True)
    """
    dataroot = Path(dataroot)
    ann_path = dataroot / "annotations.json"
    if not ann_path.is_file():
        raise FileNotFoundError(f"File '{ann_path}' doesn't exist.")

    with open(ann_path, "r") as f:
        annotations_file = json.load(f)

    # Tables
    annotations = pd.DataFrame(annotations_file["annotations"]["pieces"])
    categories = pd.DataFrame(annotations_file["categories"])
    images = pd.DataFrame(annotations_file["images"])

    # Resolve image_id
    if isinstance(image, int):
        img_rows = images[images["id"] == image]
    else:
        img_rows = images[images["path"] == str(image)]

    if len(img_rows) != 1:
        raise ValueError(
            f"Could not uniquely resolve image '{image}', matches found: {len(img_rows)}"
        )

    image_id = int(img_rows.iloc[0]["id"])

    # Category id -> name
    cat_id_to_name = {
        int(row["id"]): str(row["name"]).replace("_", " ").lower()
        for _, row in categories.iterrows()
    }

    # Filter annotations for this image
    anns_img = annotations[annotations["image_id"] == image_id].copy()

    positions: Dict[str, str] = {}
    for _, row in anns_img.iterrows():
        pos = str(row["chessboard_position"])  # e.g., "a8"
        cat_id = int(row["category_id"])
        name = cat_id_to_name.get(cat_id, f"id_{cat_id}")
        if include_empty or name != "empty":
            positions[pos] = name

    return positions

def num_of_incorrect_pieces(img_file_path, conf_score, iou_score):
    pos_map = read_chess_positions(dataroot="", image= img_file_path, include_empty=False)
    detected_pos_map = main(img_file_path, conf_score, iou_score)
    cnt = 0
    for row in range(1, 9):
        for col in ["a", "b", "c", "d", "e", "f", "g", "h"]:
            if pos_map.get(col+str(row)) != detected_pos_map.get(col+str(row)):
                cnt+=1
    return cnt

def test_folder_accuracy(folder_num):
    folder_path = create_folder_path(folder_num)
    files = os.listdir(folder_path)
    accuracy = {}
    cnt = 0
    cnt2 = 0
    for i in range(len(files)):
        accuracy[i] = num_of_incorrect_pieces(create_file_path(i, folder_num), 0.2, 0.3)
        if(accuracy[i] == 0):
            cnt+=1
        if(accuracy[i] <= 1):
            cnt2 += 1
    return len(files), cnt, cnt2, accuracy

def test_folders_accuracy(folder_nums):
    sum = 0
    sum2 = 0
    sum3 = 0 
    accuracies = []
    for i in folder_nums:
        num_files, cnt, cnt2, accuracy = test_folder_accuracy(i)
        sum +=cnt; sum2+=cnt2; sum3 += num_files
        accuracies.append(accuracy)
    print(f"percentage of correct: {sum/sum3}")
    print(f"percentage of fewer than 1 mistakes: {sum2/sum3}")
    print(accuracies)

def print_folder_fen(folder_num):
    folder_path = create_folder_path(folder_num)
    files = os.listdir(folder_path)
    fens = {}
    for i in range(1):
        fens[i] = pos_to_fen(main(create_file_path(i, folder_num), 0.2, 0.3))
    print(fens)

if __name__ == "__main__":
    print(pos_to_fen(main(create_file_path(0, 33), 0.2, 0.3), "w"))
    print(test_folders_accuracy([6, 33, 76]))