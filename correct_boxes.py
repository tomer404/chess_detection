import os
def transform_coords(x, y, w, h, length):
    center_x = x + w/2
    center_y = y + h/2
    return center_x-length/2, center_y-length/2, length, length


def correct_boxes(input_file):
    with open(input_file, "r") as f:
        lines = f.readlines()
    new_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls, x, y, w, h = parts
        x, y, w, h = map(float, (x, y, w, h))

        x, y, w, h = transform_coords(x, y, w, h, 0.2)

        new_lines.append(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")    
    with open(input_file, "w") as f:
        f.writelines(new_lines)


def correct_input_files(folder_name):
    for file_name in os.listdir(folder_name):
        file_path = os.path.join(folder_name, file_name)
        correct_boxes(file_path)


correct_input_files("test\labels")
correct_input_files("valid\labels")
correct_input_files("train\labels")