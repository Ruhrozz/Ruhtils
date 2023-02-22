import cv2
import numpy as np
import pathlib
import matplotlib.pyplot as plt
from matplotlib import patches


def intersection(bb1, bb2):
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[2], bb2[2])
    y_bottom = min(bb1[3], bb2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    return (x_right - x_left) * (y_bottom - y_top)

def yolo_bbox(x, y, w, h, shape):
    return patches.Rectangle(
        ((x - w/2) * shape[1], 
         (y - h/2) * shape[0]), 
            w * shape[1], 
            h * shape[0], 
            linewidth=1, 
            edgecolor="green", 
            facecolor='none')
    
    
def get_yolo_xy(x_max, x_min, y_max, y_min, shape):
    x = (x_max + x_min) / 2 / shape[1]
    y = (y_max + y_min) / 2 / shape[0]
    w = (x_max - x_min) / shape[1]
    h = (y_max - y_min) / shape[0]
    
    return x, y, w, h    


def yolo_show_label(img, label, raw_label=False, size=10):
    if isinstance(img, str):
        img = img = cv2.imread(img)
        
    if not raw_label:
        label_file = open(label, "r")
        label = label_file.read()

    fig, ax = plt.subplots(figsize=(size, size))
    ax.imshow(img)

    for row in filter(None, label.split("\n")):
        row = row.split()
        cls = int(row[0])
        x, y, w, h = map(lambda x: float(x), row[1::])

        patch = yolo_bbox(x, y, w, h, img.shape)
        ax.add_patch(patch)
        
    plt.show()
    
    if not raw_label:
        label_file.close()

def calcule_fit_step(shape, frame, box):
    fit = shape // frame
    intersection = (frame * (fit + 1) - shape) / fit

    while intersection < box:
        fit += 1
        intersection = (frame * (fit + 1) - shape) / fit

    return fit, intersection


def get_fit_step(img_shape, frame_shape, min_box):
    fit_h, inter_h = calcule_fit_step(img_shape[0], frame_shape[0], min_box[0])
    fit_w, inter_w = calcule_fit_step(img_shape[1], frame_shape[1], min_box[1])
    
    return (fit_h, fit_w), (inter_h, inter_w)


def make_coords(row, shape, x0, y0, w, h):
    x_min = (row[1] - row[3]/2) * shape[1] - x0
    y_min = (row[2] - row[4]/2) * shape[0] - y0
    x_max = (row[1] + row[3]/2) * shape[1] - x0
    y_max = (row[2] + row[4]/2) * shape[0] - y0

    x_min = np.maximum(0, np.minimum(x_min, w))
    x_max = np.maximum(0, np.minimum(x_max, w))
    y_min = np.maximum(0, np.minimum(y_min, h))
    y_max = np.maximum(0, np.minimum(y_max, h))
    
    return x_min, y_min, x_max, y_max


def cope_with_labels(img_crop, label, x0, y0, shape, need_blur, threshold):
    label_crop = []
    h, w, _ = img_crop.shape
    
    for row in filter(None, label.split("\n")):
        row = list(map(float, row.split(" ")))
        cls = int(row[0])

        x_min, y_min, x_max, y_max = make_coords(row, shape, x0, y0, w, h)

        square_prev = row[3] * shape[1] * row[4] * shape[0]
        square_new = (y_max - y_min) * (x_max - x_min)

        if square_new/square_prev > threshold:
            label_crop.append([None, f"{cls} {(x_min + x_max) / (2 * w)} {(y_min + y_max) / (2 * h)} {(x_max - x_min) / w} {(y_max - y_min) / h}\n"])
        elif square_new > 0 and need_blur:
            label_crop.append([[int(y_min), int(y_max), int(x_min), int(x_max)], f"0 {(x_min + x_max) / (2 * w)} {(y_min + y_max) / (2 * h)} {(x_max - x_min) / w} {(y_max - y_min) / h}\n"])
    
    return img_crop, label_crop


def maxmin(row):
    row[0] = row[0] - row[2]/2
    row[1] = row[1] - row[3]/2
    row[2] = row[0] + row[2]
    row[3] = row[1] + row[3]
    
    return row
    

def box_contains_box(row1, row2):
    row1 = list(map(float, row1.split(" ")))[1:]
    row1 = maxmin(row1)
    row2 = list(map(float, row2.split(" ")))[1:]
    row2 = maxmin(row2)
    return intersection(row1, row2)


def square(bbox):
    bbox = list(map(float, bbox.split(" ")))
    return bbox[-1] * bbox[-2]


def blur_cropes(img_crop, label_crop):
    if len(label_crop) == 0:
        return img_crop, ""
    
    label_crop = np.array(label_crop, dtype=object)
    delete = np.ones(len(label_crop), dtype=bool)
    
    blur = cv2.GaussianBlur(img_crop,(31,31),0)
    
    for i, (blur_coord, box) in enumerate(label_crop, 0):
        if blur_coord is not None:
            delete[i] = 0
            y1, y2, x1, x2 = blur_coord
            img_crop[y1:y2, x1:x2] = blur[y1:y2, x1:x2]
            for j, (bc, b) in enumerate(label_crop, 0):
                if box_contains_box(box, b) > square(b)*0.5 and bc is None:
                    pass
                    delete[j] = 0
                    
    label_crop = label_crop[:, 1][delete]
                    
    return img_crop, np.sum(label_crop) if len(label_crop) else ""
    


def crop_function(img, label, frame, min_box, need_blur, threshold):
    fit, inter = get_fit_step(img.shape, frame, min_box)
    frames = []

    for i in range(fit[1] + 1):
        for j in range(fit[0] + 1):
            x0 = i*frame[1] - inter[1]*i
            x1 = (i+1)*frame[1] - inter[1]*i

            y0 = j*frame[0] - inter[0]*j
            y1 = (j+1)*frame[0] - inter[0]*j
            
            img_crop = img[int(y0):int(y1), int(x0):int(x1)].copy()
            
            img_crop, label_crop_blur = cope_with_labels(img_crop, label, x0, y0, img.shape, need_blur, threshold)
            img_crop, label_crop = blur_cropes(img_crop, label_crop_blur)

            frames.append((img_crop, label_crop))
            
    return frames


def crop(img_p, label_p, frame=(640, 640), min_box=(100, 100), need_blur=False, threshold=0.5):
    with open(label_p, "r") as label:
        
        img = cv2.imread(str(img_p))

        return crop_function(img, label.read(), frame, min_box, need_blur, threshold)
        

if __name__ == "__main__":

    p = pathlib.Path("valid")

    for sub_dir in p.iterdir():
        for image_p in (sub_dir/"images").iterdir():
            label_p = pathlib.Path(f"valid/{sub_dir.stem}/labels/" + image_p.stem + ".txt")
            image_p = image_p

            frames = crop(image_p, label_p, need_blur=True, threshold=0.5)

            for i, (image, label) in enumerate(frames, 0):
                with open(f"new_valid/labels/{label_p.stem}_{sub_dir.stem}_{i}.txt", "w") as new_label:
                    new_label.write(label)
                    cv2.imwrite(f"new_valid/images/{label_p.stem}_{sub_dir.stem}_{i}.jpg", image)
