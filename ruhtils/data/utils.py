import pathlib
import matplotlib.pyplot as plt
import cv2
import numpy as np

def yolo_bbox(x, y, w, h, shape):
    return patches.Rectangle(
        ((x - w/2) * img.shape[1], 
         (y - h/2) * img.shape[0]), 
            w * img.shape[1], 
            h * img.shape[0], 
            linewidth=1, 
            edgecolor="green", 
            facecolor='none')
    
    
def get_yolo_xy(x_max, x_min, y_max, y_min, shape):
    x = (x_max + x_min) / 2 / shape[1]
    y = (y_max + y_min) / 2 / shape[0]
    w = (x_max - x_min) / shape[1]
    h = (y_max - y_min) / shape[0]
    
    return x, y, w, h  


def yolo_show_label(img_p, label_p, size=(10)):
    with open(label_p, "r") as label:
        fig, ax = plt.subplots(figsize=(size, size))

        label = label.read()
        img = cv2.imread(img_p)
        ax.imshow(img)

        for row in filter(None, label.split("\n")):
            row = row.split()
            cls = int(row[0])
            x, y, w, h = map(lambda x: float(x), row[1::])
            
            patch = yolo_bbox(x, y, w, h, img.shape)
            ax.add_patch(patch)
    plt.show()
    
