import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', '-s', '--source', type=str, help='Video to transform', default=None)
    parser.add_argument('--dst', '-d', '--destination', type=str, help='Where save to', default='.')
    return parser.parse_args()


def main():
    args = parse_args()

    if args.dst[-1] != '/':
        args.dst += '/'

    if args.src is None:
        print("No video was provided")
        return

    out_name = args.dst + 'output.avi'
    vod = cv2.VideoCapture(args.src)

    out = cv2.VideoWriter(
        out_name,
        cv2.VideoWriter_fourcc(*"MJPG"), 
        vod.get(cv2.CAP_PROP_FPS), 
        (
            int(vod.get(cv2.CAP_PROP_FRAME_WIDTH)), 
            int(vod.get(cv2.CAP_PROP_FRAME_HEIGHT))
        )
    )


    mask = np.zeros([
        int(vod.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        int(vod.get(cv2.CAP_PROP_FRAME_WIDTH)),
        3,
    ])

    mask[290:750, 70:200] = 1
    mask[290:320, 200:290] = 1
    mask[100:180, 70:250] = 1
    mask[50:130, 610:1300] = 1
    mask[960:1055, 1700:1880] = 1

    print("Starting transform...")
    while 1:
        ret, frame = vod.read()
        if not ret:
            break
            
        blured = cv2.medianBlur(frame, 25)
        new_frame = np.where(mask, blured, frame)
        out.write(new_frame)
        
    print("Saved as ", out_name)


if __name__ == '__main__':
    main()
