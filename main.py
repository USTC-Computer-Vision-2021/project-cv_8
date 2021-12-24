import argparse
import cv2
from auto_match import *


def configure():
    parser = argparse.ArgumentParser(description='A look into the past')
    # Images
    parser.add_argument('--source', type=str, help='The old image', required=True)
    parser.add_argument('--target', type=str, help='The new image', required=True)
    # Overlapped Objects
    parser.add_argument('--precise_pos', type=tuple, help='(upper_left, bottom_right) of the overlapped objects')

    parser.add_argument('--vague_pos', type=str, help='The new image',
                        choices=['upper left', 'upper', 'upper right',
                                 'left', 'middle', 'right',
                                 'bottom left', 'bottom', 'bottom right'])
    parser.add_argument('--auto_match', type=bool, help='Whether to use auto match method')

    args = parser.parse_args()

    return args


def main():
    # Configure parameters
    args = configure()

    # Read Images
    img_source = cv2.imread(args.source, 0)
    img_target = cv2.imread(args.target, 0)

    # Locate overlapped objects
    if args.precise_pos:
        img_overlapped = cropObjectFromSource(img_source, args.precise_pos)
    elif args.vague_pos:
        img_overlapped = cropObjectFromDescription(img_source, args.vague_pos)
    else:
        img_overlapped = autoMatch(img_source, img_target)

    # Match the overlapped objects
    w, h = img_overlapped.shape[::-1]
    res = cv2.matchTemplate(img_target, img_overlapped, cv2.TM_CCORR_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)


