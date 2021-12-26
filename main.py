import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np
from utils import *


def configure():
    parser = argparse.ArgumentParser(description='A look into the past')
    # Images
    parser.add_argument('--img_old', type=str,   help='The old image')
    parser.add_argument('--img_new', type=str,   help='The new image')

    # Overlapped Objects
    parser.add_argument('--box_old', type=str,   help='(upper_left, bottom_right) of the objects in the old')
    parser.add_argument('--box_new', type=str,   help='(upper_left, bottom_right) of the objects in the new')

    # Threshold to control feature quality
    parser.add_argument('--th_dist', type=float, help='Control the distance of 2 similar feature')

    # Misc
    parser.add_argument('--verbose', type=bool,  help='if True, show image and print numerical results')
    parser.add_argument('--smooth',  type=bool,  help='if True, smooth the edge of 2 images')

    args = parser.parse_args()

    # TODO: DEBUG
    args.img_old = './images/OldNorthGate_old.png'
    args.img_new = './images/OldNorthGate_new.png'
    args.box_old = '158,61,426,539'
    args.box_new = '85,156,455,700'
    args.th_dist = 0.85
    args.verbose = 'True'

    args.img_old = './images/WesternGate_old.jpeg'
    args.img_new = './images/WesternGate_new.jpg'
    args.box_old = '0,0,355,1000'
    args.box_new = '0,50,226,821'
    args.th_dist = 0.75
    args.verbose = 'True'

    # convert str to int list and str to bool
    args.box_old = [int(str_num) for str_num in args.box_old.split(',')]
    args.box_new = [int(str_num) for str_num in args.box_new.split(',')]
    args.verbose = True if args.verbose == 'True' else False
    args.smooth = True if args.smooth == 'True' else False

    return args


def main():
    # Configure parameters
    args = configure()

    # Read Images
    img_old = cv2.imread(args.img_old, 0)
    img_new = cv2.imread(args.img_new, 0)

    # Locate overlapped objects
    obj_old = cropObjectFromSource(img_old, args.box_old)
    obj_new = cropObjectFromSource(img_new, args.box_new)
    if args.verbose:
        plt.imshow(obj_old, cmap='gray'), plt.show()
        plt.imshow(obj_new, cmap='gray'), plt.show()

    # Match and align the objects
    H, img_trans = match(obj_old, obj_new, args.th_dist, args.verbose)

    # Smooth the edge


if __name__ == '__main__':
    main()
