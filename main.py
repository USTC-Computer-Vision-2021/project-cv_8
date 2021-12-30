import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np
from utils import *


# push
def configure():
    parser = argparse.ArgumentParser(description='A look into the past')
    # Images
    parser.add_argument('--img_old', type=str, help='The old image')
    parser.add_argument('--img_new', type=str, help='The new image')

    # Overlapped Objects
    parser.add_argument('--box_old', type=str, help='(upper_left, bottom_right) of the objects in the old')
    parser.add_argument('--box_new', type=str, help='(upper_left, bottom_right) of the objects in the new')

    # Threshold to control feature quality
    parser.add_argument('--th_dist', type=float, help='Control the distance of 2 similar feature')

    # Misc
    parser.add_argument('--verbose', type=bool, help='if True, show image and print numerical results')
    parser.add_argument('--smooth', type=bool, help='if True, smooth the edge of 2 images')

    args = parser.parse_args()

    # adjust box_old, box_new and th_dist
    # OldNorthGate
    args.img_old = './images/OldNorthGate_old.png'
    args.img_new = './images/OldNorthGate_new.png'
    args.box_old = '158,61,426,539'
    args.box_new = '85,156,455,700'
    args.th_dist = 0.75
    args.verbose = 'True'

    # WesternGate
    '''args.img_old = './images/WesternGate_old.jpeg'
    args.img_new = './images/WesternGate_new.jpg'
    args.box_old = '0,0,355,1000'
    args.box_new = '0,50,226,821'
    args.th_dist = 0.75
    args.verbose = 'True'
    '''
    # convert str to int list and str to bool
    args.box_old = [int(str_num) for str_num in args.box_old.split(',')]
    args.box_new = [int(str_num) for str_num in args.box_new.split(',')]
    args.verbose = True if args.verbose == 'True' else False
    args.smooth = True if args.smooth == 'True' else False

    return args

class IMAGE:
    def __init__(self):
        self.gray = None
        self.color = None
        self.gray_crop = None
        self.color_crop = None
        self.trans = None
        self.color_trans = None



def main():
    # Configure parameters
    args = configure()

    img_old = IMAGE()
    img_new = IMAGE()

    # Read Images in gray and RGB
    img_old.gray = cv2.imread(args.img_old, 0)
    img_old.color = cv2.imread(args.img_old)

    img_new.gray = cv2.imread(args.img_new, 0)
    img_new.color = cv2.imread(args.img_new)

    # Locate overlapped objects
    img_old.gray_crop = cropObjectFromSource(img_old.gray, args.box_old)
    img_old.color_crop = cropObjectFromSource(img_old.color, args.box_old)

    img_new.gray_crop = cropObjectFromSource(img_new.gray, args.box_new)
    img_new.color_crop = cropObjectFromSource(img_new.color, args.box_new)

    if args.verbose:
        plt.imshow(img_old.gray_crop, cmap='gray'), plt.show()
        plt.imshow(img_new.gray_crop, cmap='gray'), plt.show()

    # Match and align the objects
    H, img_old.trans = match(img_old.gray_crop, img_new.gray_crop, args.th_dist, args.verbose)

    img_old.color_trans = np.zeros((img_new.gray_crop.shape[0], img_new.gray_crop.shape[1], 3))

    for i in range(3):
        img_old.color_trans[:, :, i] = cv2.warpPerspective(img_old.color_crop[:, :, i],
                                                           H, (img_new.gray_crop.shape[1], img_new.gray_crop.shape[0]))

    # Smooth the edge
    ## Average Blur
    img_result = img_old.gray.copy()
    c = args.box_new
    img_result[c[0] + 10:c[2] - 10, c[1] + 50:c[3] - 60] = img_old.trans[10:-10, 50:-60]

    img_result_color = img_new.color.copy()
    img_result_color[c[0] + 20:c[2] - 70, c[1] + 20:c[3] - 20, :] = img_old.color_trans[20:-70, 20:-20, :]

    img_result_color = img_result_color[:, :, ::-1]
    blurred_img = cv2.blur(img_result_color, (9, 9), 0)
    img_result_color[max(c[0] - 10, 0):min(c[2] + 10, img_result_color.shape[0]),
    c[1] + 40:c[3] - 50, :] \
        = blurred_img[max(c[0] - 10, 0):min(c[2] + 10, img_result_color.shape[0]),
          c[1] + 40:c[3] - 50, :]
    img_result_color[c[0] + 20:c[2] - 70, c[1] + 60:c[3] - 70, :] = img_old.color_trans[20:-70, 60:-70, ::-1]

    plt.imshow(img_old.color_trans.astype('uint8')), plt.show()

    # Poisson Blending
    mask = 255 * np.ones(img_old.color_trans[20:-70, 60:-70, ::-1].shape,
                         img_old.color_trans[20:-70, 60:-70, ::-1].astype('uint8').dtype)
    result = cv2.seamlessClone(img_old.color_trans[20:-70, 60:-70, ::-1].astype('uint8'), img_new.color.astype('uint8'),
                               mask.astype('uint8'), ((c[1] + 60 + c[3] - 70) // 2, (c[0] + 20 + c[2] - 70) // 2),
                               cv2.MONOCHROME_TRANSFER)
    plt.imshow(result[:, :, ::-1]), plt.show()
    cv2.imwrite("./result/result.png", result[:, :, :])


if __name__ == '__main__':
    main()
