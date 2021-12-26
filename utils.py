import numpy as np
import cv2
import matplotlib.pyplot as plt


def cropObjectFromSource(source: np.ndarray, position: list[int]) -> np.ndarray:
    ul_h, ul_w, br_h, br_w = position
    h, w = br_h - ul_h, br_w - ul_w
    return source[ul_h: ul_h + h, ul_w: ul_w + w]


def match(obj_old, obj_new, th_dist=0.8, verbose=True):
    # extract local feature using SIFT
    sift = cv2.SIFT_create()  # initial SIFT class

    kp_old, des_old = sift.detectAndCompute(obj_old, None)  # keypoint: (M, N) and descriptor:feature matrix of (a, 128)
    kp_new, des_new = sift.detectAndCompute(obj_new, None)  # keypoint: (M, N) and descriptor:feature matrix of (b, 128)
    # feature matching
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des_old, des_new, k=2)
    # get good feature points
    good_matches = []
    good_points = []
    for m, n in matches:
        if m.distance < th_dist * n.distance:  # control the threshold, like 0.75, 0.85
            good_points.append((m.trainIdx, m.queryIdx))
            good_matches.append([m])
    # draw feature map
    if verbose:
        img_temp = np.zeros((1000, 1000), dtype='uint8')
        img_match = cv2.drawMatchesKnn(obj_old, kp_old, obj_new, kp_new, good_matches, img_temp, flags=2)
        plt.imshow(img_match), plt.show()

    # regression of 2 features
    kp_old = np.float32([kp_old[i].pt for (_, i) in good_points])
    kp_new = np.float32([kp_new[i].pt for (i, _) in good_points])
    H, status = cv2.findHomography(kp_old, kp_new, cv2.RANSAC, 5.0)

    transformed_img_old = cv2.warpPerspective(obj_old, H, (obj_new.shape[1], obj_new.shape[0]))
    if verbose:
        plt.imshow(transformed_img_old, cmap='gray'), plt.show()

    return H, transformed_img_old
