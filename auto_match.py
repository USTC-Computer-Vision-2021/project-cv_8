import numpy as np


def cropObjectFromSource(source: np.ndarray, position: tuple) -> np.ndarray:
    ul, br = position
    ul_h, ul_w = ul
    br_h, br_w = br
    h, w = br_h - ul_h, br_w - ul_w
    return source[ul_h: ul_h + h, ul_w: ul_w + w]


def cropObjectFromDescription(source: np.ndarray, position: str) -> np.ndarray:
    description = ['upper left', 'upper', 'upper right',
                   'left', 'middle', 'right',
                   'bottom left', 'bottom', 'bottom right']
    index = description.index(position)  # which index is the input description

    H, W = source.shape  # height and width of the source image
    H_block, W_block = H // 3, W // 3  # height and width of the grid, 9 grids in total
    row, col = index // 3, index % 3  # column and row of the grid given the description

    return source[H_block * row: H_block * row + H_block, W_block * col: W_block * col + W_block]


def autoMatch(source, target):
    raise NameError("This version does not support auto match")
