import numpy as np


def generate_anchor_points(stride=8, row=2, line=2):
    """
    Base offsets for a small grid of anchor points WITHIN one stride x stride
    cell (paper default: 2x2 = 4 points per cell). These get tiled across
    every feature-map location by shift_anchor_points below.
    """
    row_step = stride / row
    line_step = stride / line
    x = (np.arange(1, line + 1) - 0.5) * line_step - stride / 2
    y = (np.arange(1, row + 1) - 0.5) * row_step - stride / 2
    xx, yy = np.meshgrid(x, y)
    anchor_points = np.vstack((xx.ravel(), yy.ravel())).T
    return anchor_points  # shape: (row*line, 2)


def shift_anchor_points(feature_map_size, stride, anchor_points):
    """
    Tiles the base anchor offsets across every location of the feature map,
    producing absolute (x, y) anchor point coordinates in the ORIGINAL image
    resolution (not the feature map's downsampled resolution).
    """
    h, w = feature_map_size
    shift_x = (np.arange(0, w) + 0.5) * stride
    shift_y = (np.arange(0, h) + 0.5) * stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel())).T  # (h*w, 2)

    A = anchor_points.shape[0]
    K = shifts.shape[0]
    all_points = (anchor_points.reshape((1, A, 2)) + shifts.reshape((K, 1, 2))).reshape((K * A, 2))
    return all_points  # shape: (h*w*A, 2)