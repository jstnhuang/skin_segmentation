# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------
"""Blob helper functions."""

import numpy as np
import cv2


def im_list_to_blob(ims, num_channels):
    """Convert a list of images into a network input.

    A blob is a 4D array:
      num_images
      height
      width
      channels

    If the images have different resolutions, then the height and width of the
    blob is the max height and width of all the images. Smaller images are
    placed in the upper left corner of the blob and the rest filled with zeros.

    Assumes images are already prepared (means subtracted, BGR order, ...).
    """
    max_shape = np.array([im.shape for im in ims]).max(axis=0)
    num_images = len(ims)
    blob = np.zeros(
        (num_images, max_shape[0], max_shape[1], num_channels),
        dtype=np.float32)
    for i in xrange(num_images):
        im = ims[i]
        if num_channels == 1:
            blob[i, 0:im.shape[0], 0:im.shape[1], :] = im[:, :, np.newaxis]
        else:
            blob[i, 0:im.shape[0], 0:im.shape[1], :] = im

    return blob


def pad_im(im, factor, value=0):
    """Pads the image so its width and height are a multiple of the given factor.

    The padding will be added to the right/bottom of the image, if needed.

    Args:
        im: The image, as a numpy array.
        factor: The factor which should divide the width and height.
        value: The constant value that fills the padding.
    """
    height = im.shape[0]
    width = im.shape[1]

    pad_height = int(np.ceil(height / float(factor)) * factor - height)
    pad_width = int(np.ceil(width / float(factor)) * factor - width)

    if len(im.shape) == 3:
        return np.lib.pad(
            im, ((0, pad_height), (0, pad_width), (0, 0)),
            'constant',
            constant_values=value)
    elif len(im.shape) == 2:
        return np.lib.pad(
            im, ((0, pad_height), (0, pad_width)),
            'constant',
            constant_values=value)


def unpad_im(im, factor):
    """Truncates an image so that its height and width are a multiple of the given factor.

    Args:
        im: The image, as a numpy array.
        factor: The factor which
    """
    height = im.shape[0]
    width = im.shape[1]

    pad_height = int(np.ceil(height / float(factor)) * factor - height)
    pad_width = int(np.ceil(width / float(factor)) * factor - width)

    if len(im.shape) == 3:
        return im[0:height - pad_height, 0:width - pad_width, :]
    elif len(im.shape) == 2:
        return im[0:height - pad_height, 0:width - pad_width]


def chromatic_transform(im, label=None, d_h=None, d_s=None, d_l=None):
    """
    Given an image array, add the hue, saturation and luminosity to the image
    """
    # Set random hue, luminosity and saturation which ranges from -0.1 to 0.1
    if d_h is None:
        d_h = (np.random.rand(1) - 0.5) * 0.2 * 180
    if d_l is None:
        d_l = (np.random.rand(1) - 0.5) * 0.2 * 256
    if d_s is None:
        d_s = (np.random.rand(1) - 0.5) * 0.2 * 256
    # Convert the BGR to HLS
    hls = cv2.cvtColor(im, cv2.COLOR_BGR2HLS)
    h, l, s = cv2.split(hls)
    # Add the values to the image H, L, S
    new_h = (h + d_h) % 180
    new_l = np.clip(l + d_l, 0, 255)
    new_s = np.clip(s + d_s, 0, 255)
    # Convert the HLS to BGR
    new_hls = cv2.merge((new_h, new_l, new_s)).astype('uint8')
    new_im = cv2.cvtColor(new_hls, cv2.COLOR_HLS2BGR)

    if label is not None:
        I = np.where(label > 0)
        new_im[I[0], I[1], :] = im[I[0], I[1], :]
    return new_im
