import random
import time

import cv2

from src.option import args
import numpy as np
import skimage.color as sc
from src.RCNN import RCNN

import torch
import torch.cuda
import torch.nn.functional as F
import cv2
import torch.multiprocessing
torch.multiprocessing.set_start_method('spawn', force=True)

device = torch.device("cuda")
import numpy as np

# The code randomly selects the starting position of a patch and extracts the corresponding region from the input image.
def get_patch(*args, patch_size=96, scale=2, multi=False, input_large=False):
    ih, iw = args[0].shape[:2]

    if not input_large:
        p = scale if multi else 1
        tp = p * patch_size
        ip = tp // scale
    else:
        tp = patch_size
        ip = patch_size

    ix = random.randrange(0, iw - ip + 1)
    iy = random.randrange(0, ih - ip + 1)

    if not input_large:
        tx, ty = scale * ix, scale * iy
    else:
        tx, ty = ix, iy

    ret = [
        args[0][iy:iy + ip, ix:ix + ip, :],
        *[a[ty:ty + tp, tx:tx + tp, :] for a in args[1:]]
    ]
    return ret



def set_channel(lr, hr, n_channels=3):
    if lr.ndim == 2 and hr.ndim == 2:
        lr = np.expand_dims(lr, axis=2)
        hr = np.expand_dims(hr, axis=2)
    c = lr.shape[2]

    #     Convert the image from RGB to YCbCr color space and select the luminance channel as the new image
    if n_channels == 1 and c == 3:
        lr = np.expand_dims(sc.rgb2ycbcr(lr)[:, :, 0], 2)

    # Copy the image three times and concatenate along the channel dimension to obtain an image with three channels
    elif n_channels == 3 and c == 1:
        lr = np.concatenate([lr] * n_channels, 2)


    return lr, hr

# The code creates a NumPy array by rearranging the channel dimensions in a specified order.
def np2Tensor(*args, rgb_range=65536, ori_rgb_range=65536):
    def _np2Tensor(img):
        np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
        tensor = torch.from_numpy(np_transpose).float()
        tensor.mul_(rgb_range / ori_rgb_range)
        return tensor

    return [_np2Tensor(a) for a in args]

# Whether to perform horizontal flipping, vertical flipping, and 90-degree rotation. These variables are randomly determined based on the given probability.
def augment(*args, hflip=True, rot=True):
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):
        if hflip: img = img[:, ::-1, :]
        if vflip: img = img[::-1, :, :]
        if rot90: img = img.transpose(1, 0, 2)

        return img

    return [_augment(a) for a in args]

