'''
The purpose of augmentations is to increase the size of the training set
by applying random (or selected) transformations to the training images.

Create augmentation classes for use with the PyTorch Compose class
that takes a list of transformations and applies them in order, which
can be chained together simply by defining a __call__ method for each class.
'''
from bisect import bisect_left
import cv2
import numpy as np
import random
import torch
from typing import Any


class NormalizeBPS(object):
    """Class to normalize a BPS numpy array image"""

    def __call__(self, img_array) -> np.array(np.float32):
        """
        Normalize the array values between 0 - 1
        """
        if np.max(img_array := img_array.astype(np.float32)) != 0:
            return img_array / np.max(img_array)
        else:
            return img_array


class ResizeBPS(object):
    """Class to resize a BPS numpy array image"""
    def __init__(self, resize_height: int, resize_width: int):
        self.resize_size = (resize_height, resize_width)

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
        Resize the image to the specified width and height

        args:
            img (np.ndarray): image to be resized.
        returns:
            np.ndarray: resized image.
        """
        return cv2.resize(img, dsize=self.resize_size)


class VFlipBPS(object):
    """Class to vertically flip a BPS numpy array image """
    def __call__(self, image) -> np.ndarray:
        """
        Flip the image vertically
        """
        return np.flipud(image)


class HFlipBPS(object):
    """Class to horizontally flip a BPS numpy array image """
    def __call__(self, image) -> np.ndarray:
        """
        Flip the image horizontally
        """
        return np.fliplr(image)


def _take_closest(num: int, num_list: list[int]) -> int:
    """
    Assumes num_list is sorted. Returns closest value to num.

    If two numbers are equally close, return the smallest number.
    """
    pos = bisect_left(num_list, num)
    if pos == 0:
        return num_list[0]
    if pos == len(num_list):
        return num_list[-1]

    before = num_list[pos - 1]
    after = num_list[pos]

    if after - num < num - before:
        return after
    else:
        return before


class RotateBPS(object):
    """Class to rotate a BPS numpy array image by a multiple of 90"""

    def __init__(self, rotate: int):
        self.rotate = _take_closest(rotate, [90, 80, 270])

    def __call__(self, image) -> Any:
        '''
        Initialize an object of the Augmentation class
        Parameters:
            rotate (int):
                Optional parameter to specify a 90, 180, or 270
                degrees of rotation.
        Returns:
            np.ndarray
        '''
        len_y, len_x = image.shape
        center = (len_y-1) / 2, (len_x-1) / 2
        rotation_matrix = cv2.getRotationMatrix2D(center, self.rotate, 1)

        # Affine transformation to rotate the image and output size s[1],s[0]
        return cv2.warpAffine(image, rotation_matrix, (len_x, len_y))


class RandomCropBPS(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
        is made.
    """
    def __init__(self, output_height: int, output_width: int):
        self.output_height = output_height
        self.output_width = output_width

    def __call__(self, image) -> Any:
        img_h, img_w = image.shape[:2]
        vert_crop_start = random.randrange(0, abs(img_h-self.output_height))
        horiz_crop_start = random.randrange(0, abs(img_w-self.output_width))

        return image[vert_crop_start:vert_crop_start + self.output_height,
                     horiz_crop_start:horiz_crop_start + self.output_width]


class ZoomBPS(object):
    """Class to Zoom in on a BPS numpy array image """
    def __init__(self, zoom: float = 1):
        self.zoom = zoom

    def __call__(self, image) -> np.ndarray:
        s = image.shape
        s1 = (int(self.zoom*s[0]), int(self.zoom*s[1]))
        img = np.zeros((s[0], s[1]))
        img_resize = cv2.resize(
            image, (s1[1], s1[0]), interpolation=cv2.INTER_AREA)

        # Resize the image using zoom as scaling factor with area interpolation
        if self.zoom < 1:
            y1 = s[0]//2 - s1[0]//2
            y2 = s[0]//2 + s1[0] - s1[0]//2
            x1 = s[1]//2 - s1[1]//2
            x2 = s[1]//2 + s1[1] - s1[1]//2
            img[y1:y2, x1:x2] = img_resize
            return img
        else:
            return img_resize


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, image: np.ndarray) -> torch.Tensor:
        """numpy image: H x W x C
           torch image: C x H x W"""
        # unsqueeze shifts the dimension over
        # like a ring counter
        return torch.from_numpy(image).unsqueeze(0)


def main():
    """Driver function for testing the augmentations.
    Make sure the file paths work for you."""

    # load image using cv2
    img_key = 'P280_73668439105-F5_015_023_proj.tif'
    img_array = cv2.imread(img_key, cv2.IMREAD_ANYDEPTH)
    print(img_array.shape, img_array.dtype)
    test_resize = ResizeBPS(500, 500)
    type(test_resize)


if __name__ == "__main__":
    main()
