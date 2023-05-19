'''
The purpose of augmentations is to increase the size of the training set
by applying random (or selected) transformations to the training images.

Create augmentation classes for use with the PyTorch Compose class 
that takes a list of transformations and applies them in order, which 
can be chained together simply by defining a __call__ method for each class. 
'''
import cv2
import numpy as np
import torch
from typing import Any, Tuple
import torchvision

class NormalizeBPS(object):
    def __call__(self, img_array) -> np.array(np.float32):
        """
        Normalize the array values between 0 - 1
        """
        # Convert the array to float32
        img_array = img_array.astype(np.float32)
        
        # Normalize the array values between 0 - 1
        # divide all by max value (so max value is 1)
        if np.max(img_array) != 0:
            img_array /= np.max(img_array)
        
        return img_array
        # raise NotImplementedError

class ResizeBPS(object):
    def __init__(self, resize_height: int, resize_width: int):
        self.resize_height = resize_height
        self.resize_width = resize_width
        #pass
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
        Resize the image to the specified width and height

        Args:
            img (np.ndarray): Image to be resized.

        Returns:
            np.ndarray: Resized image.
        """
        resized_array = cv2.resize(img, (self.resize_width, self.resize_height))
        
        return resized_array
        #raise NotImplementedError

class ZoomBPS(object):
    def __init__(self, zoom: float=1) -> None:
        self.zoom = zoom

    def __call__(self, image) -> np.ndarray:
        s = image.shape
        s1 = (int(self.zoom*s[0]), int(self.zoom*s[1]))
        img = np.zeros((s[0], s[1]))
        img_resize = cv2.resize(image, (s1[1],s1[0]), interpolation = cv2.INTER_AREA)
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

class VFlipBPS(object):
    def __call__(self, image) -> np.ndarray:
        """
        Flip the image vertically
        """
        image = cv2.flip(image, 0)
        return image
        #raise NotImplementedError


class HFlipBPS(object):
    def __call__(self, image) -> np.ndarray:
        """
        Flip the image horizontally
        """
        image = cv2.flip(image, 1)
        return image

        #raise NotImplementedError


class RotateBPS(object):
    def __init__(self, rotate: int) -> None:
        self.rotate = rotate
        
        #pass

    def __call__(self, image) -> Any:
        '''
        Initialize an object of the Augmentation class
        Parameters:
            rotate (int):
                Optional parameter to specify a 90, 180, or 270 degrees of rotation.
        Returns:
            np.ndarray
        '''
        s = image.shape
        cy = (s[0]-1)/2  # y center : float
        cx = (s[1]-1)/2  # x center : float
        M = cv2.getRotationMatrix2D((cx, cy), self.rotate, 1)  # rotation matrix
        # Affine transformation to rotate the image and output size s[1],s[0]
        return cv2.warpAffine(image, M, (s[1], s[0]))
    
        #raise NotImplementedError


class RandomCropBPS(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
        is made.
    """

    def __init__(self, output_height: int, output_width: int):
        self.output_height = output_height
        self.output_width = output_width
        # pass

    def __call__(self, image):
        # Get the input image dimensions
        # first to params in image.shape
        input_height, input_width = image.shape[:2]

        # Calculate the top-left corner coordinates for random cropping
        top = np.random.randint(0, input_height - self.output_height)
        left = np.random.randint(0, input_width - self.output_width)

        # Perform the random crop
        cropped_image = image[top:top+self.output_height, left:left+self.output_width]
        return cropped_image
    
        #raise NotImplementedError

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, image: np.ndarray) -> torch.Tensor:
        # numpy image: H x W x C
        # torch image: C x H x W
        
        #The np.transpose() function is used to change the order of dimensions in the NumPy array,
        #swapping the height (H) and width (W) dimensions to match the desired shape of the PyTorch tensor.
        # tensor = torch.from_numpy(np.transpose(image, (2, 0, 1)))
        tensor = torch.from_numpy(image).unsqueeze(0)
        return tensor

        #raise NotImplementedError

def main():
    """Driver function for testing the augmentations. Make sure the file paths work for you."""
    # load image using cv2
    img_key = 'P280_73668439105-F5_015_023_proj.tif'
    img_array = cv2.imread(img_key, cv2.IMREAD_ANYDEPTH)
    print(img_array.shape, img_array.dtype)
    test_resize = ResizeBPS(500, 500)
    type(test_resize)

if __name__ == "__main__":
    main()