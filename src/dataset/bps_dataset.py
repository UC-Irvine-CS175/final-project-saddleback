"""
This module contains the BPSMouseDataset class which is a subclass of torch.utils.data.Dataset.
"""

import sys

import os
import torch
import cv2
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl
from torchvision import transforms, utils

import pandas as pd
from sklearn.model_selection import train_test_split

import numpy as np
from matplotlib import pyplot as plt
import pyprojroot
from pyprojroot import here

root = pyprojroot.find_root(pyprojroot.has_dir(".git"))
pytorch_dataset_dir = root / 'dataset'
data_dir = root / 'data'

import sys
sys.path.append(str(root))

# from utils.utils import get_bytesio_from_aws
import boto3
from botocore import UNSIGNED
from botocore.config import Config
import io
from io import BytesIO
from PIL import Image
from src.dataset.augmentation import (
    NormalizeBPS,
    ResizeBPS,
    ZoomBPS,
    VFlipBPS,
    HFlipBPS,
    RotateBPS,
    RandomCropBPS,
    ToTensor
)

from src.data_utils import (
    get_bytesio_from_s3,
    get_file_from_s3,
    save_tiffs_local_from_s3,
    export_subset_meta_dose_hr,
    train_test_split_subset_meta_dose_hr,
)


class BPSMouseDataset(torch.utils.data.Dataset):
    """
    Custom PyTorch Dataset class for the BPS microscopy data.

    args:
        meta_csv_file (str): name of the metadata csv file
        meta_root_dir (str): path to the metadata csv file
        bucket_name (str): name of bucket from AWS open source registry.
        transform (callable, optional): Optional transform to be applied on a sample.

    attributes:
        meta_df (pd.DataFrame): dataframe containing the metadata
        bucket_name (str): name of bucket from AWS open source registry.
        train_df (pd.DataFrame): dataframe containing the metadata for the training set
        test_df (pd.DataFrame): dataframe containing the metadata for the test set
        transform (callable): The transform to be applied on a sample.

    raises:
        ValueError: if the metadata csv file does not exist
    """

    def __init__(
            self,
            meta_csv_file:str,
            meta_root_dir:str,
            s3_client: boto3.client = None,
            bucket_name: str = None,
            transform=None,
            file_on_prem:bool = True):
        
        #assign passed-in attributes to object attributes
        self.meta_csv_file = meta_csv_file
        self.meta_root_dir = meta_root_dir
        self.s3_client = s3_client
        self.bucket_name = bucket_name
        self.transform = transform
        self.file_on_prem = file_on_prem

        # formulate the full path to metadata csv file
        full_path = os.path.join(meta_root_dir, meta_csv_file)
        # if the file is not on the local file system, use the get_bytesio_from_s3 function
        if not os.path.exists(full_path):
            print("?")
            #raise ValueError("File does not exist on the local file system.")
            self.meta_df = get_bytesio_from_s3(s3_client, bucket_name, full_path)
            # self.meta_df = pd.read_csv(self.meta_df)
        
        else:  
            self.meta_df = pd.read_csv(full_path)

        # to fetch the file as a BytesIO object, else read the file from the local file system.
        #pass


    def __len__(self):
        """
        Returns the number of images in the dataset.

        returns:
          len (int): number of images in the dataset
        """
        return len(self.meta_df)

        #raise NotImplementedError

    def __getitem__(self, idx):
        """
        Fetches the image and corresponding label for a given index.

        Args:
            idx (int): index of the image to fetch

        Returns:
            img_tensor (torch.Tensor): tensor of image data
            label (int): label of image
        """

        # get the bps image file name from the metadata dataframe at the given index
        image_file_name = self.meta_df.loc[idx, 'filename']
        
        # formulate path to image given the root directory (note meta.csv is in the
        # same directory as the images)

        image_file_path = os.path.join(self.meta_root_dir, image_file_name)
        # If on_prem is False, then fetch the image from s3 bucket using the get_bytesio_from_s3
        # function, get the contents of the buffer returned, and convert it to a  numpy array
        # with datatype unsigned 16 bit integer used to represent microscopy images.

        if not self.file_on_prem:
            temp_bytesio = get_bytesio_from_s3(self.s3_client, self.bucket_name, image_file_path)
            image_numpy_array = np.frombuffer(temp_bytesio.getvalue(), dtype=np.uint8)

        
        # If on_prem is True load the image from local. 
        else:  
            image_numpy_array = cv2.imread(image_file_path, cv2.IMREAD_ANYDEPTH)
        
        # apply tranformation if available
        if self.transform:
            image_numpy_array = self.transform(image_numpy_array)

        for x in image_numpy_array:
            print(x.dtype)
        # return the image and associated label
        return torch.Tensor(image_numpy_array), self.meta_df.loc[idx, 'particle_type']
        
        #raise NotImplementedError


def show_label_batch(image: torch.Tensor, label: str):
    """Show image with label for a batch of samples."""
    images_batch, label_batch = \
            image, label
    batch_size = len(images_batch)
    im_size = images_batch.size(2)
    grid_border_size = 2

    # grid is a 4 dimensional tensor (channels, height, width, number of images/batch)
    # images are 3 dimensional tensor (channels, height, width), where channels is 1
    # utils.make_grid() takes a 4 dimensional tensor as input and returns a 3 dimensional tensor
    # the returned tensor has the dimensions (channels, height, width), where channels is 3
    # the returned tensor represents a grid of images
    grid = utils.make_grid(images_batch)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    plt.title(f"Label: {label}")
    plt.savefig('test_grid_1_batch.png')



def main():
    """main function to test PyTorch Dataset class (Make sure the directory structure points to where the data is stored)"""
    bucket_name = "nasa-bps-training-data"
    s3_path = "Microscopy/train"
    s3_meta_csv_path = f"{s3_path}/meta.csv"

    
    #define directory
    csv_dir = str(root).replace("download-dataset-augmentations-dataloader-max9001" , "data/DataFolder")
    
    #define .csv name
    csv_name = "meta.csv"

    #### testing get file functions from s3 ####

    local_file_path = "../data/raw"
    local_train_csv_path = "../data/processed/meta_dose_hi_hr_4_post_exposure_train.csv"


    #### testing dataset class ####
    train_csv_path = 'meta_dose_hi_hr_4_post_exposure_train.csv'
    training_bps = BPSMouseDatasetLocal(train_csv_path, '../data/processed', transform=None, file_on_prem=True)
    print(training_bps.__len__())
    print(training_bps.__getitem__(0))

    transformed_dataset = BPSMouseDataset(train_csv_path,
                                           '../data/processed',
                                           transform=transforms.Compose([
                                               NormalizeBPS(),
                                               ResizeBPS(224, 224),
                                               VFlipBPS(),
                                               HFlipBPS(),
                                               RotateBPS(90),
                                               RandomCropBPS(200, 200),
                                               ToTensor()
                                            ]),
                                            file_on_prem=True
                                           )

    # Use Dataloader to package data for batching, shuffling, 
    # and loading in parallel using multiprocessing workers
    # Packaging is image, label
    dataloader = DataLoader(transformed_dataset, batch_size=4,
                        shuffle=True, num_workers=2)

    for batch, (image, label) in enumerate(dataloader):
        print(batch, image, label)

        if batch == 5:
            show_label_batch(image, label)
            print(image.shape)
            break

if __name__ == "__main__":
    main()
#The PyTorch Dataset class is an abstract class that is used to provide an interface for accessing all the samples
# in your dataset. It inherits from the PyTorch torch.utils.data.Dataset class and overrides two methods:
# __len__ and __getitem__. The __len__ method returns the number of samples in the dataset and the __getitem__

