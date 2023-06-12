"""
Prototype for PyTorch model for BPS data
"""
import os
from typing import Any
# import warnings
# warnings.filterwarnings("ignore")
import pyprojroot
root = pyprojroot.find_root(pyprojroot.has_dir(".git"))
import sys
sys.path.append(str(root))
from datetime import datetime
from sklearn.metrics import accuracy_score
from src.dataset.bps_dataset import BPSMouseDataset
from src.dataset.bps_datamodule import BPSDataModule
from torchmetrics import Accuracy

import wandb
import torch
from torch.optim import Adam
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from torch.nn import CrossEntropyLoss

import boto3
from botocore import UNSIGNED
from botocore.config import Config

from dataclasses import dataclass
from datetime import datetime
import io
from io import BytesIO
from PIL import Image
import numpy as np
import random

# from wandb tutorial
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
from torch.optim import Adam
from torchmetrics.functional import accuracy
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import Callback, ModelCheckpoint


@dataclass
class BPSConfig:
    """ Configuration options for BPS Microscopy dataset.

    Args:
        data_dir: Path to the directory containing the image dataset. Defaults
            to the `data/processed` directory from the project root.

        train_meta_fname: Name of the training CSV file.
            Defaults to 'meta_dose_hi_hr_4_post_exposure_train.csv'

        val_meta_fname: Name of the validation CSV file.
            Defaults to 'meta_dose_hi_hr_4_post_exposure_test.csv'
        
        save_dir: Path to the directory where the model will be saved. Defaults
            to the `models/SAP_model` directory from the project root.

        batch_size: Number of images per batch. Defaults to 4.

        max_epochs: Maximum number of epochs to train the model. Defaults to 3.

        accelerator: Type of accelerator to use for training.
            Can be 'cpu', 'gpu', 'tpu', 'ipu', 'auto', or None. Defaults to 'auto'
            Pytorch Lightning will automatically select the best accelerator if
            'auto' is selected.

        devices: Number of devices to use for training. Defaults to 1.
    """
    data_dir:           str = root / 'data' / 'processed'
    train_meta_fname:   str = 'meta_dose_hi_hr_4_post_exposure_train.csv'
    val_meta_fname:     str = 'meta_dose_hi_hr_4_post_exposure_test.csv'
    save_dir:           str = root / 'models' / 'RESNET_50'
    batch_size:         int = 16
    max_epochs:         int = 15
    accelerator:        str = 'auto'
    devices:            int = 1
    num_workers:        int = 12
   

current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
wandb_logger = WandbLogger(project='deeplearning-eda-saddleback', log_model="all",
                               entity='saddleback', name=f'resnet50_5epoch_{current_datetime}',
                               save_dir= root / 'models' / 'RESNET_50',
                               )


class BPSClassifier(LightningModule):
    def __init__ (self, model, n_classes=2, lr=1e-3):

        super().__init__()
        self.model = model

        self.loss = CrossEntropyLoss()
        self.accuracy_fn = Accuracy(task='binary', num_classes=2)
        self.lr = lr
        self.save_hyperparameters
        # self.wb_step = 0

        self.epoch_confmat = None

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        _, loss, acc = self._get_pred_loss_acc(batch)
        
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        self.log("train_accuracy", acc, on_step=False, on_epoch=True)

        return loss

    # ===============================================================================
    #=============== PER BATCH LOGGING OF CONFUSION MATRIX ==========================
    def validation_step(self, batch, batch_idx):
        preds, loss, acc = self._get_pred_loss_acc(batch)
        y = batch[1]

        # Plotting a confusion matrix that is always overwritten
        confmat = wandb.plot.confusion_matrix(y_true=torch.argmax(y, dim=1).numpy(),
                                                      preds=preds.numpy(),
                                                      class_names=['Fe', 'Xray'])
        wandb.log({"conf_mat": confmat})

        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.log("val_accuracy", acc, on_step=False, on_epoch=True)

        return preds

    # ===============================================================================
        

    # def validation_step(self, batch, batch_idx):
    #     preds, loss, acc = self._get_pred_loss_acc(batch)
    #     y = batch[1]

    #     # Accumulate predictions and true labels across batches
    #     if self.epoch_confmat is None:
    #         self.epoch_confmat = [torch.argmax(y, dim=1).numpy(), preds.numpy()]
    #     else:
    #         self.epoch_confmat[0] = np.concatenate((self.epoch_confmat[0], torch.argmax(y, dim=1).numpy()))
    #         self.epoch_confmat[1] = np.concatenate((self.epoch_confmat[1], preds.numpy()))

    #     self.log("val_loss", loss, on_step=False, on_epoch=True)
    #     self.log("val_accuracy", acc, on_step=False, on_epoch=True)
        
    #     return preds
    
    # def on_validation_epoch_end(self):
    #     # Generate and save the confusion matrix at the end of the epoch
    #     confmat = wandb.plot.confusion_matrix(y_true=self.epoch_confmat[0],
    #                                         preds=self.epoch_confmat[1],
    #                                         class_names=['Fe', 'Xray'])
    #     wandb.log({"conf_mat": confmat}, step=self.current_epoch)

    #     # Reset epoch_confmat for the next epoch
    #     self.epoch_confmat = None    

    def test_step(self, batch, batch_idx):
        _, loss, acc = self._get_preds_loss_acc(batch)
        self.log("test_loss", loss)
        self.log("test_accuracy", acc)
    
    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr)
        return optimizer

    def _get_pred_loss_acc(self, batch):
        x, y = batch
        logits= self.forward(x)

        loss = self.loss(logits, y)
        preds = torch.argmax(logits, dim=1)

        # This matches the tutorial
        # accuracy_fn = Accuracy(task='binary', num_classes=2)
        
        # acc = self.accuracy_fn(preds, y)
        acc = self.accuracy_fn(logits, y)

        return preds, loss, acc

class LogPredictionsCallback(Callback):
    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx):
        """Called when the validation batch ends."""
 
        # `outputs` comes from `LightningModule.validation_step`
        # which corresponds to our model predictions in this case
        
        # Let's log 20 sample image predictions from first batch
        if batch_idx == 0:
            n = 20
            x, y = batch
            images = [img for img in x[:n]]
            captions = [f'Ground Truth: {y_i[1]} - Prediction: {y_pred}' for y_i, y_pred in zip(y[:n], outputs[:n])]
            
            # Option 1: log images with `WandbLogger.log_image`
            wandb_logger.log_image(key='sample_images', images=images, caption=captions)

            # Option 2: log predictions as a Table
            columns = ['image', 'ground truth', 'prediction']
            data = [[wandb.Image(x_i), y_i[1], y_pred] for x_i, y_i, y_pred in list(zip(x[:n], y[:n], outputs[:n]))]
            wandb_logger.log_table(key='sample_table', columns=columns, data=data)



def main():
    my_config = BPSConfig()

    # Set seed for reproducibility 🌱 -> 🌻
    seed = 85
    torch.manual_seed(seed)
    random.seed(seed) 
    np.random.seed(seed)

    # Instantiate BPSDataModule ⌛️
    bps_datamodule = BPSDataModule(train_csv_file=my_config.train_meta_fname,
                                   train_dir=my_config.data_dir,
                                   val_csv_file=my_config.val_meta_fname,
                                   val_dir=my_config.data_dir,
                                   resize_dims=(256, 256),
                                   batch_size=my_config.batch_size,
                                   num_workers=my_config.num_workers)
    
    # Using BPSDataModule's setup, define the stage name ('train' or 'val')
    bps_datamodule.setup(stage='train') #training set dataloader
    bps_datamodule.setup(stage='validate') #validation set dataloader

    # Load weights for model
    # Uncomment the model you want to use
    # Be sure to change the last layer for the model you choose
    my_model = torch.hub.load\
    (
        'pytorch/vision',
        # 'resnet18',
        # 'resnet50',
        # 'resnet101',
        # 'vgg11_bn',
        'squeezenet1_1',
        weights='DEFAULT'
    )

    #  # Change last layer for RESNET
    # num_features = my_model.fc.in_features
    # my_model.fc = nn.Linear(num_features, 2)

    # Change last layer for VGG
    # num_features = my_model.classifier[6].in_features
    # my_model.classifier[6] = nn.Linear(num_features, 2)

    # Change last layer for SQUEEZENET
    final_conv = nn.Conv2d(512, 2, kernel_size=1)
    my_model.classifier._modules['1'] = final_conv
    my_model.num_classes = 2

    # Freeze pre-trained layers except for the last fully connected layer
    # This was done to prevent overfitting while checking the training_step for bugs.
    # The model still makes use of the pre-trained weights in the frozen layers,
    # but the gradients will not be updated for these layers.
    # See http://ethereon.github.io/netscope/#/gist/b21e2aae116dc1ac7b50
    for name, param in my_model.named_parameters():
        if "classifier" not in name:
            param.requires_grad_(False)  # Set requires_grad to False

    my_model = BPSClassifier(my_model)

    checkpoint_callback = ModelCheckpoint(monitor='val_accuracy', mode='max')

    log_predictions_callback = LogPredictionsCallback()
    trainer = Trainer(
        logger=wandb_logger,
        callbacks=[log_predictions_callback, checkpoint_callback],
        accelerator=my_config.accelerator,
        max_epochs=my_config.max_epochs
    )
    trainer.fit(my_model, bps_datamodule.train_dataloader(), bps_datamodule.val_dataloader())
    wandb.finish()


if __name__ == "__main__"\
:
    main()