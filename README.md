[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/xrP3eqMC)


<div align="center">

<img alt="Lightning" src="https://pl-public-data.s3.amazonaws.com/assets_lightning/LightningColor.png" width="800px" style="max-width: 100%;">

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)


![](https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-28.svg)


<div align="left">

# Team Name
**final-project-saddleback**

- @max9001
- @campjake

## Quickstart: Files to Run

### Get the data
- `src/dataset/bps_datamodule.py`

### Visualize with dimensionality reduction (PCA, TSNE, UMAP)
- `src/model/unsupervised/pca_tsne.py`
- `src/model/unsupervised/resnet101.py`

### Training and Validation with Pretrained Weights
- `src/model/supervised/bps_classifier.py`
- ### [View our results on Weights & Biases](https://wandb.ai/saddleback/deeplearning-eda-saddleback)


# More info on the files

## Dimensionality Reduction Techniques for Image Feature Embedding Intuition
### Principal Component Analysis (PCA)
PCA is a popular dimensionality reduction technique in computer vision. It can be used to reduce the number of features in an image dataset without losing too much information. This can be useful for tasks such as object recognition and image classification. PCA can also be used to find the most important features in an image dataset. This can be done by calculating the eigenvalues of the covariance matrix of the dataset. The eigenvalues represent the variance of each feature. The features with the largest eigenvalues are the most important features. In other words, PCA can be used to compress images by reducing the number of pixels in each image. This can be done by finding the principal components of the image and then projecting the image onto the subspace spanned by the first few principal components.

Since the most important features have the most variance, PCA focuses more on preserving global trends in the data and less on preserving local relationships between specific points.

We will visualize the feature embeddings found by PCA by fitting the compressed representation of the image to t-SNE plot which will allow us to visualize our images as points on a 2D and 3D plane which we can then color based on the target label of particle type.

### ResNet101: A Pre-trained Feature Extractor
![resnet101](documentation/images/resnet_architecture.png)

ResNet101 has been pretrained on large-scale image classification datasets, such as ImageNet. As a result, the network has learned to extract meaningful and discriminative features from images. By utilizing a pretrained ResNet101 model, you can leverage its learned features as a starting point for extracting image features, which can be useful in downstream tasks like dimensionality reduction for BPS microscopy images. We will download the state dictionary of the model from online, inherit from the original class in PyTorch and remove the final fully connected layer, and use the output of the final convolutional layer to investigate the features that the model has learned. We can download pretrained models that have been trained on large image datasets from `torch.hub`.

## Visualizing High Dimensional Data in Lower Dimensions
### T-distributed Stochastic Neighbor Embedding (t-SNE) 
T-SNE is a non-linear dimensionality reduction technique that is commonly used for visualization of high-dimensional data. It was developed by Laurens van der Maaten and Geoffrey Hinton in 2008.

t-SNE works by finding a low-dimensional representation of the data that preserves the local structure of the high-dimensional data. This means that points that are close together in the high-dimensional space will also be close together in the low-dimensional space. In other words, t-SNE preserves local patterns.

t-SNE is a popular choice for visualizing high-dimensional data in computer vision because it can be used to visualize data that is too high-dimensional to be visualized directly. For example, t-SNE can be used to visualize the features of images, such as the color and texture of the pixels.

We will be using t-SNE to visualize the results of our raw images, our PCA data representation, as well as our pre-trained deep learning representation for both particle_type labels.

## References:
- [Image t-SNE](https://notebook.community/ml4a/ml4a-guides/notebooks/image-tsne)
- [Faces Dataset Decomposition Tutorial](https://github.com/olekscode/Examples-PCA-tSNE/blob/master/Python/Faces%20Dataset%20Decomposition.ipynb) 
- [Using t-SNE for Data Visualisation](https://medium.com/analytics-vidhya/using-t-sne-for-data-visualisation-8a83f46fbad3)
- [Pytorch: Models and pre-trained Weights](https://pytorch.org/vision/stable/models.html)
    - [SQUEEZENET](https://pytorch.org/vision/stable/models/squeezenet.html)
    - [RESNET](https://pytorch.org/vision/stable/models/resnet.html)
    - [VGG](https://pytorch.org/vision/stable/models/vgg.html)
