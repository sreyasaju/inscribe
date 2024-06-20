import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import imutils
from imutils.contours import sort_contours
from matplotlib import cm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop

# MNIST dataset
mnist = keras.datasets.mnist
(train_images_mnist, train_labels_mnist), (test_images_mnist, test_labels_mnist) = mnist.load_data()
# reshaping the dataset
print("Reshaping MNIST data...")
train_images_mnist = np.reshape(train_images_mnist, (train_images_mnist.shape[0],28, 28, 1))
test_images_mnist = np.reshape(test_images_mnist, (test_images_mnist.shape[0], 28, 28, 1))

# A-Z dataset
az_path = 'model/az.csv'
az_data = pd.read_csv(az_path, header=None)
az_labels = az_data.values[:, 0]  # to take labels form the first column of az_data
az_images = az_data.values[:, 1:]
az_labels = az_labels.astype(int)
# reshaping....
print("Reshaping A-Z data...")
az_images = np.reshape(az_images, (az_images.shape[0], 28, 28,1))

test_size = len(test_labels_mnist) / len(train_labels_mnist)
print(f"MNIST test set size: {test_size:.4f}")
# split that csv into train set and test set
train_images_az, test_images_az, train_labels_az, test_labels_az = train_test_split(
    az_images, az_labels, test_size=test_size, random_state=42)
print("Data set split into train and test sets.")
# to prevent any clashing and overlapping...
# A-Z will be 0-25, and MNIST will be 26-35
train_labels_mnist = train_labels_mnist + 26
test_labels_mnist = test_labels_mnist + 26

# grouping 'em all together...
print("Concatenating...")
train_images = np.concatenate((train_images_az, train_images_mnist), axis=0)
train_labels = np.concatenate((train_labels_az, train_labels_mnist), axis=0)
test_images = np.concatenate((test_images_az, test_images_mnist), axis=0)
test_labels = np.concatenate((test_labels_az, test_labels_mnist), axis=0)
print("Concatenated!")
print("Dataset ready.")

"""
Plot first 7 images from both datasets
fig, axes = plt.subplots(1, 10, figsize=(20, 4))
for i in range(10):
    # Plot MNIST images
    axes[i].imshow(train_images_mnist[i].squeeze(), cmap='gray')  # Squeeze to remove channel dimension for grayscale
    axes[i].set_title(f"MNIST Label: {train_labels_mnist[i]}")
    axes[i].axis('off')

plt.tight_layout()
plt.show()

# Plot first 7 images from A-Z dataset
fig, axes = plt.subplots(1, 10, figsize=(20, 4))

for i in range(10):
    # Plot A-Z images
    axes[i].imshow(train_images_az[i].squeeze(), cmap='gray')  # Squeeze to remove channel dimension for grayscale
    axes[i].set_title(f"A-Z Label: {train_labels_az[i]}")
    axes[i].axis('off')
plt.tight_layout()
plt.show()
"""