import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# MNIST dataset
mnist = keras.datasets.mnist
(train_images_mnist, train_labels_mnist), (test_images_mnist, test_labels_mnist) = mnist.load_data()
# reshaping the dataset
print("Reshaping MNIST data...")
train_images_mnist = np.reshape(train_images_mnist, (train_images_mnist.shape[0],28, 28, 1))
test_images_mnist = np.reshape(test_images_mnist, (test_images_mnist.shape[0], 28, 28, 1))
train_images_mnist = train_images_mnist.astype('float32') / 255.0
test_images_mnist = test_images_mnist.astype('float32') / 255.0
print(f"Train images - Min: {np.min(train_images_mnist)}, Max: {np.max(train_images_mnist)}")
print(f"Test images - Min: {np.min(test_images_mnist)}, Max: {np.max(test_images_mnist)}")

# A-Z dataset
az_path = 'model/az.csv'
az_data = pd.read_csv(az_path, header=None)
az_labels = az_data.values[:, 0]  # to take labels form the first column of az_data
az_images = az_data.values[:, 1:]
az_labels = az_labels.astype(int)
az_images = np.nan_to_num(az_images)
print(f"NaN values in az_images: {np.isnan(az_images).any()}")
# reshaping....
print("Reshaping A-Z data...")
az_images = np.reshape(az_images, (az_images.shape[0], 28, 28,1))
az_images = az_images.astype('float32') / 255.0


test_size = len(test_labels_mnist) / len(train_labels_mnist)
print(f"MNIST test set size: {test_size:.4f}")
# split that AZ csv into train set and test set
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

'''
# Plotting random 5 images from MNIST dataset
plt.figure(figsize=(12, 6))
plt.suptitle('Random 5 Images from MNIST Dataset', fontsize=16)
for i in range(5):
    idx = np.random.randint(0, len(train_images_mnist))
    plt.subplot(1, 5, i + 1)
    plt.imshow(train_images_mnist[idx], cmap='gray')
    plt.title(f"Label: {train_labels_mnist[idx]}")
    plt.axis('off')

# Plotting random 5 images from A-Z dataset
plt.figure(figsize=(12, 6))
plt.suptitle('Random 5 Images from A-Z Dataset', fontsize=16)
for i in range(5):
    idx = np.random.randint(0, len(az_images))
    plt.subplot(1, 5, i + 1)
    plt.imshow(az_images[idx], cmap='gray')
    plt.title(f"Label: {az_labels[idx]}")
    plt.axis('off')
plt.show()
'''