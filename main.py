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
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop

# MNIST DATASET
mnist = keras.datasets.mnist
(train_images_mnist, train_labels_mnist), (test_images_mnist, test_labels_mnist) = mnist.load_data()
# reshaping.....
train_images_mnist = np.reshape(train_images_mnist, (train_images_mnist.shape[0], 28, 28, 1))
test_images_mnist = np.reshape(test_images_mnist, (test_images_mnist.shape[0], 28, 28, 1))

# A-Z DATASET
az_path = 'model/az.csv'
az_data = pd.read_csv(az_path, header=None)
az_labels = az_data.values[:, 0]
az_images = az_data.values[:, 1:]
# reshaping.........
az_images = np.reshape(az_images, (az_images.shape[0], 28, 28, 1))

# make into train set and train set
test_size = float(len(test_labels_mnist))/len(train_labels_mnist)
print(f"MNIST set size: {test_size} ")

train_images_az, test_images_az, train_labels_az, test_labels_az = train_test_split(az_images, az_labels, test_size=test_size)
train_labels_mnist = train_labels_mnist + max(az_labels)+1 # to make sure labels don't overlap
test_labels_mnist = test_labels_mnist + max(az_labels)+1
print("concatenated")
# grouping them all together
train_images = np.concatenate((train_images_az, train_images_mnist), axis=0) # concatenation in first axis or row!!!
train_labels = np.concatenate((train_labels_az, train_labels_mnist))
test_images = np.concatenate((test_images_az, test_images_mnist),axis=0)
test_labels = np.concatenate((test_labels_az, test_labels_mnist))
print("Dataset initialized")



model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 150x150 with 3 bytes color
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('cats') and 1 for the other ('dogs')
    tf.keras.layers.Dense(len(np.unique(train_labels)), activation='softmax')
])

model.compile(optimizer=RMSprop(learning_rate=1e-4),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()


train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=15,
      width_shift_range=0.1,
      height_shift_range=0.1,
      shear_range=0.1,
      zoom_range=0.2,
      horizontal_flip=False,
      fill_mode='nearest')
test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow(train_images, train_labels, batch_size=50, shuffle=True)
validation_generator = test_datagen.flow(test_images, test_labels, batch_size=50, shuffle=True)


history = model.fit(
      train_generator,
      steps_per_epoch=500,
      epochs=100,
      validation_data=validation_generator,
      validation_steps=50,
      verbose=2)
model.save('model/model')


# load model
model_path = 'model/model'
print("Loading model...")
model = load_model(model_path)
print("Done")

# loads the input image
image_path = 'handwriting_example1.jpg'
image = cv2.imread(image_path)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cropped = gray[120:,:]
blurred = cv2.GaussianBlur(cropped, (5, 5), 0)

#matplotlib inline
fig = plt.figure(figsize=(16,4))
ax = plt.subplot(1,4,1)
ax.imshow(image)
ax.set_title('original image')

ax = plt.subplot(1,4,2)
ax.imshow(gray,cmap=cm.binary_r)
ax.set_axis_off()
ax.set_title('grayscale image')

ax = plt.subplot(1,4,3)
ax.imshow(cropped,cmap=cm.binary_r)
ax.set_axis_off()
ax.set_title('cropped image')

ax = plt.subplot(1,4,4)
ax.imshow(blurred,cmap=cm.binary_r)
ax.set_axis_off()
ax.set_title('blurred image')
