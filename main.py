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
from tensorflow.keras.models import load_model
from model import create_model

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
model = create_model()
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

print("Training the model...")
history = model.fit(
    train_generator,
    batch_size = 30,
    epochs = 10,
    verbose = 1,
    validation_data = validation_generator,
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)]
)

model_path = 'model/model.keras'
model.save(model_path)

print("Loading model...")
model = load_model(model_path)
print("Done!")

# Plotting accuracy and loss
plt.figure(figsize=(12, 5))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()

# Save the plots
plt.savefig('accuracy_loss_plots.png')

# Show plots
plt.show()



cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video capture device.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from camera.")
        break

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blurred, 50, 150)

# Detect contours (bounding boxes) around letters
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Process each contour (bounding box)
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)

    # Crop the ROI (region of interest) from the grayscale frame
    roi = gray[y:y + h, x:x + w]

    # Preprocess the ROI for your model (resize, normalize, reshape)
    resized_roi = cv2.resize(roi, (28, 28))  # Resize to match model's input size
    normalized_roi = resized_roi.astype("float32") / 255.0  # Normalize pixel values
    reshaped_roi = np.expand_dims(normalized_roi, axis=-1)  # Add batch dimension

    # Make prediction using your model
    prediction = model.predict(np.array([reshaped_roi]))
    predicted_label = np.argmax(prediction)

    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw bounding box
    cv2.putText(frame, str(predicted_label), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame with bounding boxes and predictions
    cv2.imshow('Frame with Predictions', frame)

    # Exit loop if ESC is pressed
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release the capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
