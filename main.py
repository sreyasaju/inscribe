import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import cv2
import easyocr
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

train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
print("Dataset ready.")


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

def augment(image, label):

    # Random zoom
    scales = list(np.arange(0.8, 1.2, 0.1))
    boxes = np.zeros((len(scales), 4))
    for i, scale in enumerate(scales):
        x1 = y1 = 0.5 - 0.5 * scale
        x2 = y2 = 0.5 + 0.5 * scale
        boxes[i] = [x1, y1, x2, y2]

    def random_crop(img):
        crops = tf.image.crop_and_resize([img], boxes=boxes, box_indices=np.zeros(len(scales)), crop_size=(28, 28))
        return crops[tf.random.uniform(shape=[], minval=0, maxval=len(scales), dtype=tf.int32)]

    image = random_crop(image)

    # Random brightness
    image = tf.image.random_brightness(image, max_delta=0.1)

    # Random contrast
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)

    return image, label


batch_size = 30
train_dataset = train_dataset.shuffle(buffer_size=len(train_images)).map(augment, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(batch_size).repeat()
test_dataset = test_dataset.batch(batch_size).repeat()

model = create_model()
model.summary()


print("Training the model...")
history = model.fit(
    train_dataset,
    batch_size = 30,
    steps_per_epoch=len(train_images) // batch_size,
    epochs = 10,
    verbose = 1,
    validation_data = test_dataset,
    validation_steps=len(test_images) // batch_size,
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


# Open the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video capture device.")
    exit()

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from camera.")
        break

    # Convert frame to grayscale and blur it
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Perform edge detection and find contours
    edged = cv2.Canny(blurred, 30, 150)
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=lambda c: cv2.boundingRect(c)[0])

    # Initialize list to store characters
    chars = []

    # Loop over contours
    for c in cnts:
        # Compute bounding box of contour
        (x, y, w, h) = cv2.boundingRect(c)

        # Filter out bounding boxes based on width and height
        if (w >= 5 and w <= 150) and (h >= 15 and h <= 120):
            # Extract the character region and threshold it
            roi = gray[y:y + h, x:x + w]
            thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

            # Resize the thresholded image to 28x28
            thresh = cv2.resize(thresh, (28, 28))

            # Ensure the image has a single channel (grayscale)
            thresh = np.expand_dims(thresh, axis=-1)

            # Prepare the image for classification
            thresh = thresh.astype("float32") / 255.0

            # Update list of characters
            chars.append((thresh, (x, y, w, h)))
    # Extract the bounding box locations and padded characters
    boxes = [b[1] for b in chars]
    chars_images = np.array([c[0] for c in chars], dtype="float32")

    # OCR the characters using our handwriting recognition model
    preds = model.predict(chars_images)

    # Define the list of label names
    labelNames = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    labelNames += "0123456789"
    labelNames = [l for l in labelNames]

    # Loop over the predictions and bounding box locations together
    for (pred, (x, y, w, h)) in zip(preds, boxes):
        # Find the index of the label with the largest corresponding
        # probability, then extract the probability and label
        i = np.argmax(pred)
        prob = pred[i]
        label = labelNames[i]


        # Draw the prediction on the image
        print("[INFO] {} - {:.2f}%".format(label, prob * 100))
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

    # Show the image with predictions
    cv2.imshow("Image", frame)

    # Exit loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()

