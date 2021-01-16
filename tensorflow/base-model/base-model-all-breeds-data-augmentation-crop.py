import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import xml.etree.ElementTree as et
import os
from PIL import Image
import sys
import time

# ==========================================
# Download Dataset
# ==========================================
image_dataset_url = "http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar"
images_dir = tf.keras.utils.get_file("Images",
                                     image_dataset_url,
                                     untar=True)
annotations_dataset_url = "http://vision.stanford.edu/aditya86/ImageNetDogs/annotation.tar"
annotations_dir = tf.keras.utils.get_file("Annotation",
                                          annotations_dataset_url,
                                          untar=True)

print("Images:", images_dir)
print("Annotation:", annotations_dir)

# ==========================================
# Crop Images
# ==========================================
dataset_images_dir = "cropped"

if not os.path.exists(dataset_images_dir):
    os.mkdir(dataset_images_dir)
    breed_list = os.listdir(images_dir)
    counter = 1
    counter_max = len(breed_list)

    for breed in breed_list:
        os.mkdir(dataset_images_dir + "/" + breed)
        print("Created ({}/{}): {}/{}"
              .format(counter, counter_max, dataset_images_dir, breed))
        counter += 1
        for file in os.listdir(annotations_dir + "/" + breed):
            img = Image.open(images_dir + "/" + breed + "/" + file + ".jpg")
            xml_tree = et.parse(annotations_dir + "/" + breed + "/" + file)
            x_min = int(xml_tree.getroot().findall('object')[0]
                        .find('bndbox').find('xmin').text)
            x_max = int(xml_tree.getroot().findall('object')[0]
                        .find('bndbox').find('xmax').text)
            y_min = int(xml_tree.getroot().findall('object')[0]
                        .find('bndbox').find('ymin').text)
            y_max = int(xml_tree.getroot().findall('object')[0]
                        .find('bndbox').find('ymax').text)
            img = img.crop((x_min, y_min, x_max, y_max))
            img = img.convert('RGB')
            img.save(dataset_images_dir + "/" + breed + "/" + file + ".jpg")

# ==========================================
# Load dataset into Tensorflow
# ==========================================
img_height = 300
img_width = 300
batch_size = 32

# Dataset for training
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    directory=dataset_images_dir,
    validation_split=0.2,
    subset="training",
    seed=6214,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# Dataset for validation
validation_ds = tf.keras.preprocessing.image_dataset_from_directory(
    directory=dataset_images_dir,
    validation_split=0.2,
    subset="validation",
    seed=9423,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# Print parsed dog breeds (classes)
class_names = train_ds.class_names
num_classes = len(class_names)
print("{} classes: {}".format(num_classes, class_names))

# Display some examples from the training dataset
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
plt.show()

# ==========================================
# Data Augmentation
# ==========================================
data_augmentation = tf.keras.Sequential(
  [
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.2, input_shape=(img_height, img_width, 3)),
    tf.keras.layers.experimental.preprocessing.RandomZoom(0.3),
  ]
)

# ==========================================
# Define model
# ==========================================
# Architecture example from https://www.tensorflow.org/tutorials/images/classification
model = tf.keras.Sequential([
    data_augmentation,
    tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255),
    tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_classes)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.summary()

# ==========================================
# Train model
# ==========================================
epochs = 10
history = model.fit(
    train_ds,
    validation_data=validation_ds,
    epochs=epochs
)

# ==========================================
# Visualize Training
# ==========================================
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# ==========================================
# Test model against unknown image from the web
# ==========================================
test_url = "https://www.telegraph.co.uk/content/dam/science/2017/09/10/TELEMMGLPICT000107300056_trans_NvBQzQNjv4BqyuLFFzXshuGqnr8zPdDWXiTUh73-1IAIBaONvUINpkg.jpeg"
test_path = tf.keras.utils.get_file("test", origin=test_url)

img = tf.keras.preprocessing.image.load_img(test_path, target_size=(img_height, img_width))
img_array = tf.keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(class_names[np.argmax(score)], 100 * np.max(score))
)
