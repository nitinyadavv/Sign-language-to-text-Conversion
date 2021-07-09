#!/usr/bin/env python
# coding: utf-8

# ## Importing Libraries

# In[17]:

import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,models
from tensorflow.keras.models import Sequential

# ## Checking Data

# In[2]:
import pathlib
data_dir = "data/"
data_dir = pathlib.Path(data_dir)


# In[3]:
image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)


# In[4]:
gesture_Z = list(data_dir.glob('Z/*'))
PIL.Image.open(str(gesture_Z[0]))


# ## Image and Batch Configuration
batch_size = 32
img_height = 128
img_width = 128



# ## Training Data
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  color_mode="grayscale",
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

# ## Validation Data (20%)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  color_mode="grayscale",
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)


# ## Actual Labels

# In[8]:


class_names = train_ds.class_names
print(class_names)






# ## Dataset Dimensions

# In[9]:


for image_batch, labels_batch in train_ds:
  print(image_batch.shape)
  print(labels_batch.shape)
  break


# ## Building CNN Model
normalization_layer = layers.experimental.preprocessing.Rescaling(1./255)

normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]


num_classes = 26
model = Sequential([
  layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 1)),
  layers.Conv2D(20, (5,5), padding='same', activation='relu'),
  layers.MaxPooling2D(3,3),
  layers.Conv2D(32, (5,5), padding='same', activation='relu'),
  layers.MaxPooling2D(6,6),
  layers.Dropout(0.2),
  layers.Flatten(),
  layers.Dense(26, activation='relu'),
  layers.Dense(num_classes)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# In[63]:

model.summary()


# ## Training Model

# In[64]:


epochs = 5
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)


# ## Saving Model

# In[65]:

model.save('saving_model/myModel.h5')

# ## Visualization

# In[66]:

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

