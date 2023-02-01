# -*- coding: utf-8 -*-
"""nn_project.ipynb

---
<center><h1>Sieci neuronowe i algorytmy uczenia</h1></center>
<center><h2>Temat: Rozpoznawanie liter</h2></center>

---
"""

!pip install emnist
from emnist import extract_training_samples
from emnist import extract_test_samples

import tensorflow as tf
from tensorflow.keras import layers as lyrs
from tensorflow.keras.utils import plot_model

from matplotlib import pyplot as plt
import numpy as np
import random

train_images, train_labels = extract_training_samples('letters')
test_images, test_labels = extract_test_samples('letters')

train_images = tf.keras.utils.normalize(train_images, axis=1)
test_images = tf.keras.utils.normalize(test_images, axis=1)

img_height = train_images.shape[1]
img_width = train_images.shape[2]
classes_number = 27
train_number = train_images.shape[0]
test_number = test_images.shape[0]
print('Number of train images: ', train_number, '\nNumber of test images: ', test_number)

plt.imshow(train_images[0], cmap=plt.cm.binary)
plt.axis('off')
plt.show()
print(train_images[0])

tf.keras.backend.set_floatx('float64')

# Model 1 - 87.8% accuracy (3 epochs), but fast
model = tf.keras.models.Sequential([
          lyrs.Flatten(),
          lyrs.Dense(128, activation=tf.nn.relu),
          lyrs.Dense(128, activation=tf.nn.relu),
          lyrs.Dense(classes_number, activation=tf.nn.softmax)
])

# Model 2 - 90.71% accuracy (3 epochs), but slower
train_images = train_images.reshape(train_images.shape[0], img_height, img_width, 1)
test_images = test_images.reshape(test_images.shape[0], img_height, img_width, 1)

model = tf.keras.models.Sequential([
          lyrs.Conv2D(32, kernel_size=(3, 3), activation='relu',input_shape=(img_height, img_width, 1)),
          lyrs.MaxPooling2D(pool_size=(2, 2)),
          lyrs.Flatten(),
          lyrs.Dense(128, activation='relu'),
          lyrs.Dense(classes_number, activation='softmax')
])

# Model 3 - 93.58% accuracy (3 epochs)
tf.keras.backend.set_floatx('float32')
train_images = train_images.reshape(train_images.shape[0], img_height, img_width, 1)
test_images = test_images.reshape(test_images.shape[0], img_height, img_width, 1)

model = tf.keras.models.Sequential([
         lyrs.Conv2D(32, (5, 5), input_shape=(img_height, img_width, 1)),
         lyrs.BatchNormalization(axis=-1),
         lyrs.Activation('relu'),
         lyrs.Conv2D(32, (4, 4)),
         lyrs.BatchNormalization(axis=-1),
         lyrs.Activation('relu'),
         lyrs.MaxPooling2D(pool_size=(2, 2)),

         lyrs.Conv2D(64, (3, 3)),
         lyrs.BatchNormalization(axis=-1),
         lyrs.Activation('sigmoid'),
         lyrs.Conv2D(64, (3, 3)),
         lyrs.BatchNormalization(axis=-1),
         lyrs.Activation('relu'),
         lyrs.MaxPooling2D(pool_size=(2, 2)),

         lyrs.Flatten(),

         # Fully connected layer
         lyrs.Dense(512),
         lyrs.BatchNormalization(),
         lyrs.Activation('relu'),
         lyrs.Dropout(0.2),

         lyrs.Dense(classes_number, activation='softmax')
])

# Model 4 - 92.52% accuracy (3 epochs)
tf.keras.backend.set_floatx('float32')
train_images = train_images.reshape(train_images.shape[0], img_height, img_width, 1)
test_images = test_images.reshape(test_images.shape[0], img_height, img_width, 1)

model = tf.keras.models.Sequential([
         lyrs.Conv2D(filters=32, kernel_size=(3, 3), input_shape=(img_height, img_width, 1), activation='relu'),
         lyrs.Conv2D(filters=64, kernel_size=(3, 3), input_shape=(img_height, img_width, 1), activation='relu'),
         lyrs.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="same"),
         lyrs.Conv2D(filters=64, kernel_size=(3, 3), input_shape=(img_height, img_width, 1), activation='relu'),
         lyrs.Conv2D(filters=64, kernel_size=(3, 3), input_shape=(img_height, img_width, 1), activation='relu'),
         lyrs.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="same"),
         lyrs.Dropout(0.2),
         lyrs.Flatten(),
         lyrs.Dense(units=128, activation='relu', kernel_initializer='uniform'),
         lyrs.Dense(units=64, activation='relu', kernel_initializer='uniform'),
         lyrs.Dense(units=classes_number, activation='softmax', kernel_initializer='uniform')
])

# Reload model before start
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


val_split = 0.16
print('Number of train images: ', train_number, '\nNumber of validation images: ', int(train_number*val_split))
sequential_model = model.fit(train_images, train_labels, validation_split=val_split, epochs=5, shuffle=False, batch_size=124800)

model.summary()

plot_model(model, to_file='model.png')

# Plot training and validation accuracy values
plt.plot(sequential_model.history['accuracy'])
plt.plot(sequential_model.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training and validation loss values
plt.plot(sequential_model.history['loss'])
plt.plot(sequential_model.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

validation_lose, validation_accuracy = model.evaluate(test_images, test_labels)
print('Test images validation:')
print('Validation lose: ', validation_lose, '\nValidation accuracy:', validation_accuracy)

model.save('letters.model')

input_model = tf.keras.models.load_model('letters.model')

predictions = input_model.predict(test_images)

labels_letter = {0:'', 1:'A', 2:'B', 3:'C', 4:'D', 5:'E', 6:'F', 7:'G', 8:'H', 9:'I', 10:'J', 11:'K', 12:'L', 13:'M',
                 14:'N', 15:'O', 16:'P', 17:'Q', 18:'R', 19:'S', 20:'T', 21:'U', 22:'V', 23:'W', 24:'X', 25:'Y', 26:'Z', 27:''}

test_images = test_images.reshape(test_images.shape[0], img_height, img_width)
rows = 5
cols = 4
figure, axis = plt.subplots(rows, cols)
plt.subplots_adjust(right=2, top=2, hspace=0.5)

for i in range(0,rows):
  for j in range(0,cols):
    sample = random.randint(0, test_images.shape[0])
    axis[i, j].imshow(test_images[sample], cmap=plt.cm.binary)
    axis[i, j].set_title('Label: {0}, Prediction: {1}'.format(labels_letter[test_labels[sample]], labels_letter[np.argmax(predictions[sample])]))
    axis[i, j].axis('off')
plt.show()

"""# Do prezentacji:
---
## Wstęp:
- technologie Python, TensorFlow, Keras,
- środowisko Google Colab,
- zbiór EMNIST Letters (wielkie/małe litery, mogą być obrócone) 

## Do zbadania (wpływ na czas trwania uczenia i jego dokładność):
- wpływ obrazów (0-255) i znormalizowanych (0-1),
- wpływ liczby epok,
- wpływ przetasowanania danych (shuffle) lub jego braku,
- wpływ modelu w Keras (rodzaje warstw i ich kolejność),
- wpływ procentowego podziału danych na zbiór uczący i testowy,
- wpływ zmiennej batch size,
- z jakimi literami sieć ma największe problemy w rozpoznaniu (I - L),
- predykcja własnoręcznie napisanych i zeskanowanych liter (czarne na białym tle i białe na czarnym tle),
- kiedy nastąpi przeuczenie zbioru. 

## Teoria ([Link](https://towardsdatascience.com/epoch-vs-iterations-vs-batch-size-4dfb9c7ce9c9))
- One Epoch is when an ENTIRE dataset is passed forward and backward through the neural network only ONCE.
- Batch Size - total number of training examples present in a single batch.
- Iterations is the number of batches needed to complete one epoch.


"""