# Libraries
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf 
from tensorflow import keras
from PIL import Image
from sklearn.model_selection import train_test_split
from keras.utils import normalize
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense

# Load Dataset
image_dir = "Brain_DataSet/"


no_tumor_images = os.listdir(image_dir + "No/")
yes_tumor_images = os.listdir(image_dir + "Yes/")
dataset = []
label = []
input_size = 64

# Preprocess image
for i, image_name in enumerate(no_tumor_images):
    image = cv2.imread(image_dir + "No/" + image_name)
    image = Image.fromarray(image, mode='RGB')
    image = image.resize((input_size, input_size))
    dataset.append(np.array(image))
    label.append(0)
        

for i, image_name in enumerate(yes_tumor_images):
    image = cv2.imread(image_dir + "Yes/" + image_name)
    image = Image.fromarray(image, mode='RGB')
    image = image.resize((input_size, input_size))
    dataset.append(np.array(image))
    label.append(1)
        


dataset = np.array(dataset)
label = np.array(label)


# Split the dataset
x_train, x_test, y_train, y_test = train_test_split(dataset, label, test_size = 0.2, random_state = 0)

x_train = normalize(x_train, axis=1)
x_test = normalize(x_test, axis=1)



# Model Building
model = Sequential()

model.add(Conv2D(32, (3,3), input_shape = (input_size, input_size, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32, (3,3), kernel_initializer="he_uniform"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32, (3,3), kernel_initializer="he_uniform"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation("sigmoid"))


model.compile(loss = "binary_crossentropy", optimizer="adam", metrics=["accuracy"])
history = model.fit(x_train,y_train, batch_size=16, verbose=1,
          epochs=25, validation_data=(x_test, y_test),
          shuffle= False)


## Accuracy graph
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()


# Loss graph
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.legend()
plt.show()



img_path = "Brain_DataSet/Yes/y18.jpg"

img = cv2.imread(img_path)
img = cv2.resize(img, (64,64))
img = np.expand_dims(img, axis = 0)
output = model.predict(img)
print(output[0][0])

if (output[0][0] > 0.5):
  plt.title("Brain is affected by 'Tumour'", color="red")
  plt.imshow(cv2.imread(img_path))
  plt.show()
else:
  plt.title("Brain is 'Not' affected by 'Tumour'", color="green")  
  plt.imshow(cv2.imread(img_path))
  plt.show()
