# Lecture 1

### The difference between traditional programming and Machine Learning
- Traditional programming takes rules and data in and get the results.
- Machine learning takes results and data in and get the rules.
- Machine learning is all about a computer learning the patterns that distinguish things

## Basics
- Keras is an api in tensorflow library
- we need to define layers, neuron and input shape when we create a neural network
- when compile a neural network we need to define a loss function and an optimizer
    - The LOSS function measures the guessed answers against the known correct answers and measures how well or how badly it did.
    - OPTIMIZER function to make another guess
- we train the model using the fit call
- Overfitting
    -  occurs when the network learns the data from the training set really well, but it's too specialised to only that data, and as a result is less effective at seeing other data.
```python
import tensorflow as tf
import numpy as np
from tensorflow import keras

# the Graded function
def house_model(y_new):
    xs = np.array()
    ys = np.array()
    model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
    model.compile(optimizer='sgd', loss='mean_squared_error')
    model.fit(xs, ys, epochs=500)
    return model.predict(y_new[0])
```

## Computer vision (Deep Neural Networks)
```python
import tensorflow as tf


class MyCallback(tf.keras.callbacks.Callback):
    def onEpochEnd(self, epoch, logs={}):
        if logs.get('acc') >= 0.9:
            print("\nReached target so cancelling training!")
            self.model.stop_training = True


callbacks = MyCallback()
mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()
training_images = training_images / 255.0
test_images = test_images / 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
# too many epochs might cause Overfitting problem
# while training accuracy is quite good but bad on the test set
model.fit(training_images, training_labels, epochs=5, callbacks=[callbacks])

```

## CNN (Convolutional Neural Networks)
- Convolution
    - narrow down the content of the image to focus on specific and distinct details. hence information passed drops but accuracy may increased
    - caculate the pixel value of current one by use the sum of its neighbors * their weights
- Pooling
    - a way of compressing an image
    - e.g. convert 16 pixels to 4 pixels (by choosing the biggest one in very 2 * 2 square)

CNN Examples:
```Python
import tensorflow as tf
print(tf.__version__)

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, log={}):
    if log.get('accuracy') >= 0.98:
      print("\nReached the target accuracy now cancelling")
      self.model.stop_training = True

callbacks = myCallback()
mnist = tf.keras.datasets.mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()
# the convolution layer must see all the images in the first place
training_images=training_images.reshape(60000, 28, 28, 1)
training_images=training_images / 255.0
test_images = test_images.reshape(10000, 28, 28, 1)
test_images=test_images/255.0
model = tf.keras.models.Sequential([
  # add enough convolutions could help improve the performance
  # but too many convolutions will decrease the performance as it loss too many
  # features.
  tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(28, 28, 1)),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Conv2D(16, (3,3), activation='relu'),
  tf.keras.layers.MaxPool2D(2, 2),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(training_images, training_labels, epochs=5, callbacks=[callbacks])
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(test_acc)
```

Code using matplot to see the process:
```Python
import matplotlib.pyplot as plt
f, axarr = plt.subplots(3,4)
FIRST_IMAGE=0
SECOND_IMAGE=23
THIRD_IMAGE=28
CONVOLUTION_NUMBER = 3
from tensorflow.keras import models
layer_outputs = [layer.output for layer in model.layers]
activation_model = tf.keras.models.Model(inputs = model.input, outputs = layer_outputs)
for x in range(0,4):
  f1 = activation_model.predict(test_images[FIRST_IMAGE].reshape(1, 28, 28, 1))[x]
  axarr[0,x].imshow(f1[0, : , :, CONVOLUTION_NUMBER], cmap='inferno')
  axarr[0,x].grid(False)
  f2 = activation_model.predict(test_images[SECOND_IMAGE].reshape(1, 28, 28, 1))[x]
  axarr[1,x].imshow(f2[0, : , :, CONVOLUTION_NUMBER], cmap='inferno')
  axarr[1,x].grid(False)
  f3 = activation_model.predict(test_images[THIRD_IMAGE].reshape(1, 28, 28, 1))[x]
  axarr[2,x].imshow(f3[0, : , :, CONVOLUTION_NUMBER], cmap='inferno')
  axarr[2,x].grid(False)
```
