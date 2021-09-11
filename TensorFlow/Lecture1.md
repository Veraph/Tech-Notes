# Lecture 1

### The difference between traditional programming and Machine Learning
- Traditional programming takes rules and data in and get the results.
- Machine learning takes results and data in and get the rules.
- Machine learning is all about a computer learning the patterns that distinguish things

### Basics
- Keras is an api in tensorflow library
- we need to define layers, neuron and input shape when we create a neural network
- when compile a neural network we need to define a loss function and an optimizer
    - The LOSS function measures the guessed answers against the known correct answers and measures how well or how badly it did.
    - OPTIMIZER function to make another guess
- we train the model using the fit call
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

### Computer vision
```python
import tensorflow as tf


class MyCallback(tf.keras.callbacks.Callback):
    def onEpochEnd(self, epoch, logs={}):
        if logs.get('accuracy') >= 0.9:
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
model.fit(training_images, training_labels, epochs=5, callbacks=[callbacks])

```