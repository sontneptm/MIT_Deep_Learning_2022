import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import MSE

"""
this is psuedocode
"""

x = None
y = None

model = Sequential()
optimizer = SGD()
weights = tf.Variable([tf.random.normal()])

while True: # loop forever
    prediction = model(x)

    with tf.GradientTape() as tape:
        loss = MSE(y, prediction)
        
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))