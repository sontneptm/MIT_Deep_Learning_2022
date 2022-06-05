import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import RNN, Dense
from tensorflow.keras.losses import MSE
from tensorflow.keras.optimizers import Adam


x = [1,3]
y = [2,4]

model = Sequential()
model.add(Dense(4))
model.add(Dense(1))
model.compile(loss=MSE, optimizer=Adam(1.0e-3))
model.fit(x,y, epochs=100)
model.summary()

print(model.predict(x))



