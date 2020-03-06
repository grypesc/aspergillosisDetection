import matplotlib.pyplot as plt
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils.np_utils import to_categorical
from keras.optimizers import Adam
from keras.callbacks.callbacks import ModelCheckpoint

train_data = np.loadtxt('mobileNetV2_train.csv', delimiter=",")
s = np.arange(train_data.shape[0])
np.random.shuffle(s)
train_data = train_data[s]

validation_data = np.loadtxt('mobileNetV2_validation.csv', delimiter=",")


model = Sequential()

model.add(Dense(1280, activation='relu', kernel_initializer='he_normal', input_shape=(1280,)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu', kernel_initializer='he_normal'))
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu', kernel_initializer='he_normal'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu', kernel_initializer='he_normal'))
model.add(Dropout(0.2))
model.add(Dense(3, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=200e-6),
              metrics=['acc'])

history = model.fit(
    x=train_data[:, :-1],
    y=to_categorical(train_data[:, -1], num_classes=3),
    epochs=250,
    batch_size=1024,
    validation_data=(validation_data[:, :-1], to_categorical(validation_data[:, -1], num_classes=3)),
    callbacks=[ModelCheckpoint("mobileNetV2Top{val_acc:.4f}_{val_loss:.4f}.h5", save_best_only=True, monitor='val_loss',
                               verbose=0, mode='auto', period=1)],
    verbose=2)

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
