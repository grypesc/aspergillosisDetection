import matplotlib.pyplot as plt
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils.np_utils import to_categorical
from keras.optimizers import Adam
from keras.callbacks.callbacks import ModelCheckpoint

trainData = np.loadtxt('xception_train_no_aug.csv', delimiter=",")
s = np.arange(trainData.shape[0])
np.random.shuffle(s)
trainData = trainData[s]

labels = trainData[:,len(trainData[0])-1]
labels = to_categorical(labels, num_classes=3)

validationData = np.loadtxt('xception_validation.csv', delimiter=",")

model = Sequential()
model.add(Dense(2048, activation='relu', input_shape=(2048,)))
model.add(Dropout(0.2))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(3, activation='softmax'))


model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=20e-6),
              metrics=['acc'])

history = model.fit(
    x=trainData[:, :-1],
    y=labels,
    epochs=100,
    batch_size=512,
    validation_data=(validationData[:,:-1], to_categorical(validationData[:,-1], num_classes=3)),
    callbacks=[ModelCheckpoint("xceptionTop{val_loss:.4f}_{val_acc:.4f}.h5", save_best_only=True, monitor='val_loss', verbose=0, mode='auto', period=1)],
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
