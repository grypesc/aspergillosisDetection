import matplotlib.pyplot as plt
import numpy as np
from keras import regularizers

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils.np_utils import to_categorical
from keras.optimizers import Adam
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from keras.callbacks.callbacks import ModelCheckpoint

train_data = np.loadtxt('mobileNetV2_train.csv', delimiter=",")
s = np.arange(train_data.shape[0])
np.random.shuffle(s)

train_data = train_data[s]
train_X = train_data[:, :-1]


validation_data = np.loadtxt('mobileNetV2_validation.csv', delimiter=",")

# pca = PCA(n_components=800)
# pca = pca.fit(train_X)
# train_X = pca.transform(train_X)
# validation_X = pca.transform(validation_data[:, :-1])


# tree = RandomForestClassifier(n_estimators=1000)
# tree = tree.fit(train_X, train_data[:, -1])
# print(tree.score(train_X, train_data[:, -1]))
# print(tree.score(validation_X, validation_data[:, -1]))
# exit()
model = Sequential()
model.add(Dense(50, activation='relu', kernel_initializer='he_uniform', input_shape=(1280,)))
model.add(Dropout(0.5))
model.add(Dense(15, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(1, activation='sigmoid'))
model.summary()

model.compile(loss='binary_crossentropy',
              optimizer=Adam(lr=100e-6),
              metrics=['acc'])

history = model.fit(
    x=train_X,
    y=train_data[:, -1],
    epochs=150,
    batch_size=1024,
    validation_data=(validation_data[:, :-1], validation_data[:, -1]),
    callbacks=[ModelCheckpoint("mobileNetV2Top_is_fungus_{val_acc:.4f}_{val_loss:.4f}.h5", save_best_only=False, monitor='val_acc',
                               verbose=0, mode='auto', period=1)],
    verbose=2)

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
# plt.title('Dokładność modelu')
plt.ylabel('Dokładność modelu')
plt.xlabel('Epoka')
plt.legend(['Zbiór trenujący', 'Zbiór walidujący'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
# plt.title('model loss')
plt.ylabel('Funkcja kosztu')
plt.xlabel('Epoka')
plt.legend(['Zbiór trenujący', 'Zbiór walidujący'], loc='upper left')
plt.show()
