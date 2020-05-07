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

results = {}
for i in range(1, 6):
    results[i] = []

for i in range(1, 6):
    train_data = np.loadtxt('mobileNetV2_train_#' + str(i) + '.csv', delimiter=",")
    s = np.arange(train_data.shape[0])
    np.random.shuffle(s)

    train_data = train_data[s]
    train_X = train_data[:, :-1]

    validation_data = np.loadtxt('mobileNetV2_validation_#' + str(i) + '.csv', delimiter=",")
    validation_X = validation_data[:, :-1]

    # pca = PCA(n_components=80)
    # pca = pca.fit(train_X)
    # train_X = pca.transform(train_X)
    # validation_X = pca.transform(validation_data[:, :-1])

    # tree = RandomForestClassifier(n_estimators=100)
    # tree = tree.fit(train_X, train_data[:, -1])
    # print(tree.score(train_X, train_data[:, -1]))
    # print(tree.score(validation_X, validation_data[:, -1]))
    # exit()
    model = Sequential()
    # model.add(Dropout(0.5, input_shape=(1280,)))
    # model.add(Dense(1280, activation='relu', kernel_initializer='he_uniform', input_shape=(1280,), kernel_regularizer=regularizers.l2(0.1), bias_regularizer=regularizers.l2(0.1)))
    # model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu', kernel_initializer='he_uniform', input_shape=(1280,), kernel_regularizer=regularizers.l2(0.1), bias_regularizer=regularizers.l2(0.1)))
    model.add(Dense(1, activation='sigmoid'))
    # model.summary()

    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(lr=50e-6),
                  metrics=['acc'])

    history = model.fit(
        x=train_X,
        y=train_data[:, -1],
        epochs=150,
        batch_size=1024,
        validation_data=(validation_X, validation_data[:, -1]),
        # callbacks=[ModelCheckpoint("mobileNetV2Top_is_fungus_{val_loss:.4f}_{val_acc:.4f}_{epoch:02d}.h5", save_best_only=False, monitor='val_acc',
        #                            verbose=0, mode='auto', period=1)],
        verbose=0)

    results[i].append(np.max(history.history['val_acc'][3:]))
    best_val_acc_index = history.history['val_acc'].index(results[i][0])
    results[i].append(history.history['acc'][best_val_acc_index])
    results[i].append(best_val_acc_index)

for i in range(1, 6):
    print(str(i) + ": " + str(results[i]))

print()
print([np.mean([results[i][0] for i in range(1, 6)]), np.mean([results[i][1] for i in range(1, 6)])])