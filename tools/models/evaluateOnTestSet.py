import numpy as np
import sys

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input

from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Cropping2D

np.set_printoptions(threshold=sys.maxsize)

from keras.utils.np_utils import to_categorical

model = Sequential()
model.add(Cropping2D(cropping=((50, 50), (50, 50)), input_shape=(512, 512, 3)))
model.add(MobileNetV2(weights='imagenet', include_top=False, input_shape=(412, 412, 3), pooling='avg'))
model.add(load_model('mobilenetV2_is_fungus/mobileNetV2Top_is_fungus_0.5659_0.7149_106.h5'))

imageDataGen = ImageDataGenerator(preprocessing_function=preprocess_input)
generator = imageDataGen.flow_from_directory(
    '../../data/test/fungus',
    target_size=(512, 512),
    batch_size=32,
    class_mode=None)
fungusPreds = model.predict_generator(generator, verbose=1)

imageDataGen = ImageDataGenerator(preprocessing_function=preprocess_input)
generator = imageDataGen.flow_from_directory(
    '../../data/test/notFungus',
    target_size=(512, 512),
    batch_size=32,
    class_mode=None)
notFungusPreds = model.predict_generator(generator, verbose=1)

TP, FP, FN, TN = 0, 0, 0, 0
for index, prediction in enumerate(fungusPreds, start=0):
    if prediction[0] >= 0.5:
        TP += 1
FN = len(fungusPreds) - TP

for index, prediction in enumerate(notFungusPreds, start=0):
    if prediction[0] < 0.5:
        TN += 1
FP = len(notFungusPreds) - TN

print(str(TP) + '   ' + str(FP))
print(str(FN) + '   ' + str(TN))

accuracy = (TP + TN) / (len(fungusPreds) + len(notFungusPreds))
recall = TP / len(fungusPreds)
precision = TP / (TP + FP)
f1 = 2 * precision * recall / (precision + recall)

print("Accuracy: " + str(accuracy))
print("Recall: " + str(recall))
print("Precision: " + str(precision))
print("F1 Score: " + str(f1))
