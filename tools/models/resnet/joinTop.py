import matplotlib.pyplot as plt
import numpy as np
import os


from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, Cropping2D
from keras.utils.np_utils import to_categorical
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K

# trainImageDataGen = ImageDataGenerator(preprocessing_function=preprocess_input, horizontal_flip=False)
# trainGenerator = trainImageDataGen.flow_from_directory(
#     '../../../data/train/fungus',
#     target_size=(512, 512),
#     batch_size=64,
#     class_mode='categorical')

validImageDataGen = ImageDataGenerator()
validGenerator = validImageDataGen.flow_from_directory(
    '../../../data/valid/notLungs',
    target_size=(512, 512),
    batch_size=64,
    class_mode='categorical')

model = Sequential()
model.add(Cropping2D(cropping=((100, 100), (100, 100)),
                     input_shape=(512, 512, 3)))

resnet = ResNet50(weights='imagenet', include_top=True, input_shape=(224, 224, 3))
model.add(resnet)
topTop = load_model('resnetTop.h5')
model.add(topTop)

predictions = model.predict_generator(validGenerator, verbose=1)
len = predictions.shape[0]
predictions = np.sum(predictions[:,2])
predictions/=len
print(predictions)

model.save("../../../app/resources/models/resnet50.h5")
