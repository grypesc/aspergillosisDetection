import numpy as np

from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from keras.models import Sequential, Model, load_model
from keras.layers import Cropping2D
from keras.preprocessing.image import ImageDataGenerator


model = Sequential()
model.add(Cropping2D(cropping=((100, 100), (100, 100)), input_shape=(512, 512, 3)))
model.add(MobileNetV2(weights='imagenet', include_top=False, input_shape=(312, 312, 3), pooling='avg'))
model.add(load_model('amobileNetV2Top0.5009_0.7834.h5'))
model.save("../../../app/resources/models/mobileNetV2Top0.5009_0.7834.h5")

# validImageDataGen = ImageDataGenerator(preprocessing_function=preprocess_input)
# validGenerator = validImageDataGen.flow_from_directory(
#     '../../../data/valid/notLungs',
#     target_size=(512, 512),
#     batch_size=64,
#     class_mode='categorical')
#
# predictions = model.predict_generator(validGenerator, verbose=1)
# len = predictions.shape[0]
# predictions = np.sum(predictions[:,2])
# predictions/=len
# print(predictions)
