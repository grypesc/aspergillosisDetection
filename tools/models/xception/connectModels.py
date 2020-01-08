from keras.applications.xception import Xception
from keras.models import Sequential, Model, load_model
from keras.layers import Cropping2D
from keras.preprocessing.image import ImageDataGenerator


model = Sequential()
model.add(Cropping2D(cropping=((50, 50), (50, 50)), input_shape=(512, 512, 3)))
model.add(Xception(weights='imagenet', include_top=False, input_shape=(412, 412, 3), pooling='avg'))
model.add(load_model('xceptionTop0.5376_0.7558.h5'))
model.save("../../../app/resources/models/xceptionTop0.5653_0.7807.h5")

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
