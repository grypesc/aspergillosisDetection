import numpy as np
import os
import sys
np.set_printoptions(threshold=sys.maxsize)

from keras.models import load_model
from keras.utils.np_utils import to_categorical

model = load_model('mobilenetV2_is_fungus/amobileNetV2Top0.3484_0.8699.h5')

test_X = np.zeros(shape=(len(self._images), 512, 512, 3), dtype="float32")
for index in range(0, len(self.images)):
    img = image.load_img(os.path.join(self._images_directory, self.images[index].name),
                         target_size=(512, 512, 3))
    test_X[index] = image.img_to_array(img)
print(test_X[0, 200])
test_X = preprocess_input(test_X)
predictions = self._model.predict(test_X, verbose=1)

testY = np.ones(testX.shape[0])
testY = to_categorical(testY, num_classes= 2)

model = load_model('../../app/resources/models/example.h5')
yPredictions = model.predict(x=testX, batch_size=32, verbose=1)
print(yPredictions)
accuracy = np.sum(yPredictions[:,1])/fungusImages
print(accuracy)
