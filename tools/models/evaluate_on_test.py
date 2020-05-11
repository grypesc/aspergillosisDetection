import numpy as np
import sys

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input

from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Cropping2D
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt

np.set_printoptions(threshold=sys.maxsize)


model = Sequential()
model.add(Cropping2D(cropping=((50, 50), (50, 50)), input_shape=(512, 512, 3)))
model.add(MobileNetV2(weights='imagenet', include_top=False, input_shape=(412, 412, 3), pooling='avg'))
model.add(load_model('mobilenetV2_is_fungus/#5.h5'))

imageDataGen = ImageDataGenerator(preprocessing_function=preprocess_input)
generator = imageDataGen.flow_from_directory(
    '../../data/train/fungus/1',
    target_size=(512, 512),
    batch_size=32,
    class_mode=None)
fungus_preds = model.predict_generator(generator, verbose=1)

imageDataGen = ImageDataGenerator(preprocessing_function=preprocess_input)
generator = imageDataGen.flow_from_directory(
    '../../data/train/notFungus/1',
    target_size=(512, 512),
    batch_size=32,
    class_mode=None)
not_fungus_preds = model.predict_generator(generator, verbose=1)

TP, FP, FN, TN = 0, 0, 0, 0
for index, prediction in enumerate(fungus_preds, start=0):
    if prediction[0] >= 0.5:
        TP += 1
FN = len(fungus_preds) - TP

for index, prediction in enumerate(not_fungus_preds, start=0):
    if prediction[0] < 0.5:
        TN += 1
FP = len(not_fungus_preds) - TN

print(str(TP) + '   ' + str(FP))
print(str(FN) + '   ' + str(TN))

accuracy = (TP + TN) / (len(fungus_preds) + len(not_fungus_preds))
recall = TP / len(fungus_preds)
precision = TP / (TP + FP)
f1 = 2 * precision * recall / (precision + recall)

print("Accuracy: " + str(accuracy))
print("Recall: " + str(recall))
print("Precision: " + str(precision))
print("F1 Score: " + str(f1))



from sklearn.metrics import roc_curve, auc


y_true = np.append([0 for i in range (0, not_fungus_preds.shape[0])], [1 for i in range (0, fungus_preds.shape[0])])
y_score = np.append(not_fungus_preds, fungus_preds)
fpr, tpr, _ = roc_curve(y_true, y_score)
roc_auc = auc(fpr, tpr)
print("AUC: " + str(roc_auc))


plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()

