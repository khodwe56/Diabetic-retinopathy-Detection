from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from personalCV.preprocessor_methods import ImgToArray
from personalCV.preprocessor_methods import AspectRatio
from personalCV.data import DataLoader
from personalCV.Algorithms.ConvAlg import FullyConnecteed
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop
from keras.optimizers import SGD
from keras.applications import VGG16
from keras.layers import Input
from keras.models import Model
from imutils import paths
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", required=True,help="path to input data")
parser.add_argument("-m", "--model", required=True,help="path for the  output model")
arguments = vars(parser.parse_args())

augumenter = ImageDataGenerator(rotation_range=25, width_shift_range=0.2,height_shift_range=0.15, shear_range=0.25, zoom_range=0.20,horizontal_flip=True, fill_mode="nearest")

print("[INFO] loading Images into memory")
path_to_classes = list(paths.list_images(args["dataset"]))
name_of_classes = [path_.split(os.path.sep)[-2] for path_ in path_to_classes]
name_of_classes = [str(val) for val in np.unique(name_of_classes)]

ar = AspectRatio(224, 224)
iar = ImgToArr()

dl = DataLoader(preprocessor_methods=[ar, iar])

(data, labels) = dl.load(path_to_classes, verbose=500)
data = data.astype("float") / 255.0

(X_train, X_test, y_train, y_test) = train_test_split(data, labels,test_size=0.25, random_state=42)

y_train = LabelBinarizer().fit_transform(y_train)
y_test = LabelBinarizer().fit_transform(y_test)


base_model = VGG16(weights="imagenet", include_top=False,input_tensor=Input(shape=(224, 224, 3)))
head_model = FullyConnecteed.create_model(base_model, len(name_of_classes), 256)
model = Model(inputs=base_model.input, outputs=head_model)

for lyr in base_model.layers:
	lyr.trainable = False

opt = RMSprop(lr=0.002)
model.compile(loss="categorical_crossentropy", optimizer=opt,metrics=["accuracy"])

model.fit_generator(augumenter.flow(X_train, y_train, batch_size=8),validation_data=(X_test, y_test), epochs=100,steps_per_epoch=len(X_train) // 8, verbose=1)

pred = model.predict(X_test, batch_size=8)
print(classification_report(y_test.argmax(axis=1),
pred.argmax(axis=1), target_names=name_of_classes)


