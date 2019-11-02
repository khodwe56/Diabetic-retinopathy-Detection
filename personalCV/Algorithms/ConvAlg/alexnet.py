from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.regularizers import l2
from keras import backend as K

class AlexNet:

	def create_model(width, height, depth, classes, regLambda=0.0003):

		model = Sequential()
		input_shape = (height, width, depth)
		channel_dimmension = -1

		if K.image_data_format() == "channels_first":
			input_shape = (depth,height,width)
			channel_dimmension = 1

		model.add(Conv2D(96, (11, 11), strides=(4, 4),input_shape=input_shape, padding="same",kernel_regularizer=l2(regLambda)))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=channel_dimmension))
		model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
		model.add(Dropout(0.20))

		model.add(Conv2D(256, (5, 5), padding="same",kernel_regularizer=l2(regLambda)))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=channel_dimmension))
		model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
		model.add(Dropout(0.20))

		model.add(Conv2D(384, (3, 3), padding="same",kernel_regularizer=l2(regLambda)))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=channel_dimmension))
		model.add(Conv2D(384, (3, 3), padding="same",kernel_regularizer=l2(regLambda)))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=channel_dimmension))
		model.add(Conv2D(256, (3, 3), padding="same",kernel_regularizer=l2(regLambda)))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=channel_dimmension))
		model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
		model.add(Dropout(0.20))

		model.add(Flatten())
		model.add(Dense(4096, kernel_regularizer=l2(regLambda)))
		model.add(Activation("relu"))
		model.add(BatchNormalization())
		model.add(Dropout(0.5))

		model.add(Dense(4096, kernel_regularizer=l2(regLambda)))
		model.add(Activation("relu"))
		model.add(BatchNormalization())
		model.add(Dropout(0.5))

		model.add(Dense(classes, kernel_regularizer=l2(regLambda)))
		model.add(Activation("softmax"))

		return model



			