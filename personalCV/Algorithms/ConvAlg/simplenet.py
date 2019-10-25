from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation,Flatten,Dense
from keras import backend as K


class SimpleNet:

	def create_model(width, height, depth, no_of_classes):
		
		model = Sequential()
		input_shape = (height, width, depth)

		if K.image_data_format() == "channels_first":
			input_shape = (depth,height,width)

		model.add(Conv2D(32, (3, 3), padding="same",input_shape=input_shape))
		model.add(Activation("relu"))
		model.add(Flatten())
		model.add(Dense(no_of_classes))
		model.add(Activation("softmax"))	

		return model

