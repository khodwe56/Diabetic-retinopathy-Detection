from keras.layers.core import Dense,Flatten,Dropout

class FullyConnected:

	def create_model(base_model,no_of_classes,no_of_nodes_in_FC):
		head_model = base_model.output
		head_model = Flatten(name="flatten")(head_model)
		head_model = Dense(no_of_nodes_in_FC, activation="relu")(head_model)
		head_model = Dropout(0.5)(head_model)
    	
    	return head_model