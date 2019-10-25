from keras.preprocessing.image import img_to_array

class ImgToArray:


	def __init__(self,data_format = None):
		self.data_format = data_format

	def preprocess(self,img):
		return img_to_array(img, data_format=self.data_format)	
