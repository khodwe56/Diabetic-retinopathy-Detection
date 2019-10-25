import cv2
import os
import numpy as np


class DataLoader:

	def __init__(self,preprocessor_methods = None):

		self.preprocessor_methods = preprocessor_methods

		if self.preprocessor_methods is None:
			self.preprocessor_methods = []


	def loader(self,path_of_image,verbose = -1):
		
		images = []
		classes = []

		for verI,image_path in enumerate(path_of_image):
			
			img = cv2.imread(image_path)
			label_of_image = image_path.split(os.path.sep)[-2]

			if self.preprocessor_methods is not None:
				for preprocessor in self.preprocessor_methods:
					img = preprocessor.preprocess(img)

			images.append(img)
			classes.append(label_of_image)


			if verbose > 0 and verI > 0 and (verI + 1) % verbose == 0:
				print("[INFO] images processed {}/{}".format(verI + 1,len(path_of_image)))	

		return (np.array(images),np.array(classes))			
