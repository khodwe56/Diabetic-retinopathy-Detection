import cv2


class Resizer:

	def __init__(self,width,height,inter = cv2.INTER_AREA):

		self.width = width
		self.height = height
		self.inter = inter

	def preprocess(self,img):
		return cv2.resize(img,(self.width,self.height),interpolation = self.inter)
