import imutils
import cv2

class AspectRatio:

	def __init__(self, width, height, inter=cv2.INTER_AREA):
		
		self.width = width
		self.height = height
		self.inter = inter

	def preprocess(self,img):
		
		(h, w) = img.shape[:2]
		dwidth = 0
		dheight = 0

		if w < h:
			img = imutils.resize(img, width=self.width,inter=self.inter)
			dheight = int((img.shape[0] - self.height) / 2.0)
		else:
			img = imutils.resize(img, height=self.height,inter=self.inter)
			dwidth = int((img.shape[1] - self.width) / 2.0)
		(h, w) = img.shape[:2]
		img = img[dheight:h - dheight, dwidth:w - dwidth]
		
		return cv2.resize(image, (self.width, self.height),interpolation=self.inter)			
