import numpy as np
import cv2

class Crop:
    
    def __init__(self,width,height,horizontal_flips = True,inter = cv2.INTER_CUBIC):
    	
    	self.width = width
    	self.horizontal_flips = horizontal_flips
    	self.height = height
    	self.inter = inter

    def preprocess(self,img):
    	
    	crops = []
    	(h, w) = img.shape[:2]
	co-ordinates = [[0, 0, self.width, self.height],[w - self.width, 0, w, self.height],[w - self.width, h - self.height, w, h],[0, h - self.height, self.width, h]]
	delta_w = int(0.5 * (w - self.width))
	delta_h = int(0.5 * (h - self.height))
	co-ordinates.append([delta_w, delta_h, w - delta_w, h - delta_h])

	if self.horizontal_flips:
		m = [cv2.flip(c, 1) for c in crops]
		crops.extend(m)
	return np.array(crops)	
		
