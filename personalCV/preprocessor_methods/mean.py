import cv2

class Mean:
    
    def __init__(self,rm,gm,bm):
        
        self.rm = rm
        self.gm = gm
        self.bm = bm
        
    def preprocess(self,img):
        b,g,r = cv2.split(img.astype("float32"))
        b = b - self.bm
        g = g - self.gm
        r = r - self.rm
        return cv2.merge([b,g,r])
    
        