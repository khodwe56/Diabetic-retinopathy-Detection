from sklearn.feature_extraction.image import extract_patches_2d

class Patch:
    
    def __init__(self,width,height):
        
        self.width = width
        self.height = height
               
    def preprocess(self,img):
        return extract_patches_2d(img,(self.height,self.width),max_patches = 1)[0]    
        