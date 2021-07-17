import colorsys
import os,sys
from timeit import default_timer as timer

import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw

"""
call yolo modules from yolo folder here

"""
from .yolo import YOLO

class ImageDetector():
    
    
    def __init__(self, model_path, classes_path):
        try:
            if not os.path.isfile(model_path):
                print('model path does not exist')
                return None
            
            if not os.path.isfile(classes_path):
                print('class path does not exist')
                return None
            
            self.yolo = YOLO(model_path=model_path, classes_path=classes_path)
        except Exception as  e:
            print('Open Model Error:', e)
            return None
    
    # detect image function
    def detect_image(self, img_path):
        try:
            image = Image.open(img_path)
        except Exception as e:
            print('Open Image Error:', img_path, e)
            return None, None
        else:
            r_image, result = self.yolo.detect_image(image)
        
        return r_image, result
    
    def close_session(self):
        self.yolo.close_session()