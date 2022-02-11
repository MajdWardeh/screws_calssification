import os
import math
import numpy as np
import cv2
from PIL import Image, ImageEnhance

from tensorflow.keras.utils import Sequence

from tensorflow.keras.preprocessing.image import random_shift, random_zoom

class DataGenerator(Sequence):
    '''
        A custom data generator to generate training/validation/testing samples with augmentation capabilities.
    '''
    def __init__(self, x_set, y_set, config={}):
        self.x_set = x_set
        self.y_set = y_set
        self.__process_config(config)
        
    def __process_config(self, config):
        target_image_size = config.get('target_image_size', (90, 335))
        self.target_height, self.target_width = target_image_size
        self.batch_size = config.get('batch_size', 32)

        self.resize_fill_mode = config.get('resize_fill_mode', 'nearest')
        self.random_horizontal_flip = config.get('random_horizontal_flip', 0)
        self.random_vertical_flip = config.get('random_vertical_flip', 0)

        self.random_shift_prob = config.get('random_shift_prob', 0)
        self.horizontal_shift_amount = config.get('horizontal_shift_amount', 1./10.)
        self.vertical_shift_amount = config.get('vertical_shift_amount', 1./10.)

        self.random_zoom_prob = config.get('random_zoom_prob', 0)
        self.zoom_range = config.get('zoom_range', (0.1, 0.1))
        
        self.random_brightness_shift = config.get('random_brightness_shift', False)
        self.brightness_shift_std = config.get('brightness_shift_std', 0.2)
        self.fill_mode = config.get('fill_mode', 'nearest')

    def __len__(self):
        return math.ceil(len(self.x_set) / self.batch_size)

    def __resize_with_horizontal_padding(self, x):
        if x.shape[0:1] != (self.target_height, self.target_width):
            f = self.target_width/x.shape[1]
            x = cv2.resize(x, (0, 0), fx=f, fy=f)
            if x.shape[0] < self.target_height:
                diff = self.target_height - x.shape[0]
                i1 = diff // 2
                i2 = diff - i1
                tmp_img = np.zeros((self.target_height, self.target_width, 3), dtype=np.uint8)
                tmp_img[i1:-i2, :, :] = x
                if self.resize_fill_mode == 'nearest':
                    tmp_img[:i1, :, :] = x[0, :, :]
                    tmp_img[-i2:, :, :] = x[-1, :, :]
                elif self.resize_fill_mode == 'constant':
                    pass
                else:
                    raise ValueError('resize_fill_mode value {} is not recognized, possible values are "constant" or "nearest"'.format(self.resize_fill_mode))
                x = tmp_img
            elif x.shape[0] > self.target_height:
                diff = x.shape[0] - self.target_height
                i1 = diff // 2
                i2 = diff - i1
                tmp_img = np.zeros((self.target_height, self.target_width, 3), dtype=np.uint8)
                tmp_img = x[i1:-i2, :, :]
                x = tmp_img
            return x
    
    def __getitem__(self, index):
        X_batch = []
        y_batch = []
        for row in range(min(self.batch_size, len(self.x_set)-index*self.batch_size)):
            image_name = self.x_set[index*self.batch_size + row]
            x = np.array(Image.open(image_name)) # color channels order is RGB


            ## resizing:
            x = self.__resize_with_horizontal_padding(x)

            ## apply random horizontal/vertical flips
            if self.random_horizontal_flip != 0 and np.random.rand() <= self.random_horizontal_flip:
                x = cv2.flip(x, 1)
            if self.random_vertical_flip != 0 and np.random.rand() <= self.random_vertical_flip:
                x = cv2.flip(x, 0)
                    
            ## apply random shift on the horizontally and/or vertically
            if self.random_shift_prob != 0.0 and np.random.rand() <= self.random_shift_prob:
                x = random_shift(x, self.horizontal_shift_amount, self.vertical_shift_amount, row_axis=0, col_axis=1, channel_axis=2,
                        fill_mode=self.fill_mode, cval=0)
            

            ## apply random zoom in/out
            if self.random_zoom_prob != 0.0 and np.random.rand() <= self.random_zoom_prob:
                x = random_zoom(x, self.zoom_range, row_axis=0, col_axis=1, channel_axis=2,
                        fill_mode=self.fill_mode, cval=0.0, interpolation_order=1) 

            ## apply random brightness shift 
            if self.random_brightness_shift:
                img = Image.fromarray(x)
                br_enhancer = ImageEnhance.Brightness(img)
                img_enhanced = br_enhancer.enhance(np.random.normal(1.0, self.brightness_shift_std)) 
                x = np.array(img_enhanced)

            X_batch.append(x)
            if not self.y_set is None:
                y = self.y_set[index*self.batch_size + row]
                y_batch.append(y)

        X_batch = np.array(X_batch).astype(np.float32)/255.0
        if self.y_set is None:
            return X_batch
        else:
            y_batch = np.array(y_batch).astype(np.float32)
            return (X_batch, y_batch)


