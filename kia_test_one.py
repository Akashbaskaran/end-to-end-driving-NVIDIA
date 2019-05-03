import cv2
import glob
import os
import pandas as pd
import datetime
import matplotlib.pyplot as plt

import matplotlib.image as mpimg
import numpy as np

from keras.models import Sequential, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization, Lambda, Cropping2D, ELU
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.convolutional import Convolution2D
from keras import regularizers, optimizers, initializers
from keras.callbacks import EarlyStopping,ModelCheckpoint, TensorBoard, Callback
print("Done importing data")

model=load_model('/media/smartctlab/SSD2TB/akash/checkpoints/model-e007.h5')

print(model.summary())

image = cv2.imread('/media/smartctlab/SSD2TB/akash/data/crop_test/frame001207.jpg')
image = image[50:140, 0:320]
                    
                    # Resize to 200x66 pixel
image = cv2.resize(image, (200,66), interpolation=cv2.INTER_AREA)
images=np.reshape(image, (1, 66,200,3))
        
# image.reshape()
val=model.predict(images)
