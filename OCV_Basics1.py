import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import pandas as pd
import pprint as pp


#####################
# Version of OpenCV:
print(cv.__version__)
#####################

###############################################################################################

# Reading images:

# Flags:
#  1  : Loads a color image. Any transparency of image will be neglected. It is the default flag
#  0  : Loads image in grayscale mode
# -1  : Loads image as such including alpha channel
img = cv.imread('Desert.jpg', 0)
# img is of type ndarray (an n-dimensional array)
type(img)
# array properties
array = img
arraySize = img.shape
arrayTypeValues = img.dtype
arrayDim = img.ndim
arrayMemory = img[3, 4].data
# id() function returns the unique identity if the object
id(img)
# res = cv.resize(img,None,fx=2, fy=2, interpolation = cv.INTER_CUBIC)
# Create a window and then display an image
# Flags :
# WINDOW_NORMAL   : To resize the image
# WINDOW_AUTOSIZE :
# cv.namedWindow('image', cv.WINDOW_NORMAL)
cv.imshow('image', img)
cv.waitKey(0)
cv.destroyAllWindows()

###############################################################################################

# Determining image dimensions

dimensions = img.shape

height = dimensions[0]
width = dimensions[1]

print('Image Dimension    : ', dimensions)
print('Image Height       : ', height)
print('Image Width        : ', width)

###############################################################################################
