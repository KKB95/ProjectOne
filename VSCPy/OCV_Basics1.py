import numpy as np
import cv2 as cv
import pandas as pd
import pprint as pp
import OCV_Images_P1

# Plotting libraries
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors
from mpl_toolkits.mplot3d import Axes3D

import sys
import os
sys.path.append('/ProjectG/ProjectOne/VSCPy')


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
img = cv.imread("ImagesTesting\Desert.jpg")
imgConvert = cv.cvtColor(img, cv.COLOR_BGR2RGB)
# img is of type ndarray (an n-dimensional array)
type(img)

# array properties
array = img
arraySize = img.shape
arrayTypeValues = img.dtype
arrayDim = img.ndim
imgSize = img.size
arrayMemory = img[3, 4].data

# id() function returns the unique identity if the object
id(img)

# res = cv.resize(img,None,fx=2, fy=2, interpolation = cv.INTER_CUBIC)
# Create a window and then display an image
# Flags :
# WINDOW_NORMAL   : To resize the image
# WINDOW_AUTOSIZE :
cv.namedWindow('image', cv.WINDOW_NORMAL)
cv.imshow('image', imgConvert)
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

# Color spaces in images

flags = [i for i in dir(cv) if i.startswith('COLOR_')]
len(flags)
flags[1:200]

# Convert image to using YCbCr color space:


# 


##############################################################################################

# Plot 3D : image color distribution of pixels

imgflower = cv.imread('ImagesTesting\Chrysanthemum.jpg', cv.IMREAD_COLOR)
imgflower.ndim
r, g, b = cv.split(imgflower)
fig = plt.figure()
axis = fig.add_subplot(1, 1, 1, projection='3d')

pixel_colors = imgflower.reshape((np.shape(imgflower)[0]*np.shape(imgflower)[1], 3))
norm = colors.Normalize(vmin=-1., vmax=1.)
norm.autoscale(pixel_colors)
pixel_colors = norm(pixel_colors).tolist()

axis.scatter(r.flatten(), g.flatten(), b.flatten(), facecolors=pixel_colors, marker=".")
axis.set_xlabel("Red")
axis.set_ylabel("Green")
axis.set_zlabel("Blue")
plt.show()

#############################################################################################

# Image segmenatation

#############################################################################################
