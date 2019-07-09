import numpy as np
import cv2 as cv
import pandas as pd
import pprint as pp
import OCV_Images_P1 as ocv

# Plotting libraries
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors
from mpl_toolkits.mplot3d import Axes3D


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
img = cv.imread("ImagesTesting/Desert.jpg", cv.IMREAD_COLOR)
imgConvert = cv.cvtColor(img, cv.COLOR_BGR2RGB)
# img is of type ndarray (an n-dimensional array)
print(img)
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
cv.imshow('image', imgConverted)
cv.waitKey(0)
cv.destroyAllWindows()

###############################################################################################

# Basic image operations using numpy
img = cv.imread("ImagesTesting/Desert.jpg", cv.IMREAD_COLOR)

ocv.LoadImageData("ImagesTesting/Desert.jpg")
px = img[20, 20]  # access pixel values of the image
pxB = img[20, 20, 0]  # access blue pixel value of the image
pxG = img[20, 20, 1]  # access green pixel value of the image
pxR = img[20, 20, 2]  # access red pixel value of the image

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
index1 = flags.index('COLOR_RGB2YCrCr')
'COLOR_RGB2YCrCb' in flags
flags[1:200]

# Convert image to using YCbCr color space:
ocv.IndexFinder('COLOR_RGB2YCrCb')
imgActual = cv.imread('ImagesTesting/Desert.jpg', cv.IMREAD_COLOR)
imgRGBC = cv.cvtColor(imgActual, cv.COLOR_BGR2RGB)
imgConverted = cv.cvtColor(imgRGBC, cv.COLOR_RGB2YCrCb)

# Convert image to using HSV color space:
ocv.IndexFinder('COLOR_RGB2HSV')
imgActual = cv.imread('ImagesTesting/Desert.jpg', cv.IMREAD_COLOR)
imgRGBC = cv.cvtColor(imgActual, cv.COLOR_BGR2RGB)
imgConverted = cv.cvtColor(imgRGBC, cv.COLOR_RGB2HSV)
ocv.LoadImageInfo(imgConverted)

ocv.JustDispImag(imgConverted)

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
# Image histogram

img = cv.imread('ImagesTesting/Desert.jpg', cv.IMREAD_COLOR)
color = ('b', 'g', 'r')
for i, col in enumerate(color):
    histr = cv.calcHist([img], [i], None, [256], [0, 256])
    plt.plot(histr, color=col)
    plt.xlim([0, 256])
plt.show()

ocv.LoadImageData('ImagesTesting/Desert.jpg')
ocv.PlotHisto('ImagesTesting/Desert.jpg')

#############################################################################################
