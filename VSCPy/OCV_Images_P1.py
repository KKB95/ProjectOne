import numpy as np
import cv2 as cv
import pandas as pd
import pprint as pp
import time

# Plotting libraries
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors
from mpl_toolkits.mplot3d import Axes3D


# OpenCV - Getting started with images

# Function to convert img and show/save


def DispImg(imageName, convertTo, saveFlag):
    converterFlag = convertTo
    imgToConvert = cv.imread(imageName, converterFlag)
    if saveFlag is True:
        cv.imwrite('DesertConvert.jpg', imgToConvert)
    else:
        cv.namedWindow('ConvertedImage', cv.WINDOW_NORMAL)
        cv.imshow('ConvertedImage', imgToConvert)
        cv.waitKey(0)


# Function to draw shapes on images

# for rectangle pass the top-left corner and bottom-right
# corner points of rectangle.
# for circle you need its center coordinates and radius.


def DrawShapes(imageName, typShape, thickness):
    imgToDraw = cv.imread(imageName, 1)
    dimensions = imgToDraw.shape
    height = dimensions[0]
    width = dimensions[1]
    if typShape == 0:
        imgRect = cv.rectangle(
            imgToDraw, (0, 200), (100, 100), (0, 255, 0), thickness)
        cv.namedWindow('RectImage', cv.WINDOW_NORMAL)
        cv.imshow('RectImage', imgRect)
        cv.waitKey(0)
    if typShape == 1:
        imgCircle = cv.circle(
            imgToDraw, (300, 300), 69, (255, 0, 0), thickness)
        cv.namedWindow('CircleImage', cv.WINDOW_NORMAL)
        cv.imshow('CircleImage', imgCircle)
        cv.waitKey(0)

# Function to write text on images


def WriteText(imageName, textOnImage, size):
    font = cv.FONT_HERSHEY_SIMPLEX
    imgWithText = cv.imread(imageName, 1)
    newImage = cv.putText(
        imgWithText, textOnImage,
        (0, 0), font, size, (100, 100, 100), 4, cv.LINE_AA)
    cv.namedWindow('TextImage', cv.WINDOW_NORMAL)
    cv.imshow('TextImage', newImage)
    cv.waitKey(0)

WriteText('Desert.jpg', textOnImage='The Great Desert', size=4)

# Function to plot img


def PlotDistro(imageName, IS1, IS2, IS3):
    start = time.time()
    imgflower = cv.imread(imageName)
    imgflowerC = cv.cvtColor(imgflower, cv.COLOR_BGR2RGB)
    r, g, b = cv.split(imgflowerC)
    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1, projection='3d')

    pixel_colors = imgflowerC.reshape((np.shape(imgflowerC)[0]*np.shape(imgflowerC)[1], 3))
    norm = colors.Normalize(vmin=-1., vmax=1.)
    norm.autoscale(pixel_colors)
    pixel_colors = norm(pixel_colors).tolist()

    axis.scatter(r.flatten(), g.flatten(), b.flatten(), facecolors=pixel_colors, marker=".")
    axis.set_xlabel(IS1)
    axis.set_ylabel(IS2)
    axis.set_zlabel(IS3)
    plt.show()
    end = time.time()
    print(end - start)

# Find index of ColorConverter


def IndexFinder(indexName):
    totalFlags = [i for i in dir(cv)]
    indexValue = totalFlags.index(indexName)
    return indexValue

# using the functions:

# DispImg('Desert.jpg', ConvertTo=0, SaveFlag=True)
# PlotDistro('ImagesTesting\Desert.jpg', IS1="R", IS2="B", IS3="G")
# IndexFinder('COLOR_RGB2YCrCb')
