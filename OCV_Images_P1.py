import numpy as np
import cv2 as cv
import logging as log

# OpenCV - Getting started with images

# Function to convert img and show/save


def DispImg(ImageName, ConvertTo, SaveFlag):
    converterFlag = ConvertTo
    imgToConvert = cv.imread(ImageName, converterFlag)
    if SaveFlag is True:
        cv.imwrite('DesertConvert.jpg', imgToConvert)
    else:
        cv.namedWindow('ConvertedImage', cv.WINDOW_NORMAL)
        cv.imshow('ConvertedImage', imgToConvert)
        cv.waitKey(0)


# Function to draw shapes on images

# for rectangle pass the top-left corner and bottom-right
# corner points of rectangle.
# for circle you need its center coordinates and radius.


def DrawShapes(ImageName, TypShape, Thickness):
    imgToDraw = cv.imread(ImageName, 1)
    dimensions = imgToDraw.shape
    height = dimensions[0]
    width = dimensions[1]
    if TypShape == 0:
        imgRect = cv.rectangle(
            imgToDraw, (0, 200), (100, 100), (0, 255, 0), Thickness)
        cv.namedWindow('RectImage', cv.WINDOW_NORMAL)
        cv.imshow('RectImage', imgRect)
        cv.waitKey(0)
    if TypShape == 1:
        imgCircle = cv.circle(
            imgToDraw, (300, 300), 69, (255, 0, 0), Thickness)
        cv.namedWindow('CircleImage', cv.WINDOW_NORMAL)
        cv.imshow('CircleImage', imgCircle)
        cv.waitKey(0)

# Function to write text on images


def WriteText(ImageName, TextOnImage, Size):
    font = cv.FONT_HERSHEY_SIMPLEX
    imgWithText = cv.imread(ImageName, 1)
    newImage = cv.putText(
        imgWithText, TextOnImage,
        (0, 0), font, Size, (100, 100, 100), 4, cv.LINE_AA)
    cv.namedWindow('TextImage', cv.WINDOW_NORMAL)
    cv.imshow('TextImage', newImage)
    cv.waitKey(0)

WriteText('Desert.jpg', TextOnImage='The Great Desert', Size=4)

# using the functions:

# DispImg('Desert.jpg', ConvertTo=0, SaveFlag=True)
