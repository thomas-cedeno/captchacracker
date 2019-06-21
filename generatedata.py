from captcha.image import ImageCaptcha
from PIL import Image as img

import os
import os.path
import glob
import cv2
import imutils
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd


import tensorflow as tf
import IPython.display as display
from tensorflow.python.data import Dataset

"""
CAUTION with this set of code:
    Since the we are using the library of ImageCaptcha to generate our images to train and test on,
    this means this model for machine learning is susceptible to overfitting
Help and credit goes to Adam Geitgey, with his article, for help in getting this to work
    https://medium.com/@ageitgey/how-to-break-a-captcha-system-in-15-minutes-with-machine-learning-dbebb035a710
Instructions for running this:
    1.Run this python module to generate the dataset
    2.Run the trainmodel.py module in order to generate our trained model
    3.Profit???
"""

def resize_to_fit(image, width, height):
    """
    A helper function to resize an image to fit within a given size
    :param image: image to resize
    :param width: desired width in pixels
    :param height: desired height in pixels
    :return: the resized image
    """

    # grab the dimensions of the image, then initialize
    # the padding values
    (h, w) = image.shape[:2]

    # if the width is greater than the height then resize along
    # the width
    if w > h:
        image = imutils.resize(image, width=width)

    # otherwise, the height is greater than the width so resize
    # along the height
    else:
        image = imutils.resize(image, height=height)

    # determine the padding values for the width and height to
    # obtain the target dimensions
    padW = int((width - image.shape[1]) / 2.0)
    padH = int((height - image.shape[0]) / 2.0)

    # pad the image then apply one more resizing to handle any
    # rounding issues
    image = cv2.copyMakeBorder(image, padH, padH, padW, padW,
        cv2.BORDER_REPLICATE)
    image = cv2.resize(image, (width, height))

    # return the pre-processed image
    return image

def generateRandomText(numberOfLetters):
    #Generate a random 4 letter alpha numerical string
    #Possibly add more letters later on? This would probably increase the complexity of our model, so I'm keeping it to a limited 
    #subset for now
    possibleLetters = ['A','B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
                        'X', 'Y', 'Z', '2', '3', '4', '5', '6', '7', '8', '9', '0']
    returnString = ""
    for i in range(0, numberOfLetters):
        x = random.randint(0,possibleLetters.__len__()-1)
        returnString += possibleLetters[x]
    return returnString    

def generateDataset(trainingSize):
    #Array for storing our labels
    labels = []
    for i in range(0, trainingSize):
        #Generate the captcha image from the ImageCaptcha library
        labels.append(generateRandomText(1))

        imageGenerator = ImageCaptcha()
        image = imageGenerator.generate_image(labels[i])
        image.save("./rawcaptchas/%s.png" % labels[i])

        #-----------------------------------------------------------    
        #Code for cropping, formating and showing what our generated CAPTCHAS look like 
        # Load the image and convert it to grayscale
        
        image = cv2.imread("./rawcaptchas/%s.png" % labels[i])
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        try:
            os.remove("./rawcaptchas/%s.png" % labels[i])
        except:
            pass
        # threshold the image (convert it to pure black and white)
        threshImage = cv2.threshold(grayImage, 20, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

        
        # find the contours (continuous blobs of pixels) the image
        contours = cv2.findContours(threshImage.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Hack for compatibility with different OpenCV versions
        contours = contours[0] if imutils.is_cv2() else contours[1]

        
        for contour in contours:
            # Get the rectangle that contains the contour
            (x, y, w, h) = cv2.boundingRect(contour)
            #We want to pass over all the little dots and lines in our generated CAPTCHAS
            if(w < 20 or h < 20):
                continue

            #Pass over letters that are obviously too large - probably took in the entire image so this specific example would be 
            #basically useless
            if(w > 75):
                continue
            
            # Extract the letter from the original image
            letterImage = grayImage[y:y + h, x:x + w]
            letterImage = cv2.threshold(letterImage, 20, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
            cv2.imwrite(("./letters/%s%d.png" % (labels[i],i)), letterImage)
        #-----------------------------------------------------------
    return



generateDataset(10000)