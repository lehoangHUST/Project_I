"""
    IMPORT:

        + cv2 : Use to read image (rows,cols,3) - > Convert three channels = [Blue , Green , Red].
        + np : Calculator matrix.
        + os : Irrelevant file in computer.
        + csv : Write four feature for file csv.
        + random : Applications in choose random of list.

"""

import cv2
import numpy as np
import os
import csv
import random

# Read path file simple.
current_file = os.getcwd() + '\\' + 'training_dataset\\'

# Classification Eight Colors to Recognition
colors = ['black', 'blue', 'red', 'orange', 'yellow', 'white', 'green', 'violet']


# Return name colors of image to function self.name_color.
def color_image(image):
    for index in range(len(colors)):
        if colors[index] in image:
            return colors[index]

# Class Image - > 4 feature : Red_channels , Blue_channels , Green_channels , Name color of image reading.


# Find max histogram three channels.
class Image:

    # Create init feature of image.
    def __init__(self, path_image):
        self.name_color = color_image(path_image)
        self.path = path_image
        self.read_path_img()

    # Convert jpg or png - > matrix
    def read_path_img(self):

        image = resize_image(cv2.imread(self.path))

        '''
            Read image and give array histogram.
            histogram[0,:] -> Identity gray 0 - > 255 (channels blue)     
            histogram[1,:] -> Identity gray 0 - > 255 (channels green)     
            histogram[2,:] -> Identity gray 0 - > 255 (channels red)     
        '''

        # Count histogram from 0 to 255 in image three channels.
        histogram = count_histogram(image)

        # Return max of channel Blue
        for index in range(0, 256):
            if histogram[0][index] == max(histogram[0, :]):
                self.blue = index
                break

        # Return max of channel Green
        for index in range(0, 256):
            if histogram[1][index] == max(histogram[1, :]):
                self.green = index
                break

        # Return max of channel Red
        for index in range(0, 256):
            if histogram[2][index] == max(histogram[2, :]):
                self.red = index
                break


# Calculator Average.
class Image_Average:

    # Create init feature of image.
    def __init__(self, path_image):
        self.name_color = color_image(path_image)
        self.path = path_image
        self.read_path_img()

    # Convert jpg or png - > matrix
    def read_path_img(self):
        # Read image
        image = resize_image(cv2.imread(self.path))

        '''
            Read image and give array histogram.
            histogram[0,:] -> Identity gray 0 - > 255 (channels blue)     
            histogram[1,:] -> Identity gray 0 - > 255 (channels green)     
            histogram[2,:] -> Identity gray 0 - > 255 (channels red)     
        '''

        # Count histogram from 0 to 255 in image three channels.
        histogram = count_histogram(image)
        list = []

        # Find average histogram three channels.
        for i in range(3):
            sum = 0
            for j in range(256):
                sum += (j + 1) * histogram[i, j]
            list.append(sum / (image.shape[0] * image.shape[1]))

        self.blue, self.green, self.red = int(list[0]), int(list[1]), int(list[2])