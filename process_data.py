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
# Clustering Eight Colors to Recognition
colors = ['black', 'white', 'red', 'green']

# Convert one label to matrix 1D: truth label is 1 and last is 0.
def encoding(Y):
    encode = np.array([[1 if colors[j] == Y[i] else 0 for j in range(len(colors))] for i in range(len(Y))])
    return encode

def decoding(Y):
    decode = np.where(Y == 1)
    return decode

def softmax(Z):
    """

    Compute softmax values for each sets of scores in Z.
    each column of Z is a set of score.    
    """
    e_Z = np.exp(Z)
    sum = np.sum(e_Z, axis = 1).reshape(len(Z), 1)
    A = e_Z / sum
    return A

# Open file
def open_file(string='max'):
    """
        # Open file and save 4 features in list.

    :param string: algorithms max or average.
    :return: Return list
    """
    with open('data_max.csv' if string == 'max' else 'data_average.csv', 'r') as file:
        reader = csv.reader(file)
        list = []
        for read in reader:
            list.append(read)

    return list


# Return number color . Example return number red image , blue image ,.....
def num_color(list):
    """

    :param list:
    :return: 
    """
    len_color = []
    i, length = 0, 0

    for index in range(len(list)):
        if list[index][3] == colors[i]:
            length += 1
            if index == len(list) - 1:
                len_color.append(length)
        else:
            len_color.append(length)
            i += 1
            length += 1
    return len_color


# Find new cluster color : red , orange , blue , green , .... (change cluster present)

def new_cluster(list, list_data):
    """

    :param list: list colors
    :param list_data: list data
    :return: new cluster.
    """
    cluster_new = []

    for i in range(len(colors)):
        sum_blue, sum_green, sum_red = 0, 0, 0
        count = 0
        for j in range(len(list)):
            if list[j] == colors[i]:
                count += 1
                sum_blue += float(list_data[j][0])
                sum_green += float(list_data[j][1])
                sum_red += float(list_data[j][2])

        sum_blue, sum_green, sum_red = sum_blue / count, sum_green / count, sum_red / count
        cluster_new.append([sum_blue, sum_green, sum_red])
    return cluster_new


# Calculator norm p = 2 or distance Euclid.
def norm(X, Y, p=2):
    """
        Norm is calculate dimension X and dimensions Y with norm = p distance.
        X = [x1,x2,...,xn]
        Y = [y1,y2,...,yn]
        distance = [|x1-y1| ** p + |x2-y2| ** p + ... + |xn-yn| ** p] ** (1/p)
        The most common norm we use is p = 1, p = 2, p = 3 , p = infinity.


    :param X: n dimension
    :param Y: n dimension
    :param p: p is norm
    :return: List distance with X,Y and p
    """
    list = []
    # Algorithms distance Euclid.
    for i in range(len(X)):
        sum = 0
        for j in range(0, 3):
            sum += (X[i][j] - int(Y[j])) ** p
        list.append(sum)
    return list


# Return name colors of image to function self.name_color.
def color_image(path_file_image):
    """

    :param path_file_image:
    :return: image have colors red, black, green, blue , vvvv?
    """
    for index in range(len(colors)):
        if colors[index] in path_file_image:
            return colors[index]


# Resize image original to image have size (width = 64, length = 64) because Laptop easy calculator
def resize_image(img):
    """

    :param img: Image source : maybe image gray or image RGB or image BGR ,.....

    :return: Resize image source width width and length -> width = 64, length = 64
            - Note: Image source width and length above 64.
    """
    return cv2.resize(img, (64, 64))


# Count histogram
def count_histogram(image):
    """

    :param image:
    :return:
    """
    # Create init matrix numpy size(3,256) is zeros.
    histogram = np.zeros((3, 256), dtype=np.int64)
    image = np.array(image, dtype=np.int64)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            """
            image[i, j, 0] += random.randrange(-4, 4)
            image[i, j, 1] += random.randrange(-4, 4)
            image[i, j, 2] += random.randrange(-4, 4)
            if image[i, j, 0] < 0:
                image[i, j, 0] = 0
            if image[i, j, 1] < 0:
                image[i, j, 1] = 0
            if image[i, j, 2] < 0:
                image[i, j, 2] = 0
            if image[i, j, 0] > 255:
                image[i, j, 0] = 255
            if image[i, j, 1] > 255:
                image[i, j, 1] = 255
            if image[i, j, 2] > 255:
                image[i, j, 2] = 255
            """
            histogram[0, image[i, j, 0]] += 1
            histogram[1, image[i, j, 1]] += 1
            histogram[2, image[i, j, 2]] += 1
    return histogram


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
        # Resize image.
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

    # Convert jpg or png - > matrix.
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
