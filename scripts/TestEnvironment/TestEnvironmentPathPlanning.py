import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np 
from PIL import Image, ImageFilter
import cv2
import random
import roboticstoolbox as rtb
from roboticstoolbox import DistanceTransformPlanner
import math
from spatialmath.base import *
from math import sin, cos, atan2, radians
from random import uniform
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
from scipy.interpolate import CubicSpline
from numpy.typing import ArrayLike

'''Initialising Image'''
map_path = '/Users/theresejoseph/Documents/Neural_Network_Playground/results/TestingMaps/berlin_5kmrad_0.2Line_100pdi.png'
img=np.array(Image.open(map_path).convert("L"))

pxlPerMeter= img.shape[0]/5000

img[img<255]= 0 
img[img==255]=1


def processMap(map_path, scale_percent):
    img=np.array(Image.open(map_path).convert("L"))
    imgColor=np.array(Image.open(map_path))

    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    img=cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    imgColor=cv2.resize(imgColor, dim, interpolation = cv2.INTER_AREA)

    imgSharp=cv2.filter2D(img,-1,np.array([[0, -1, 0],[-1, 5,-1],[0, -1, 0]])) 
    imgdia=np.zeros((np.shape(img)))
    imgdia[img==255]=0
    imgdia[img<255]=1
    imgdia=cv2.dilate(imgdia,np.ones((1,1),np.uint8))

    binMap=np.zeros((np.shape(img)))
    binMap[imgSharp < 255] = 0
    binMap[imgSharp==255] = 1
    
    return img, imgColor, imgdia, binMap

def findPathsthroughRandomPoints(img):

    free_spaces=[]
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j]==0:
                free_spaces.append((j,i))

    num_locations=5
    locations=random.choices(free_spaces, k=num_locations)
    print(locations)

    path=[]
    for i in range(len(locations)-1):
        dx = DistanceTransformPlanner(img, goal=locations[i+1], distance="euclidean")
        dx.plan()
        path.extend(dx.query(start=locations[i]))
        print(f"done {i+1} paths")

        outfile='/Users/theresejoseph/Documents/Neural_Network_Playground/results/testEnvMultiplePaths3_5kmrad_100pdi_0.2line.npy'
        np.save(outfile,path)

# findPathsthroughRandomPoints(img)

def rescalePath(path, img, scale, pxlPerMeter):
    #convert path to image
    path_x, path_y = zip(*path)
    pathImg=np.zeros((np.shape(img)))
    pathImg[(path_y, path_x)]=1

    # scale down path image by given percentage 
    width = int(img.shape[1] * scale)
    height = int(img.shape[0] * scale)
    dim = (width, height)
    newImg= cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    pathImgRescaled=cv2.resize(pathImg, dim, interpolation = cv2.INTER_AREA)
    
    #identify new path from rescaled image 
    # pathImgRescaled[pathImgRescaled>0]=1
    # newYpath, newXpath=np.where(pathImgRescaled==1)
    return [np.round(x*scale) for x in path_x], [np.round(y*scale) for y in path_y], newImg, pxlPerMeter*scale 

def remove_consecutive_duplicates(coords):
    # Initialize a new list to store the filtered coordinates
    filtered = []
    # Add the first element to the filtered list
    filtered.append(coords[0])
    # Loop through the remaining elements
    for i in range(1, len(coords)):
        # Check if the current element is different from the previous element
        if coords[i] != coords[i-1]:
        # If it is, add it to the filtered list
            filtered.append(coords[i])
        # if math.tan((coords[i][1]-coords[i-1][1])/((coords[i][0]-coords[i-1][0])))>= 2*np.pi :
            
    # Return the filtered list
    return filtered


# '''Original'''
pathfile='/Users/theresejoseph/Documents/Neural_Network_Playground/results/testEnvMultiplePaths3_5kmrad_100pdi_0.2line.npy'
# path_x, path_y = zip(*np.load(pathfile))
# path= remove_consecutive_duplicates(list(zip(path_x, path_y)))
# path_x, path_y = zip(*path)


'''Scaled'''
path_x, path_y, path_img, currentPxlPerMeter= rescalePath(np.load(pathfile), img, 1, pxlPerMeter)
print(f"scaled width{np.shape(path_img)[0], np.shape(path_img)[1]}, pxlPerMeter{np.shape(path_img)[0]/5000, np.shape(path_img)[1]/5000}")
plt.imshow(path_img, cmap='gray')
plt.plot(path_x, path_y,'r.')
plt.show()