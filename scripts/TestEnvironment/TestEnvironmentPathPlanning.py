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


map_path = '../../results/TestingMaps/berlin_5kmrad_0.2Line_100pdi.png'
img=np.array(Image.open(map_path).convert("L"))

pxlPerMeter= img.shape[0]/5000

img[img<255]= 0 
img[img==255]=1

plt.imshow(img)
plt.show()

free_spaces=[]
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        if img[i][j]==0:
            free_spaces.append((j,i))

num_locations=20
locations=random.choices(free_spaces, k=num_locations)
print(locations)

path=[]
for i in range(len(locations)-1):
    dx = DistanceTransformPlanner(img, goal=locations[i+1], distance="euclidean")
    dx.plan()
    path.extend(dx.query(start=locations[i]))
    print(f"done {i+1} paths")

    outfile='../../results/testEnvMultiplePaths1_5kmrad_100pdi_0.2line.npy'
    np.save(outfile,path)