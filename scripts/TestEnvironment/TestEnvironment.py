import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np 
from PIL import Image, ImageFilter
import cv2
import random
import roboticstoolbox as rtb
from roboticstoolbox import DistanceTransformPlanner
from typing import Tuple, Dict, Callable, List, Optional, Union, Sequence
import numpy as np
from numpy.linalg import LinAlgError
from collections import deque
import control

map_path = 'results/berlin.png'
img=np.array(Image.open(map_path).convert("L"))
img=img[:,:7200]
# img=cv2.dilate(img,np.ones((2,2),np.uint8))
scale_percent = 15 # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)
img=cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
imgwithVal=img 
img[img < 255] = 0
img[img==255] = 1

# plt.imshow(img)
# plt.show()

color2speed = {81:5, 114:6, 177:7, 219:10, 118:15, 156:17, 159:20, 171:30, 155:40, 92:50, 30:60, 61:70, 92:80, 98:100}


# s=(159,194)
# g=(938,621)
# dx = DistanceTransformPlanner(img, goal=g, distance="euclidean")
# dx.plan()
# path = dx.query(start=s)
# x, y = zip(*path)
# plt.plot(x,y,'.')
# print(path)
# plt.imshow(img)
# plt.show()

