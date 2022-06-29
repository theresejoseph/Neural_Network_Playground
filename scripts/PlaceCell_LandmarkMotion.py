import pandas 
import cv2 

from os import listdir
from PIL import Image 

def loadImages(path):
    # return array of images

    imagesList = listdir(path)
    loadedImages = []
    for image in imagesList:
        img = Image.open(path + image)
        loadedImages.append(img)

    return loadedImages

path = "./data/2011_09_26 2/2011_09_26_drive_0001_sync/image_00/data/"

# your images in an array
imgs = loadImages(path)

for img in imgs:
    # you can show every image
    img.show()

