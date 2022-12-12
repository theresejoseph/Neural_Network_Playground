
import matplotlib.pyplot as plt 
import numpy as np 
import PIL.Image 
import time 

img=np.asarray(PIL.Image.open("./data/dog.jpeg").convert("L"))
plt.imshow(img,cmap="gray")


kernel=np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])

def conv(img,kernel):
    kh, kw = kernel.shape
    ih, iw = img.shape

    output = np.zeros((ih-kh+1,iw-kw+1))
    for i in range(0,ih-kh+1):
        for j in range(0,iw-kw+1):
            output[i,j]=sum(sum(img[i:i+kh, j:j+kw]*kernel))
            
    return output 

conv_img=conv(img,kernel)
plt.imshow(conv_img,cmap="gray")
plt.show()

