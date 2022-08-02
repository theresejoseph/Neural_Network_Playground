import matplotlib.pyplot as plt
# import matplotlib as cm
import numpy as np
import math
import os
import pandas as pd
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D 
from scipy import signal

from CAN import activityDecoding, activityDecodingAngle, attractorNetworkSettling, attractorNetwork, multiResolution,attractorNetworkScaling, imageHistogram
from matplotlib.colors import LinearSegmentedColormap



'''Parameters'''
num_links=[4,17]
excite=[12,7]
activity_mag=[1,1]
inhibit_scale=[0.05,0.005]

# prev_weights_trans=[np.zeros(N[0]), np.zeros(N[0]), np.zeros(N[0]), np.zeros(N[0]),np.zeros(N[0])]
# prev_weights_rot=[np.zeros(N[1]), np.zeros(N[1]), np.zeros(N[1]),np.zeros(N[1]), np.zeros(N[1])]

def visualiseMultiple1DNetworks():
    N=[30,30,30] #number of neurons
    prev_weights=[np.zeros(N[0]), np.zeros(N[1]), np.zeros(N[2])]
    '''Initlise Network Activation'''
    for n in range(len(prev_weights)):
        net=attractorNetworkScaling(N[n],num_links[0],excite[0], activity_mag[0],inhibit_scale[0])
        prev_weights[n][net.activation(N[0]//2)]=net.full_weights(num_links[0])

    '''ColourMap with transparent backgroung'''
    # get colormap
    ncolors = 256
    color_array = plt.get_cmap('viridis')(range(ncolors))
    # change alpha values
    color_array[:,-1] = np.linspace(0,1,ncolors)
    # create a colormap object
    map_object = LinearSegmentedColormap.from_list(name='rainbow_alpha',colors=color_array)
    # register this new colormap with matplotlib
    plt.register_cmap(cmap=map_object)

    '''Plotting'''
    fig = plt.figure(figsize=(11, 3))
    ax0 = fig.add_subplot(1, 3, 1)
    ax1 = fig.add_subplot(1, 3, 2)
    ax2 = fig.add_subplot(1, 3, 3, projection='3d')

    '''1D'''
    im1D=imageHistogram(prev_weights[0][:],N[0],1)
    ax0.imshow(im1D,aspect='auto',cmap='rainbow_alpha')
    ax0.invert_yaxis()
    ax0.set_ylim([-40,80])
    ax0.axis('off')
    # ax0.set_title('1D')
    ax0.tick_params(left=False,bottom = False, labelleft = False,labelbottom = False)


    '''2D'''
    im2D=np.outer(prev_weights[0][:],prev_weights[1][:])
    ax1.imshow(im2D, interpolation='nearest', aspect='auto',cmap='rainbow_alpha')
    ax1.axis('off')
    # ax1.set_title('2D')
    ax1.tick_params(left=False,bottom = False, labelleft = False,labelbottom = False)

    '''3D'''
    im3D=np.zeros((N[0],N[1],N[2]))
    for j in range(N[2]):
        im3D[:,:,j]=prev_weights[2][j]*np.outer(prev_weights[0][:],prev_weights[1][:])
    xx, yy,zz = np.meshgrid(np.linspace(0,1,N[0]), np.linspace(0,1,N[1]), np.linspace(0,1,N[2]))
    ax2.scatter3D(xx, yy, zz, c=im3D, cmap='rainbow_alpha', marker='.')
    ax2.grid(False)
    ax2.axis('off')
    # ax2.set_title('3D')
    ax2.tick_params(left=False,bottom = False, labelleft = False,labelbottom = False)
    '''removing plot ticks'''
    # for line in ax2.xaxis.get_ticklines():
    #     line.set_visible(False)
    # for line in ax2.yaxis.get_ticklines():
    #     line.set_visible(False)
    # for line in ax2.zaxis.get_ticklines():
    #     line.set_visible(False)

    '''plotting stacked images'''
    # xx, yy = np.meshgrid(np.linspace(0,1,N[0]), np.linspace(0,1,N[1]))
    # cset = ax2.contourf(xx, yy, im2D, 100, zdir='z', offset=0.5)
    # ax2.set_zlim((0.,1.))

    # fig.tight_layout()
    plt.show()

visualiseMultiple1DNetworks()