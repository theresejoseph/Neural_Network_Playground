
import matplotlib.pyplot as plt
import numpy as np
import math
import os
import pandas as pd
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D 
import time 
from os import listdir
import sys
sys.path.append('./scripts')
from CAN import  attractorNetworkSettling, attractorNetwork, attractorNetworkScaling, attractorNetwork2D
import CAN as can
import pykitti
import json 


'''Ablation study of scale ratios - Grid Search'''
def scaleVals(start_idx,ratio):
    start=[1,1/ratio,1/(ratio**2)]
    scale=[start[start_idx]]
    for i in range(4):
        scale.append(scale[-1]*ratio)
    return scale

def multiscale_1d_CAN_error(velocities,start_idx, ratio):
    scale=scaleVals(start_idx, ratio)
    # print(start_idx, ratio,scale)
    N=100
    num_links,excite,activity_mag,inhibit_scale=1,3,0.0721745813,2.96673372e-02
    integratedMag=[0]
    decodedMag=[0]

    prev_weights=[np.zeros(N), np.zeros(N), np.zeros(N),np.zeros(N), np.zeros(N)]
    net=attractorNetwork(N,num_links,excite, activity_mag,inhibit_scale)
    for n in range(len(prev_weights)):
        prev_weights[n][net.activation(0)]=net.full_weights(num_links)
    
    for i in range(1,len(velocities)):
        input=velocities[i]

        delta = [(input/scale[0]), (input/scale[1]), (input/scale[2]), (input/scale[3]), (input/scale[4])]
        split_output=np.zeros((len(delta)))
        '''updating network'''    
        for n in range(len(delta)):
            prev_weights[n][:]= net.update_weights_dynamics(prev_weights[n][:],delta[n])
            prev_weights[n][prev_weights[n][:]<0]=0
            split_output[n]=np.argmax(prev_weights[n][:])
        
        decoded_translation=np.sum(split_output*scale)

        integratedMag.append(integratedMag[-1]+input)
        decodedMag.append(decoded_translation)   

    fitness=np.sum(abs(np.array(integratedMag)-np.array(decodedMag)))*-1
    return fitness

def gridSearch(filename,error_func,velocities):
    start_idxs=[0,1,2]
    ratios=list(np.arange(2,11))
    error=np.zeros((len(start_idxs),len(ratios)))
    for i,stid in enumerate(start_idxs):
        for j,rat in enumerate(ratios):
            # print(mag,inh)
            t=time.time()
            error[i,j]=error_func(velocities,stid, rat)
            print(i,j,error[i,j], time.time()-t)
    with open(filename, 'wb') as f:
        np.save(f, np.array(error))

def plottingGridSearch(filename):
    start_idxs=[0,1,2]
    ratios=list(np.arange(2,11))
    with open(filename, 'rb') as f:
        error = np.load(f)
        norm_error=error/np.linalg.norm(error)

    fig, ax0=plt.subplots(figsize=(10, 7), ncols=1)
    print(error)
    ax0.set_title(filename)
    pos= ax0.imshow(error)
    fig.colorbar(pos)
    ax0.set_ylabel('Start Value')
    ax0.set_xlabel('Ratio')
    ax0.set_xticks(np.arange(len(ratios)),[round(a,4) for a in ratios])
    ax0.set_yticks(np.arange(len(start_idxs)), [1,'1/ratio','1/ratio^2'],rotation=90)
    # ax0.grid(True)

    plt.show()

filename=f'./results/AblationStudyScales/10scales_3starts.npy'
# gridSearch(filename,multiscale_1d_CAN_error,velocities)
# plottingGridSearch(filename)

