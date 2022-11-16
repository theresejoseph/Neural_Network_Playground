import matplotlib.pyplot as plt
import numpy as np
import math
import os
import pandas as pd
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap
from matplotlib.artist import Artist
from mpl_toolkits.mplot3d import Axes3D 
from scipy import signal
import time 
from os import listdir
import sys
sys.path.append('/home/therese/Documents/Neural_Network_Playground/scripts')
from CAN import  attractorNetworkSettling, attractorNetwork, attractorNetworkScaling, attractorNetwork2D
import CAN as can
import pykitti
import json 
from DataHandling import saveOrLoadNp      



def hierarchicalNetwork(integratedPos,decodedPos,net,input,N, iterations,wrap_iterations, wrap_mag, wrap_inhi):
    # wrap_mag=2.851745813
    # wrap_inhi=2.96673372e-02/2
    
    delta = [(input/scales[0]), (input/scales[1]), (input/scales[2]), (input/scales[3]), (input/scales[4])]
    split_output=np.zeros((len(scales)))
    

    net=attractorNetwork(N,num_links,excite, wrap_mag,wrap_inhi)
    cs_idx=scale_selection(input,scales)
    wraparound=np.zeros(len(scales))
    wraparound[cs_idx]=(can.activityDecoding(prev_weights[cs_idx][:],4,N) + delta[cs_idx])//(N-1)

    '''Update selected scale'''
    for iter in range(iterations):
        prev_weights[cs_idx][:]= net.update_weights_dynamics(prev_weights[cs_idx][:],delta[cs_idx])
        prev_weights[cs_idx][prev_weights[cs_idx][:]<0]=0

    '''Update the 100 scale based on wraparound in any of the previous scales'''
    if (cs_idx != 4) and wraparound[cs_idx]!=0:
        update_amount=(wraparound[cs_idx]*scales[cs_idx]*N)/scales[4]
        wraparound[4]=(can.activityDecoding(prev_weights[4][:],4,N) + update_amount)//(N-1)
        for iter in range(wrap_iterations):
            prev_weights[-2][:]= net.update_weights_dynamics(prev_weights[-2][:],update_amount)
            prev_weights[-2][prev_weights[-2][:]<0]=0

    '''Update the 10000 scale based on wraparound in the 100 scale'''
    if wraparound[4] !=0:
        for iter in range(wrap_iterations):
            prev_weights[-1][:]= net.update_weights_dynamics(prev_weights[-1][:],(wraparound[4]*scales[4]*N)/scales[-1])
            prev_weights[-1][prev_weights[-1][:]<0]=0

    '''Decode position'''
    split_output=np.array([can.activityDecoding(prev_weights[m][:],4,N) for m in range(len(scales))])
    decoded_translation=np.sum(((split_output))*scales)
    speeds.append(decoded_translation-decodedPos[-1])
    integratedPos.append(integratedPos[-1]+input)
    decodedPos.append(decoded_translation)   
    # print(f"translation {input} integrated decoded {round(integratedPos[-1],3)}  {str(decoded_translation )} ")

def GIF_MultiResolutionFeedthrough1D(velocities,scale, visualise=False):
    global prev_weights, speeds
    # num_links,excite,activity_mag,inhibit_scale=1,3,0.0721745813*5,2.96673372e-02*2

    integratedPos=[0]
    decodedPos=[0]
    speeds=[0]

    prev_weights=[np.zeros(N),np.zeros(N),np.zeros(N),np.zeros(N),np.zeros(N),np.zeros(N)]
    net=attractorNetwork(N,num_links,excite, activity_mag,inhibit_scale)
    for n in range(len(prev_weights)):
        prev_weights[n][net.activation(0)]=net.full_weights(num_links)

    '''initlising network and animate figures'''
    ncols=7
    fig, axs = plt.subplots(ncols,1, figsize=(10, 10))
    fig.subplots_adjust(hspace=0.9)
    fig.suptitle("Multiscale CAN", fontsize=14, y=0.98)
    axs.ravel()

    def animate(i):
        global prev_weights
        axs[-1].clear()

        hierarchicalNetwork(integratedPos,decodedPos,net,velocities[i],N,iterations,wrap_iterations, wrap_mag, wrap_inhi)
        colors=[(0.9,0.4,0.5,0.4),(0.8,0.3,0.5,0.6),(0.8,0.1,0.3,0.8),(0.8,0,0,0.9),(0.7,0,0.1,1),'r']
        for k in range(ncols-1):
            axs[k].clear()
            axs[k].set_title(f"Resolution ({scale[k]}m per Neuron)",fontsize=10)
       
            axs[k].bar(np.arange(N),prev_weights[k][:],color=colors[k])
            axs[k].spines[['top', 'left', 'right']].set_visible(False)

           

        # cs_idx=np.argmin(abs(scale-velocities[i]))
        cs_idx=scale_selection(velocities[i],scales)
        color_list=[(0.9,0.4,0.5,0.4),(0.8,0.3,0.5,0.6),(0.8,0.1,0.3,0.8),(0.8,0,0,0.9),(0.7,0,0.1,1)]
        axs[cs_idx].axis('on')
        axs[cs_idx].tick_params(axis='both', which='both', bottom=False, top=False, left= False, labelbottom=False, labelleft=False)

        axs[-1].scatter(integratedPos[-1],0.75,color=color_list[cs_idx])
        axs[-1].scatter(decodedPos[-1],0.25,color='g')
        axs[-1].set_xbound([0,40000])
        axs[-1].set_ybound([0,1])
        axs[-1].get_yaxis().set_visible(False)
        axs[-1].spines[['top', 'left', 'right']].set_visible(False)

    ani = FuncAnimation(fig, animate, interval=1,frames=len(velocities),repeat=False)
    if visualise==True:
        f = r"./results/Hierarchical_ScaleSelection_Multiscale_Citiscape.gif" 
        writergif = animation.PillowWriter(fps=10) 
        ani.save(f, writer=writergif)
    else: 
        plt.show()

def MultiResolutionFeedthrough1D(velocities,scales, fitness=False, visualise=True):
    global prev_weights, speeds
    # num_links,excite,activity_mag,inhibit_scale=1,3,0.0721745813*5,2.96673372e-02*5
    integratedPos=[0]
    decodedPos=[0]
    speeds=[0]

    prev_weights=[np.zeros(N),np.zeros(N),np.zeros(N),np.zeros(N),np.zeros(N),np.zeros(N)]
    net=attractorNetwork(N,num_links,excite, activity_mag,inhibit_scale)
    for n in range(len(prev_weights)):
        prev_weights[n][net.activation(0)]=net.full_weights(num_links)
    

    for i in range(len(velocities)):
        hierarchicalNetwork(integratedPos,decodedPos,net,velocities[i],N,iterations,wrap_iterations, wrap_mag, wrap_inhi)
    
    if visualise==True:
        '''initlising network and animate figures'''
        # fig = plt.figure(figsize=(6, 6))
        # fig_cols=2
        # fig_rows=2
        # ax10 = plt.subplot2grid(shape=(fig_rows, fig_cols), loc=(0, 0), rowspan=1,colspan=1)
        # ax11 = plt.subplot2grid(shape=(fig_rows, fig_cols), loc=(0, 1), rowspan=1,colspan=1)

        fig, axs = plt.subplots(2,2, figsize=(8, 5))
        fig.subplots_adjust(hspace=0.95)
        fig.suptitle("CAN with varying Input Speeds", fontsize=14, y=0.98)
        axs=axs.flatten()

        axs[0].set_title('Path Integrated Position')
        axs[0].plot(integratedPos)
        axs[0].set_xlabel('Time [secs]'), axs[0].set_ylabel('Position [m]')
        axs[1].set_title('Network Decoded Position')
        axs[1].set_xlabel('Time [secs]'),
        axs[1].plot(decodedPos, c='purple')

        axs[2].set_title('Input Velocities')
        axs[2].plot(velocities), axs[2].set_ylabel('Position [m]')
        axs[3].set_title('CAN velocities')
        axs[3].plot(speeds, c='purple')
        plt.show()
    elif visualise==False: 
        return integratedPos, decodedPos, speeds
    elif fitness==True:
        return np.sum(abs(np.array(integratedPos)-np.array(decodedPos)))

scales=[0.25,1,4,16,100,10000]
N=100  
num_links,excite,activity_mag,inhibit_scale,iterations=7,10,2.13954369,0.12387683,2 # best rn
wrap_iterations,wrap_mag,wrap_inhi=iterations,activity_mag,inhibit_scale
 