

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
sys.path.append('./scripts')
from CAN import  attractorNetworkSettling, attractorNetwork, attractorNetworkScaling, attractorNetwork2D
import CAN as can
import pykitti
import json 
from DataHandling import saveOrLoadNp


def inputToAllNetwork(integratedPos,decodedPos,net,i,N):
    input=velocities[i]

    delta = [(input/scale[0]), (input/scale[1]), (input/scale[2]), (input/scale[3]), (input/scale[4])]
    for i in range(9):
        split_output=np.array([np.argmax(x) for x in prev_weights])
        cs_idx=np.argmin(abs(scale-input))
        prev_weights[cs_idx][:]= net.update_weights_dynamics(prev_weights[cs_idx][:],delta[cs_idx])
        prev_weights[cs_idx][prev_weights[cs_idx][:]<0]=0
        split_output[cs_idx]=can.activityDecoding(prev_weights[cs_idx][:],5,N)
    
    decoded_translation=np.sum((split_output)*scale)
    integratedPos.append(integratedPos[-1]+input)
    decodedPos.append(decoded_translation)   
    print(f"{str(i)}  translation {input} integrated decoded {round(integratedPos[-1],3)}  {str(decoded_translation )}  ")

def hierarchicalNetwork(integratedPos,decodedPos,net,i,N):
    if i>1:
        input=velocities[i]

        delta = [(input/scale[0]), (input/scale[1]), (input/scale[2]), (input/scale[3]), (input/scale[4])]
        split_output=np.zeros((len(delta)))
        
        '''updating network'''
        for i in range(3):
            # split_output=np.array([np.argmax(x) for x in prev_weights])
            cs_idx=np.argmin(abs(scale-input))    # closest scale index
            wraparound=np.zeros(len(delta))
            wraparound[cs_idx]=(np.argmax(prev_weights[cs_idx][:]) + delta[cs_idx])//N
            # print(np.argmax(prev_weights[cs_idx][:]), delta[cs_idx], wraparound[cs_idx])
            prev_weights[cs_idx][:]= net.update_weights_dynamics(prev_weights[cs_idx][:],delta[cs_idx])
            prev_weights[cs_idx][prev_weights[cs_idx][:]<0]=0
            split_output[0:cs_idx+1]=[np.argmax(prev_weights[x][:]) for x in range(0,cs_idx+1)]

            for n in range(cs_idx+1,len(delta)):
                net=attractorNetwork(N,num_links,excite, 1.5,inhibit_scale)
                wraparound[n]=(np.argmax(prev_weights[n][:]) + wraparound[n-1])//N
                prev_weights[n][:]= net.update_weights_dynamics(prev_weights[n][:],wraparound[n-1])
                prev_weights[n][prev_weights[n][:]<0]=0
                split_output[n]=np.argmax(prev_weights[n][:])

        
        decoded_translation=np.sum(((split_output))*scale)

        integratedPos.append(integratedPos[-1]+input)
        decodedPos.append(decoded_translation)   

        print(f"{str(i)}  translation {input} integrated decoded {round(integratedPos[-1],3)}  {str(decoded_translation )} ")
        

def testing_CAN_shift():
    #Network and initliasation 
    N,num_links,excite,activity_mag,inhibit_scale=100, 1,3,0.0721745813*5,2.96673372e-02
    prev_weights=np.zeros(N)
    net=attractorNetwork(N,num_links,excite, activity_mag,inhibit_scale)
    prev_weights[net.activation(0)]=net.full_weights(num_links)

    scale=0.5
    inputs=np.linspace(0,20,200)
    outputs=np.zeros(len(inputs))
    for i in range(len(inputs)):
        prev_weights=net.update_weights_dynamics(prev_weights,inputs[i]/scale)
        outputs[i]=np.argmax(prev_weights)

    plt.plot(inputs,outputs, inputs, inputs)
    # plt.xlim([0, 20])
    plt.show()



def GIF_MultiResolution1D(velocities,scale, visualise=False):
    global prev_weights, num_links, excite, curr_parameter
    N=100
     
    # num_links,excite,activity_mag,inhibit_scale=1,3,0.0721745813*5,2.96673372e-02

    integratedPos=[0]
    decodedPos=[0]

    prev_weights=[np.zeros(N), np.zeros(N), np.zeros(N),np.zeros(N), np.zeros(N)]
    net=attractorNetwork(N,num_links,excite, activity_mag,inhibit_scale)
    for n in range(len(prev_weights)):
        prev_weights[n][net.activation(0)]=net.full_weights(num_links)

    '''initlising network and animate figures'''
    fig = plt.figure(figsize=(6, 6))
    fig_cols=1
    fig_rows=6

    ax10 = plt.subplot2grid(shape=(fig_rows, fig_cols), loc=(0, 0), rowspan=1,colspan=1)
    ax11 = plt.subplot2grid(shape=(fig_rows, fig_cols), loc=(1, 0), rowspan=1,colspan=1)
    ax12 = plt.subplot2grid(shape=(fig_rows, fig_cols), loc=(2, 0), rowspan=1,colspan=1)
    ax13 = plt.subplot2grid(shape=(fig_rows, fig_cols), loc=(3, 0), rowspan=1,colspan=1)
    ax14 = plt.subplot2grid(shape=(fig_rows, fig_cols), loc=(4, 0), rowspan=1,colspan=1)
    ax15 = plt.subplot2grid(shape=(fig_rows, fig_cols), loc=(5, 0), rowspan=1,colspan=1)
    fig.tight_layout()

    def animate(i):
        global prev_weights, num_links, excite
        ax10.clear(),ax11.clear(),ax12.clear(),ax13.clear(),ax14.clear(), ax15.clear()
        
        inputToAllNetwork(integratedPos,decodedPos,net,i,N)
        
        ax10.set_title(f"Resolution ({scale[0]}m per Neuron)",fontsize=10)
        ax11.set_title(f"Resolution ({scale[1]}m per Neuron)",fontsize=10)
        ax12.set_title(f"Resolution ({scale[2]}m per Neuron)",fontsize=10)
        ax13.set_title(f"Resolution ({scale[3]}m per Neuron)",fontsize=10)
        ax14.set_title(f"Resolution ({scale[4]}m per Neuron)",fontsize=10)

        ax10.bar(np.arange(N),prev_weights[0][:],color=(0.9,0.4,0.5,0.4))
        ax10.axis('off')

        ax11.bar(np.arange(N),prev_weights[1][:],color=(0.8,0.3,0.5,0.6))
        ax11.axis('off')

        ax12.bar(np.arange(N),prev_weights[2][:],color=(0.8,0.1,0.3,0.8))
        ax12.axis('off')

        ax13.bar(np.arange(N),prev_weights[3][:],color=(0.8,0,0,0.9))
        ax13.axis('off')
        
        ax14.bar(np.arange(N),prev_weights[4][:],color=(0.7,0,0.1,1))
        ax14.axis('off')

        input_axes=[ax10, ax11, ax12, ax13, ax14]
        color_list=[(0.9,0.4,0.5,0.4),(0.8,0.3,0.5,0.6),(0.8,0.1,0.3,0.8),(0.8,0,0,0.9),(0.7,0,0.1,1)]
        cs_idx=np.argmin(abs(scale-velocities[i]))
        input_axes[cs_idx].axis('on')
        input_axes[cs_idx].tick_params(axis='both', which='both', bottom=False, top=False, left= False, labelbottom=False, labelleft=False)

        ax15.scatter(integratedPos[-1],i,color=color_list[cs_idx])
        ax15.set_xbound([0,300])
        ax15.get_yaxis().set_visible(False)
        ax15.spines[['top', 'left', 'right']].set_visible(False)

    ani = FuncAnimation(fig, animate, interval=10,frames=len(velocities),repeat=False)
    if visualise==True:
        f = r"./results/ScaleSelection_Multiscale_Citiscape.gif" 
        writergif = animation.PillowWriter(fps=10) 
        ani.save(f, writer=writergif)
    else: 
        plt.show()

def GIF_MultiResolutionFeedthrough1D(velocities,scale, visualise=False):
    global prev_weights, num_links, excite, curr_parameter
    N=20
     
    # num_links,excite,activity_mag,inhibit_scale=1,3,0.0721745813*5,2.96673372e-02*2

    integratedPos=[0]
    decodedPos=[0]

    prev_weights=[np.zeros(N), np.zeros(N), np.zeros(N),np.zeros(N), np.zeros(N)]
    net=attractorNetwork(N,num_links,excite, activity_mag,inhibit_scale)
    for n in range(len(prev_weights)):
        prev_weights[n][net.activation(0)]=net.full_weights(num_links)

    '''initlising network and animate figures'''
    fig = plt.figure(figsize=(6, 6))
    fig_cols=1
    fig_rows=6

    ax10 = plt.subplot2grid(shape=(fig_rows, fig_cols), loc=(0, 0), rowspan=1,colspan=1)
    ax11 = plt.subplot2grid(shape=(fig_rows, fig_cols), loc=(1, 0), rowspan=1,colspan=1)
    ax12 = plt.subplot2grid(shape=(fig_rows, fig_cols), loc=(2, 0), rowspan=1,colspan=1)
    ax13 = plt.subplot2grid(shape=(fig_rows, fig_cols), loc=(3, 0), rowspan=1,colspan=1)
    ax14 = plt.subplot2grid(shape=(fig_rows, fig_cols), loc=(4, 0), rowspan=1,colspan=1)
    ax15 = plt.subplot2grid(shape=(fig_rows, fig_cols), loc=(5, 0), rowspan=1,colspan=1)
    fig.tight_layout()

    def animate(i):
        global prev_weights, num_links, excite
        ax10.clear(),ax11.clear(),ax12.clear(),ax13.clear(),ax14.clear(), ax15.clear()

        hierarchicalNetwork(integratedPos,decodedPos,net,i,N)

        ax10.set_title(f"Resolution ({scale[0]}m per Neuron)",fontsize=10)
        ax11.set_title(f"Resolution ({scale[1]}m per Neuron)",fontsize=10)
        ax12.set_title(f"Resolution ({scale[2]}m per Neuron)",fontsize=10)
        ax13.set_title(f"Resolution ({scale[3]}m per Neuron)",fontsize=10)
        ax14.set_title(f"Resolution ({scale[4]}m per Neuron)",fontsize=10)

        ax10.bar(np.arange(N),prev_weights[0][:],color=(0.9,0.4,0.5,0.4))
        ax10.axis('off')

        ax11.bar(np.arange(N),prev_weights[1][:],color=(0.8,0.3,0.5,0.6))
        ax11.axis('off')

        ax12.bar(np.arange(N),prev_weights[2][:],color=(0.8,0.1,0.3,0.8))
        ax12.axis('off')

        ax13.bar(np.arange(N),prev_weights[3][:],color=(0.8,0,0,0.9))
        ax13.axis('off')
        
        ax14.bar(np.arange(N),prev_weights[4][:],color=(0.7,0,0.1,1))
        ax14.axis('off')

        cs_idx=np.argmin(abs(scale-velocities[i]))
        input_axes=[ax10, ax11, ax12, ax13, ax14]
        color_list=[(0.9,0.4,0.5,0.4),(0.8,0.3,0.5,0.6),(0.8,0.1,0.3,0.8),(0.8,0,0,0.9),(0.7,0,0.1,1)]
        input_axes[cs_idx].axis('on')
        input_axes[cs_idx].tick_params(axis='both', which='both', bottom=False, top=False, left= False, labelbottom=False, labelleft=False)

        ax15.scatter(integratedPos[-1],i,color=color_list[cs_idx])
        ax15.set_xbound([0,10000])
        ax15.get_yaxis().set_visible(False)
        ax15.spines[['top', 'left', 'right']].set_visible(False)

    ani = FuncAnimation(fig, animate, interval=10,frames=len(velocities),repeat=False)
    if visualise==True:
        f = r"./results/Hierarchical_ScaleSelection_Multiscale_Citiscape.gif" 
        writergif = animation.PillowWriter(fps=10) 
        ani.save(f, writer=writergif)
    else: 
        plt.show()


def MultiResolution1D(velocities,scale, visualise=False):
    global prev_weights, num_links, excite, curr_parameter
    N=100
     
    # num_links,excite,activity_mag,inhibit_scale=1,3,0.0721745813*5,2.96673372e-02

    integratedPos=[0]
    decodedPos=[0]

    prev_weights=[np.zeros(N), np.zeros(N), np.zeros(N),np.zeros(N), np.zeros(N)]
    net=attractorNetwork(N,num_links,excite, activity_mag,inhibit_scale)
    for n in range(len(prev_weights)):
        prev_weights[n][net.activation(0)]=net.full_weights(num_links)

    '''initlising network and animate figures'''
    fig = plt.figure(figsize=(6, 6))
    fig_cols=2
    fig_rows=1
    ax10 = plt.subplot2grid(shape=(fig_rows, fig_cols), loc=(0, 0), rowspan=1,colspan=1)
    ax11 = plt.subplot2grid(shape=(fig_rows, fig_cols), loc=(0, 1), rowspan=1,colspan=1)


    for i in range(len(velocities)):
        inputToAllNetwork(integratedPos,decodedPos,net,i,N)
   
    ax10.set_title('Path Integrated Position')
    ax10.plot(integratedPos)
    ax10.set_xlabel('Time [secs]'), ax10.set_ylabel('Position [m]')
    ax11.set_title('Network Decoded Position')
    ax11.set_xlabel('Time [secs]'), ax10.set_ylabel('Position [m]')
    ax11.plot(decodedPos, c='r')
    plt.show()

def MultiResolutionFeedthrough1D(velocities,scale, visualise=False):
    global prev_weights, num_links, excite, curr_parameter
    N=20
    # num_links,excite,activity_mag,inhibit_scale=1,3,0.0721745813*5,2.96673372e-02*5
    integratedPos=[0]
    decodedPos=[0]

    prev_weights=[np.zeros(N), np.zeros(N), np.zeros(N),np.zeros(N), np.zeros(N)]
    net=attractorNetwork(N,num_links,excite, activity_mag,inhibit_scale)
    for n in range(len(prev_weights)):
        prev_weights[n][net.activation(0)]=net.full_weights(num_links)
  
    '''initlising network and animate figures'''
    fig = plt.figure(figsize=(6, 6))
    fig_cols=2
    fig_rows=1
    ax10 = plt.subplot2grid(shape=(fig_rows, fig_cols), loc=(0, 0), rowspan=1,colspan=1)
    ax11 = plt.subplot2grid(shape=(fig_rows, fig_cols), loc=(0, 1), rowspan=1,colspan=1)


    for i in range(len(velocities)):
        hierarchicalNetwork(integratedPos,decodedPos,net,i,N)
    
    ax10.set_title('Path Integrated Position')
    ax10.plot(integratedPos)
    ax10.set_xlabel('Time [secs]'), ax10.set_ylabel('Position [m]')
    ax11.set_title('Network Decoded Position')
    ax11.set_xlabel('Time [secs]'), ax10.set_ylabel('Position [m]')
    ax11.plot(decodedPos, c='purple')
    plt.show()

num_links,excite,activity_mag,inhibit_scale=1,3,0.0721745813*4,2.96673372e-02 #og

# scale=[0.25,0.5,1,2,4]
# velocities=np.concatenate([np.array([scale[0]]*25), np.array([scale[1]]*25), np.array([scale[2]]*25), np.array([scale[3]]*25), np.array([scale[4]]*25), np.array([scale[3]]*25),  np.array([scale[2]]*25),  np.array([scale[1]]*25),  np.array([scale[0]]*25)])
# GIF_MultiResolution1D(velocities,scale, visualise=True)

# velocities=np.concatenate([np.array([scale[0]]*10), np.array([scale[1]]*10), np.array([scale[2]]*10), np.array([scale[3]]*10), np.array([scale[4]]*5), np.array([scale[3]]*10),  np.array([scale[2]]*10),  np.array([scale[1]]*10),  np.array([scale[0]]*10)])
velocities=saveOrLoadNp(f'./data/train_extra/citiscape_speed_{1}',None,'load')

num_links,excite,activity_mag,inhibit_scale=1,3,0.0721745813*5,2.96673372e-02 #hierarchy
scale=[0.5,1,2,4,8]
# GIF_MultiResolutionFeedthrough1D(velocities,scale, visualise=False)
# MultiResolutionFeedthrough1D(velocities,scale)


# scale=[0.1,1,10,100,1000]
# GIF_MultiResolution1D(velocities,scale, visualise=True)
# MultiResolution1D(velocities,scale)
testing_CAN_shift()