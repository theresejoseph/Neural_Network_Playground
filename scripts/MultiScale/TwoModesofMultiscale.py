

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



def scale_selection(input,scales):
    swap_val=4
    if input<=scales[0]*swap_val:
        scale_idx=0
    elif input>scales[0]*swap_val and input<=scales[1]*swap_val:
        scale_idx=1
    elif input>scales[1]*swap_val and input<=scales[2]*swap_val:
        scale_idx=2
    elif input>scales[2]*swap_val:
        scale_idx=3
    return scale_idx

def inputToAllNetwork(integratedPos,decodedPos,net,i,N):
    input=velocities[i]

    delta = [(input/scales[0]), (input/scales[1]), (input/scales[2]), (input/scales[3]), (input/scales[4])]
    for i in range(9):
        split_output=np.array([np.argmax(x) for x in prev_weights])
        cs_idx=np.argmin(abs(scales-input))
        prev_weights[cs_idx][:]= net.update_weights_dynamics(prev_weights[cs_idx][:],delta[cs_idx])
        prev_weights[cs_idx][prev_weights[cs_idx][:]<0]=0
        split_output[cs_idx]=can.activityDecoding(prev_weights[cs_idx][:],5,N)
    
    decoded_translation=np.sum((split_output)*scales)
    integratedPos.append(integratedPos[-1]+input)
    decodedPos.append(decoded_translation)   
    print(f"{str(i)}  translation {input} integrated decoded {round(integratedPos[-1],3)}  {str(decoded_translation )}  ")

def hierarchicalIncrementalNetwork(integratedPos,decodedPos,net,i,N):
    if i>1:
        input=velocities[i]

        delta = [(input/scales[0]), (input/scales[1]), (input/scales[2]), (input/scales[3]), (input/scales[4])]
        split_output=np.zeros((len(delta)))
        
        '''updating network'''
        for i in range(5):
            cs_idx=np.argmin(abs(scales-input))    # closest scale index
            # cs_idx=scale_selection(input,scales)
            wraparound=np.zeros(len(delta))
            wraparound[cs_idx]=(np.argmax(prev_weights[cs_idx][:]) + delta[cs_idx])//N
            # print(np.argmax(prev_weights[cs_idx][:]), delta[cs_idx], wraparound[cs_idx])
            prev_weights[cs_idx][:]= net.update_weights_dynamics(prev_weights[cs_idx][:],delta[cs_idx])
            prev_weights[cs_idx][prev_weights[cs_idx][:]<0]=0
            split_output[0:cs_idx+1]=[np.argmax(prev_weights[x][:]) for x in range(0,cs_idx+1)]

            for n in range(cs_idx+1,len(delta)):
                net=attractorNetwork(N,num_links,excite, activity_mag,inhibit_scale)
                wraparound[n]=(np.argmax(prev_weights[n][:]) + wraparound[n-1])//N
                prev_weights[n][:]= net.update_weights_dynamics(prev_weights[n][:],wraparound[n-1])
                prev_weights[n][prev_weights[n][:]<0]=0
                split_output[n]=np.argmax(prev_weights[n][:])

        
        decoded_translation=np.sum(((split_output))*scales)

        integratedPos.append(integratedPos[-1]+input)
        decodedPos.append(decoded_translation)   

        print(f"{str(i)}  translation {input} integrated decoded {round(integratedPos[-1],3)}  {str(decoded_translation )} ")
        
def hierarchicalNetwork(integratedPos,decodedPos,net,input,N):
    wrap_mag=2.851745813
    wrap_inhi=2.96673372e-02/2


    delta = [(input/scales[0]), (input/scales[1]), (input/scales[2]), (input/scales[3]), (input/scales[4])]
    split_output=np.zeros((len(delta)))
    
    '''updating network'''
    
    # cs_idx=np.argmin(abs(scales-input))    # closest scale index
    cs_idx=scale_selection(input,scales)
    wraparound=np.zeros(len(delta))
    wraparound[cs_idx]=(np.argmax(prev_weights[cs_idx][:]) + delta[cs_idx])//N
    # print(np.argmax(prev_weights[cs_idx][:]), delta[cs_idx], wraparound[cs_idx])
    prev_weights[cs_idx][:]= net.update_weights_dynamics(prev_weights[cs_idx][:],delta[cs_idx])
    prev_weights[cs_idx][prev_weights[cs_idx][:]<0]=0
    # split_output[0:cs_idx+1]=[np.argmax(prev_weights[x][:]) for x in range(0,cs_idx+1)]

    for n in range(cs_idx+1,len(delta)):
        wraparound[n]=(np.argmax(prev_weights[n][:]) + wraparound[n-1])//N
    for k in range(2):
        for n in range(cs_idx,len(delta)-1):
            net=attractorNetwork(N,num_links,excite, wrap_mag,wrap_inhi)
            prev_weights[-1][:]= net.update_weights_dynamics(prev_weights[-1][:],(wraparound[n]*scales[n]*N)/scales[-1])
            # print(wraparound[n], (wraparound[n]*scales[n]*N)/scales[-1])
            prev_weights[-1][prev_weights[-1][:]<0]=0
    
    split_output=np.array([np.argmax(prev_weights[n][:]) for n in range(len(delta))])

    
    decoded_translation=np.sum(((split_output))*scales)

    integratedPos.append(integratedPos[-1]+input)
    decodedPos.append(decoded_translation)   

    # print(f"translation {input} integrated decoded {round(integratedPos[-1],3)}  {str(decoded_translation )} ")


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



def GIF_MultiResolutionFeedthrough1D(velocities,scale, visualise=False):
    global prev_weights, num_links, excite, curr_parameter
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

        hierarchicalNetwork(integratedPos,decodedPos,net,velocities[i],N)

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

        # cs_idx=np.argmin(abs(scale-velocities[i]))
        cs_idx=scale_selection(velocities[i],scales)
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

def MultiResolutionFeedthrough1D(velocities,scales, fitness=False, visualise=True):
    global prev_weights, num_links, excite, curr_parameter
    # num_links,excite,activity_mag,inhibit_scale=1,3,0.0721745813*5,2.96673372e-02*5
    integratedPos=[0]
    decodedPos=[0]

    prev_weights=[np.zeros(N), np.zeros(N), np.zeros(N),np.zeros(N), np.zeros(N)]
    net=attractorNetwork(N,num_links,excite, activity_mag,inhibit_scale)
    for n in range(len(prev_weights)):
        prev_weights[n][net.activation(0)]=net.full_weights(num_links)
    

    for i in range(len(velocities)):
        hierarchicalNetwork(integratedPos,decodedPos,net,velocities[i],N)
    
    if visualise==True:
        '''initlising network and animate figures'''
        fig = plt.figure(figsize=(6, 6))
        fig_cols=2
        fig_rows=1
        ax10 = plt.subplot2grid(shape=(fig_rows, fig_cols), loc=(0, 0), rowspan=1,colspan=1)
        ax11 = plt.subplot2grid(shape=(fig_rows, fig_cols), loc=(0, 1), rowspan=1,colspan=1)

        ax10.set_title('Path Integrated Position')
        ax10.plot(integratedPos)
        ax10.set_xlabel('Time [secs]'), ax10.set_ylabel('Position [m]')
        ax11.set_title('Network Decoded Position')
        ax11.set_xlabel('Time [secs]'), ax10.set_ylabel('Position [m]')
        ax11.plot(decodedPos, c='purple')
        plt.show()
    elif visualise==False: 
        return integratedPos, decodedPos
    elif fitness==True:
        return np.sum(abs(np.array(integratedPos)-np.array(decodedPos)))


def animate_CAN():
    global prev_weights
    fig, axs = plt.subplots(1,1, figsize=(8, 5))
    fig.subplots_adjust(hspace=0.9)
    fig.suptitle("CAN with varying Input Speeds", fontsize=14, y=0.98)
    
    N,num_links,excite,activity_mag,inhibit_scale=100, 1,3,0.0721745813*5,2.96673372e-02
    prev_weights=np.zeros(N)
    net=attractorNetwork(N,num_links,excite, activity_mag,inhibit_scale)
    prev_weights[net.activation(0)]=net.full_weights(num_links)
    scales=[0.25,0.5,1,2,4]
    inputs=np.linspace(0,scales[2]*20,200)/scales[2]
    outputs=np.zeros(len(inputs))

    def animate(i):
        global prev_weights
        axs.clear()
        prev_weights=net.update_weights_dynamics(prev_weights,inputs[i])
        prev_weights[prev_weights<0]=0

        outputs[i+1]=(abs(np.argmax(prev_weights)-outputs[i]))
        # print(outputs[i+1], inputs[i])

        axs.bar(np.arange(N),prev_weights,color=(0.9,0.4,0.5,0.4))
        axs.set_title(f"Input: {round(inputs[i],2)}, Output: {outputs[i+1]}")
        axs.set_ylim([0, 0.6])
        
    
    ani = FuncAnimation(fig, animate, interval=10,frames=len(inputs)-1,repeat=False)
    # plt.show()

    f = r"./results/CAN_with_varying_InputSpeeds.gif" 
    writergif = animation.PillowWriter(fps=5) 
    ani.save(f, writer=writergif)
   
    


    

def testing_CAN_shift():
    #Network and initliasation 
    N,num_links,excite,activity_mag,inhibit_scale=100, 1,3,0.0721745813*5,2.96673372e-02
    prev_weights=np.zeros(N)
    net=attractorNetwork(N,num_links,excite, activity_mag,inhibit_scale)
    prev_weights[net.activation(0)]=net.full_weights(num_links)
    scales=[0.25,0.5,1,2,4]

    fig, axs = plt.subplots(3,2, figsize=(15, 6))
    fig.subplots_adjust(hspace=0.9)
    fig.suptitle("Input vs Decoding CAN Activity Shift", fontsize=14, y=0.95)
    axs = axs.ravel()
    axs[-1].axis('off')

    for j,scale in enumerate(scales):
        inputs=np.linspace(0,scales[0]*20,200)/scales[0]
        outputs=np.zeros(len(inputs))
        minimum_error_region=[]
        for i in range(1,len(inputs)):
            prev_weights=net.update_weights_dynamics(prev_weights,inputs[i])
            outputs[i]=(abs(np.argmax(prev_weights)-outputs[i-1]))
            # print(np.argmax(prev_weights))
            # print(outputs[i])
            if (abs(outputs[i]-inputs[i]))<0.5:
                # print(scale, outputs[i], inputs[i])
                minimum_error_region.append([inputs[i],outputs[i]])
        
        min_err_out=np.array([minimum_error_region[i][0] for i in range(len(minimum_error_region))])
        min_err_in=np.array([minimum_error_region[i][1] for i in range(len(minimum_error_region))])
        axs[j].plot(inputs*scale,outputs*scale)
        axs[j].plot(min_err_in*scale,min_err_out*scale, 'r.')
        axs[j].set_title(f'{scale}meter Shift Per Neuron')
        axs[j].set_ylabel('Decoded Speeds')
        axs[j].set_xlabel('Input Speeds ')
        
        prev_weights=np.zeros(N)
        prev_weights[net.activation(0)]=net.full_weights(num_links)
        
    plt.show()

def testAllcities():
    city_names=saveOrLoadNp(f'./data/train_extra/city_names',None,'load')
    fig, axs = plt.subplots(5,5)
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.92, top=0.9, hspace=0.5, wspace=0.4)
    fig.suptitle("Integrated vs Decoded Position for all Cities", fontsize=18, y=0.95)
    axs=axs.flatten()
    axs[-1].axis('off'), axs[-2].axis('off')
    for dataset_num in range(23):
        print(dataset_num)
        speeds=saveOrLoadNp(f'./data/train_extra/citiscape_speed_{dataset_num}',None,'load')
        integrate,decode=MultiResolutionFeedthrough1D(speeds,scales,visualise=False)
        axs[dataset_num].plot(integrate)
        axs[dataset_num].plot(decode)
        axs[dataset_num].set_title(city_names[dataset_num])
    plt.show()


# num_links,excite,activity_mag,inhibit_scale=1,3,0.0721745813*5,2.96673372e-02 #og
# scale=[0.25,0.5,1,2,4]
# velocities=np.concatenate([np.array([scale[0]]*25), np.array([scale[1]]*25), np.array([scale[2]]*25), np.array([scale[3]]*25), np.array([scale[4]]*25), np.array([scale[3]]*25),  np.array([scale[2]]*25),  np.array([scale[1]]*25),  np.array([scale[0]]*25)])
# GIF_MultiResolution1D(velocities,scale, visualise=True)

# velocities=np.concatenate([np.array([scale[0]]*10), np.array([scale[1]]*10), np.array([scale[2]]*10), np.array([scale[3]]*10), np.array([scale[4]]*5), np.array([scale[3]]*10),  np.array([scale[2]]*10),  np.array([scale[1]]*10),  np.array([scale[0]]*10)])

velocities=saveOrLoadNp(f'./data/train_extra/citiscape_speed_{0}',None,'load')
N,num_links,excite,activity_mag,inhibit_scale=100,1,3,0.0721745813*100,2.96673372e-02 #hierarchy
scales=[0.5,1,2,4,100]
# GIF_MultiResolutionFeedthrough1D(velocities,scales, visualise=False)
MultiResolutionFeedthrough1D(velocities,scales)

# scale=[0.1,1,10,100,1000]
# GIF_MultiResolution1D(velocities,scale, visualise=True)
# MultiResolution1D(velocities,scale)
# animate_CAN()
# testing_CAN_shift()
# testAllcities()