

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

def decodedPosAfterupdate(net,weights,input):
    original=can.activityDecoding(weights,4,N)
    for i in range(iterations):
        weights= net.update_weights_dynamics(weights,input)
    if can.activityDecoding(weights,4,N)<original:
        return 1
    else: 
        return 0

def scale_selection(input,scales):
    swap_val=1
    if input<=scales[0]*swap_val:
        scale_idx=0
    elif input>scales[0]*swap_val and input<=scales[1]*swap_val:
        scale_idx=1
    elif input>scales[1]*swap_val and input<=scales[2]*swap_val:
        scale_idx=2
    elif input>scales[2]*swap_val and input<=scales[3]*swap_val:
        scale_idx=3
    elif input>scales[3]*swap_val:
        scale_idx=4
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
        
def hierarchicalNetwork(integratedPos,decodedPos,net,input,N, iterations,wrap_iterations, wrap_mag, wrap_inhi):
    # wrap_mag=2.851745813
    # wrap_inhi=2.96673372e-02/2
    
    delta = [(input/scales[0]), (input/scales[1]), (input/scales[2]), (input/scales[3]), (input/scales[4])]
    split_output=np.zeros((len(scales)))
    

    net=attractorNetwork(N,num_links,excite, wrap_mag,wrap_inhi)
    cs_idx=scale_selection(input,scales)
    wraparound=np.zeros(len(scales))
    wraparound[cs_idx]=(can.activityDecoding(prev_weights[cs_idx][:],4,N) + delta[cs_idx])//(N-1)

    # print(can.activityDecoding(prev_weights[cs_idx][:],4,N),cs_idx,wraparound[cs_idx],wraparound[4])

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
        print(can.activityDecoding(prev_weights[5][:],4,N))

    '''Decode position'''
    split_output=np.array([can.activityDecoding(prev_weights[m][:],4,N) for m in range(len(scales))])
    decoded_translation=np.sum(((split_output))*scales)
    speeds.append(decoded_translation-decodedPos[-1])
    integratedPos.append(integratedPos[-1]+input)
    decodedPos.append(decoded_translation)   
    # print(f"translation {input} integrated decoded {round(integratedPos[-1],3)}  {str(decoded_translation )} ")


def GIF_MultiResolution1D(velocities,scale, visualise=False):
    global prev_weights, num_links, excite, curr_parameter
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
    fig, axs = plt.subplots(ncols,1, figsize=(10, 7))
    fig.subplots_adjust(hspace=0.9)
    fig.suptitle("Multiscale CAN", fontsize=14, y=0.98)
    axs.ravel()

    def animate(i):
        global prev_weights
        axs[-1].clear()
        print(i)

        hierarchicalNetwork(integratedPos,decodedPos,net,velocities[i],N,iterations,wrap_iterations, wrap_mag, wrap_inhi)
        colors=[(0.9,0.4,0.5,0.4),(0.8,0.3,0.5,0.6),(0.8,0.1,0.3,0.8),(0.8,0,0,0.9),(0.7,0,0.1,1),'r']
        for k in range(ncols-1):
            axs[k].clear()
            axs[k].set_title(f"Resolution ({scale[k]}m per Neuron)",fontsize=10)
       
            axs[k].bar(np.arange(N),prev_weights[k][:],color=colors[k])
            axs[k].axis('off')
            # axs[k].spines[['top', 'left', 'right']].set_visible(False)

           

        # cs_idx=np.argmin(abs(scale-velocities[i]))
        cs_idx=scale_selection(velocities[i],scales)
        color_list=[(0.9,0.4,0.5,0.4),(0.8,0.3,0.5,0.6),(0.8,0.1,0.3,0.8),(0.8,0,0,0.9),(0.7,0,0.1,1)]
        axs[cs_idx].axis('on')
        axs[cs_idx].tick_params(axis='both', which='both', bottom=False, top=False, left= False, labelbottom=False, labelleft=False)

        axs[-1].scatter(decodedPos[-1],0.75,color=color_list[cs_idx])
        axs[-1].scatter(integratedPos[-1],0.25,color='k')
        axs[-1].set_title(f'Integrated Position: {round(integratedPos[-1],3)}, Decoded Position: {round(decodedPos[-1],3)}', fontsize=11)
        axs[-1].set_xbound([0,15000])
        axs[-1].set_ybound([0,1])
        axs[-1].get_yaxis().set_visible(False)
        axs[-1].spines[['top', 'left', 'right']].set_visible(False)

    ani = FuncAnimation(fig, animate, interval=1,frames=len(velocities),repeat=False)
    if visualise==True:
        f = r"./results/GIFs/Hierarchical_ScaleSelection_Multiscale_Citiscape_allScales.gif" 
        writergif = animation.PillowWriter(fps=40) 
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


def testing_and_animate_CAN(animate=True):
    global prev_weights
    fig, axs = plt.subplots(2,1, figsize=(8, 5))
    fig.subplots_adjust(hspace=0.9)
    fig.suptitle("CAN with varying Input Speeds", fontsize=14, y=0.98)
    axs.ravel()
    
    N,num_links,excite,activity_mag,inhibit_scale=100, 1,3,0.0721745813*5,2.96673372e-02
    N,num_links,excite,activity_mag,inhibit_scale=100, 4,7,2.33652075e-01,3.15397654e-02
    N,num_links,excite,activity_mag,inhibit_scale=100,6,4,3.98210738e+00,9.16665296e-02
    N,num_links,excite,activity_mag,inhibit_scale=100,4,9,3.37612420e+00,8.57140733e-02
    N,num_links,excite,activity_mag,inhibit_scale=100,2,7,3.87533363,0.07636933
    N,num_links,excite,activity_mag,inhibit_scale==100,6,1,3.98828739e+00,8.26161317e-02
    N,num_links,excite,activity_mag,inhibit_scale,iterations=100,9,4,0.49018873,0.06870083,6
    N,num_links,excite,activity_mag,inhibit_scale,iterations==100,1,1,0.63344853,0.09274966,6
    prev_weights=np.zeros(N)
    net=attractorNetwork(N,num_links,excite, activity_mag,inhibit_scale)
    prev_weights[net.activation(0)]=net.full_weights(num_links)
    inputs=np.linspace(0,1,70)
    peaks=np.zeros(len(inputs))
    outputs=np.zeros(len(inputs))
    minimum_error_region=[]

    if animate==True:
        def animate(i):
            global prev_weights
            axs[0].clear(), axs[1].clear()
            if i>0:
                for iter in range(iterations):
                    prev_weights=net.update_weights_dynamics(prev_weights,inputs[i])
                    prev_weights[prev_weights<0]=0
                peaks[i]=can.activityDecoding(prev_weights,4,N)
                outputs[i]=(abs(can.activityDecoding(prev_weights,4,N)-peaks[i-1]))
                # print(inputs[i],outputs[i], np.argmax(prev_weights))

                axs[0].bar(np.arange(N),prev_weights,color=(0.9,0.4,0.5,0.4))
                axs[0].set_title(f"Input: {round(inputs[i],2)}, Output: {outputs[i]}")
                axs[0].set_ylim([0, 0.6])

                axs[1].plot(inputs,outputs, 'g.')
                axs[1].plot(inputs,inputs, 'k')
                axs[1].set_title(f'1 meter Shift Per Neuron')
                axs[1].set_ylabel('Decoded Speeds')
                axs[1].set_xlabel('Input Speeds ')
            
        
        ani = FuncAnimation(fig, animate, interval=10,frames=len(inputs),repeat=False)
        plt.show()
        # f = r"./results/CAN_with_varying_InputSpeeds.gif" 
        # writergif = animation.PillowWriter(fps=5) 
        # ani.save(f, writer=writergif)

    else:
        for  i in range(1,len(inputs)):
            prev_weights=net.update_weights_dynamics(prev_weights,inputs[i])
            prev_weights[prev_weights<0]=0
            peaks[i]=np.argmax(prev_weights)
            outputs[i]=(abs(np.argmax(prev_weights)-peaks[i-1]))
            if (abs(outputs[i]-inputs[i]))<0.5:
                # print(scale, outputs[i], inputs[i])
                minimum_error_region.append([inputs[i],outputs[i]])

        min_err_out=np.array([minimum_error_region[i][0] for i in range(len(minimum_error_region))])
        min_err_in=np.array([minimum_error_region[i][1] for i in range(len(minimum_error_region))])
        axs[0].plot(inputs,outputs)
        axs[0].plot(min_err_in,min_err_out, 'r.')
        axs[0].set_title(f'1 meter Shift Per Neuron')
        axs[0].set_ylabel('Decoded Speeds')
        axs[0].set_xlabel('Input Speeds ')

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
        integrate,decode,CANspeeds=MultiResolutionFeedthrough1D(speeds,scales,visualise=False)
        axs[dataset_num].plot(speeds, linewidth=2,label ='GT')
        axs[dataset_num].plot(CANspeeds,'--', linewidth=1.5,label='decoded')
        axs[dataset_num].legend()
        axs[dataset_num].set_title(city_names[dataset_num])
        # axs[dataset_num].set_ylim([0,11000])
        # axs[dataset_num].set_xlim([0,1600])
    plt.show()

# num_links,excite,activity_mag,inhibit_scale=1,3,0.0721745813*5,2.96673372e-02 #og
# scale=[0.25,0.5,1,2,4]
# velocities=np.concatenate([np.array([scale[0]]*25), np.array([scale[1]]*25), np.array([scale[2]]*25), np.array([scale[3]]*25), np.array([scale[4]]*25), np.array([scale[3]]*25),  np.array([scale[2]]*25),  np.array([scale[1]]*25),  np.array([scale[0]]*25)])
# GIF_MultiResolution1D(velocities,scale, visualise=True)

# velocities=np.concatenate([np.array([scale[0]]*10), np.array([scale[1]]*10), np.array([scale[2]]*10), np.array([scale[3]]*10), np.array([scale[4]]*5), np.array([scale[3]]*10),  np.array([scale[2]]*10),  np.array([scale[1]]*10),  np.array([scale[0]]*10)])

np.random.seed(10)
velocities=saveOrLoadNp(f'./data/train_extra/citiscape_speed_{1}',None,'load')
velocities=np.concatenate([np.array([16]*200),np.array([100]*80)])
velocities=np.concatenate([np.random.uniform(0,0.9,100), np.random.uniform(1,10,100), np.random.uniform(30,100,100)])
velocities=np.concatenate([np.random.uniform(0,0.25,220), np.random.uniform(0.25,1,220), np.random.uniform(1,4,220), np.random.uniform(4,16,220), np.random.uniform(16,100,220)])

# N,num_links,excite,activity_mag,inhibit_scale=100,1,3,0.0721745813*100,2.96673372e-02 #hierarchy
# N,num_links,excite,activity_mag,inhibit_scale=100, 4,7,2.33652075e-01,3.15397654e-02
# N,num_links,excite,activity_mag,inhibit_scale,iterations=100,9,4,0.49018873,0.06870083,6
# N,num_links,excite,activity_mag,inhibit_scale,iterations=100,1,1,0.63344853,0.09274966,6
# wrap_iterations=10
# wrap_mag=0.63344853*10
# wrap_inhi=inhibit_scale
scales=[0.25,1,4,16,100,10000]
N=100   
num_links,excite,activity_mag,inhibit_scale,iterations,wrap_iterations,wrap_mag,wrap_inhi=2,3,1.2878113,0.08947041,3,2,2.21454636,0.08230016
num_links,excite,activity_mag,inhibit_scale,iterations,wrap_iterations,wrap_mag,wrap_inhi=4,6,1.44615670e+00,5.44855219e-02,2,7,2.42691166e+00,4.45296033e-02
num_links,excite,activity_mag,inhibit_scale,iterations=1,6,2.92694865,0.09309933,2
num_links,excite,activity_mag,inhibit_scale,iterations=7,1,1.15234255,0.09776188,4
num_links,excite,activity_mag,inhibit_scale,iterations=1,6,2.29804683,0.14980466,2
num_links,excite,activity_mag,inhibit_scale,iterations=7,10,2.13954369,0.12387683,2 # best rn
# num_links,excite,activity_mag,inhibit_scale,iterations=6,9,2.17651734,0.12845625,2
# num_links,excite,activity_mag,inhibit_scale,iterations=3,1,2.70241295,0.15825576,2
wrap_iterations,wrap_mag,wrap_inhi=iterations,activity_mag,inhibit_scale


# GIF_MultiResolutionFeedthrough1D(velocities,scales, visualise=False)
MultiResolutionFeedthrough1D(velocities,scales)

# scale=[0.1,1,10,100,1000]
# GIF_MultiResolution1D(velocities,scale, visualise=True)
# MultiResolution1D(velocities,scale)
# testing_and_animate_CAN(animate=True)
# testAllcities()

#  inputs=np.concatenate([np.linspace(0,1,25), np.array([4]*5),  np.array([16]*1)])