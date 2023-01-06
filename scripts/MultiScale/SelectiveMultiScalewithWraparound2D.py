
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
from CAN import attractorNetwork2D, attractorNetwork, activityDecodingAngle, activityDecoding
import CAN as can
import pykitti
import json 
from DataHandling import saveOrLoadNp  


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

def hierarchicalNetwork2D(prev_weights, speeds_x,integratedPos_x,decodedPos_x,speeds_y,integratedPos_y,decodedPos_y,net,x_input,y_input, N, iterations,wrap_iterations):
    x_delta = [(x_input/scales[0]), (x_input/scales[1]), (x_input/scales[2]), (x_input/scales[3]), (x_input/scales[4])]
    y_delta = [(y_input/scales[0]), (y_input/scales[1]), (y_input/scales[2]), (y_input/scales[3]), (y_input/scales[4])]
    x_split_output=np.zeros((len(scales)))
    y_split_output=np.zeros((len(scales)))
    
    sx_idx=scale_selection(x_input,scales)
    sy_idx=scale_selection(y_input,scales)
    wraparoundX=np.zeros(len(scales))
    wraparoundY=np.zeros(len(scales))
    wraparoundX[sx_idx]=(np.argmax(np.max(prev_weights[sx_idx], axis=0)) + x_delta[sx_idx])//(N-1)
    wraparoundY[sy_idx]=(np.argmax(np.max(prev_weights[sy_idx], axis=1)) + y_delta[sy_idx])//(N-1)

    '''Update selected scale'''
    for iter in range(iterations):
        prev_weights[sx_idx][:][:]= net.update_weights_dynamics(prev_weights[sx_idx][:][:],0,x_delta[sx_idx])
        prev_weights[sx_idx][:][prev_weights[sx_idx][:]<0]=0

        prev_weights[sy_idx][:][:]= net.update_weights_dynamics(prev_weights[sy_idx][:][:],y_delta[sy_idx],0)
        prev_weights[sy_idx][:][prev_weights[sy_idx][:]<0]=0


    '''Update the 100 scale based on wraparound in any of the previous scales'''
    update_amountX, update_amountY = 0,0
    if (sx_idx != 4) and wraparoundX[sx_idx]!=0:
        update_amountX=(wraparoundX[sx_idx]*scales[sx_idx]*N)/scales[4]
        wraparoundX[4]=(np.argmax(np.max(prev_weights[4], axis=0)) + update_amountX)//(N-1)

    if (sy_idx != 4) and wraparoundY[sy_idx]!=0:    
        update_amountY=(wraparoundY[sy_idx]*scales[sy_idx]*N)/scales[4]
        wraparoundY[4]=(np.argmax(np.max(prev_weights[4], axis=1)) + update_amountY)//(N-1)

    for iter in range(wrap_iterations):
        prev_weights[-2][:][:]= net.update_weights_dynamics(prev_weights[-2][:][:],update_amountY, update_amountX)
        prev_weights[-2][prev_weights[-2][:][:]<0]=0


    '''Update the 10000 scale based on wraparound in the 100 scale'''
    if wraparoundX[4] !=0:
        for iter in range(wrap_iterations):
            prev_weights[-1][:][:]= net.update_weights_dynamics(prev_weights[-1][:][:],(wraparoundY[4]*scales[4]*N)/scales[-1],(wraparoundX[4]*scales[4]*N)/scales[-1])
            prev_weights[-1][prev_weights[-1][:][:]<0]=0

    
    '''Decode position'''
    x_split_output=np.array([np.argmax(np.max(prev_weights[m], axis=0)) for m in range(len(scales))])
    x_decoded_translation=np.sum(((x_split_output))*scales)

    y_split_output=np.array([np.argmax(np.max(prev_weights[m], axis=1)) for m in range(len(scales))])
    y_decoded_translation=np.sum(((y_split_output))*scales)

    #store velocities and positions 
    speeds_x.append(x_decoded_translation-decodedPos_y[-1])
    speeds_y.append(y_decoded_translation-decodedPos_y[-1])

    integratedPos_x.append(integratedPos_x[-1]+x_input)
    integratedPos_y.append(integratedPos_y[-1]+y_input )
    
    decodedPos_x.append(x_decoded_translation)  
    decodedPos_y.append( y_decoded_translation) 
    # print(f"translation {input} integrated decoded {round(integratedPos[-1],3)}  {str(decoded_translation )} ")


def GIF_MultiResolutionFeedthrough2D(x_velocities, y_velocities, scale, visualise=False):
    global prev_weights
    integratedPos_x=[0]
    decodedPos_x=[0]
    speeds_x=[0]

    integratedPos_y=[0]
    decodedPos_y=[0]
    speeds_y=[0]

    prev_weights=[np.zeros((N,N)),np.zeros((N,N)),np.zeros((N,N)),np.zeros((N,N)),np.zeros((N,N)),np.zeros((N,N))]
    net=attractorNetwork2D(N,N,num_links,excite, activity_mag,inhibit_scale)
    for n in range(len(prev_weights)):
        prev_weights[n]=net.excitations(0,0)

    '''initlising network and animate figures'''
    nrows=7
    fig, axs = plt.subplots(1,nrows, figsize=(12, 2))
    fig.subplots_adjust(hspace=0.9)
    fig.suptitle("Multiscale CAN", fontsize=14, y=0.98)
    axs.ravel()

    def animate(i):
        global prev_weights
        # axs[-1].clear()
        hierarchicalNetwork2D(prev_weights,speeds_x,integratedPos_x,decodedPos_x,speeds_y,integratedPos_y,decodedPos_y,net,x_velocities[i],y_velocities[i],N,iterations,wrap_iterations)
        colors=[(0.9,0.4,0.5,0.4),(0.8,0.3,0.5,0.6),(0.8,0.1,0.3,0.8),(0.8,0,0,0.9),(0.7,0,0.1,1),'r']
        for k in range(nrows-1):
            axs[k].clear()
            axs[k].set_title(f"Scale {scale[k]}m",fontsize=10)
       
            axs[k].imshow(prev_weights[k][:][:])#(np.arange(N),prev_weights[k][:],color=colors[k])
            axs[k].spines[['top', 'left', 'right']].set_visible(False)

           

        # cs_idx=np.argmin(abs(scale-velocities[i]))
        cs_idx=scale_selection(x_velocities[i],scales)
        color_list=[(0.9,0.4,0.5,0.4),(0.8,0.3,0.5,0.6),(0.8,0.1,0.3,0.8),(0.8,0,0,0.9),(0.7,0,0.1,1)]
        axs[cs_idx].axis('on')
        axs[cs_idx].tick_params(axis='both', which='both', bottom=False, top=False, left= False, labelbottom=False, labelleft=False)

        axs[-1].scatter(integratedPos_x[-1],integratedPos_y[-1],color=color_list[cs_idx])
        axs[-1].scatter(decodedPos_x[-1],decodedPos_y[-1],color='k')
        # axs[-1].set_xbound([0,2000])
        # axs[-1].set_ybound([0,2000])
        # axs[-1].get_yaxis().set_visible(False)
        axs[-1].spines[['top', 'right']].set_visible(False)

    ani = FuncAnimation(fig, animate, interval=1,frames=len(x_velocities),repeat=False)
    if visualise==True:
        f = r"./results/Hierarchical_ScaleSelection_Multiscale_Citiscape.gif" 
        writergif = animation.PillowWriter(fps=10) 
        ani.save(f, writer=writergif)
    else: 
        plt.show()

def MultiResolutionFeedthrough2D(x_velocities,y_velocities, scales, fitness=False, visualise=True):
    # num_links,excite,activity_mag,inhibit_scale=1,3,0.0721745813*5,2.96673372e-02*5
    integratedPos_x=[0]
    decodedPos_x=[0]
    speeds_x=[0]

    integratedPos_y=[0]
    decodedPos_y=[0]
    speeds_y=[0]


    prev_weights=[np.zeros((N,N)),np.zeros((N,N)),np.zeros((N,N)),np.zeros((N,N)),np.zeros((N,N)),np.zeros((N,N))]
    network=attractorNetwork2D(N,N,num_links,excite, activity_mag,inhibit_scale)
    for n in range(len(prev_weights)):
        prev_weights[n]=network.excitations(0,0)
    

    for i in range(len(x_velocities)):
        hierarchicalNetwork2D(prev_weights,speeds_x,integratedPos_x,decodedPos_x,speeds_y,integratedPos_y,decodedPos_y,network,x_velocities[i],y_velocities[i],N,iterations,wrap_iterations)
    
    if visualise==True:
        # integratedPos_x=[val[0] for val in integratedPos]
        # decodedPos_x=[val[0] for val in decodedPos]
        # speeds_x=[val[0] for val in speeds]

        # integratedPos_y=[val[1] for val in integratedPos]
        # decodedPos_y=[val[1] for val in decodedPos]
        # speeds_y=[val[1] for val in speeds]

        '''initlising network and animate figures'''
        fig, axs = plt.subplots(2,2, figsize=(8, 5))
        fig.subplots_adjust(hspace=0.95)
        fig.suptitle("CAN with varying Input Speeds", fontsize=14, y=0.98)
        axs=axs.flatten()

        axs[0].set_title('Path Integrated Position')
        axs[0].plot(integratedPos_x,integratedPos_y )
        axs[0].set_xlabel('Time [secs]'), axs[0].set_ylabel('Position [m]')
        axs[1].set_title('Network Decoded Position')
        axs[1].set_xlabel('Time [secs]'),
        axs[1].plot(decodedPos_x,decodedPos_y, c='purple')

        axs[2].set_title('Input Velocities')
        axs[2].plot(x_velocities, y_velocities, '.'), axs[2].set_ylabel('Position [m]')
        axs[3].set_title('CAN velocities')
        axs[3].plot(speeds_x, speeds_y, '.' , c='purple')
        plt.show()
    elif visualise==False: 
        return integratedPos, decodedPos, speeds
    elif fitness==True:
        return np.sum(abs(np.array(integratedPos)-np.array(decodedPos)))


outfile='./results/testEnvPathVelocities.npy'
vel_x,vel_y=np.load(outfile)

# velocities=np.concatenate([np.random.uniform(0,0.25,20), np.random.uniform(0.25,1,20), np.random.uniform(1,4,20), np.random.uniform(4,16,20), np.random.uniform(16,100,20)])
scales=[0.25,1,4,16,100,10000]
# num_links,excite,activity_mag,inhibit_scale,iterations,wrap_iterations=1,1,1,0.0005,1, 1
# num_links,excite,activity_mag,inhibit_scale,iterations,wrap_iterations=8,1,0.27758052,0.08663314,3,6


# GIF_MultiResolutionFeedthrough2D(vel_x,vel_y,scales)
# MultiResolutionFeedthrough2D(vel_x, vel_y,scales)

# prev_weights=[np.zeros((N,N))+2,np.zeros((N,N))+1,np.zeros((N,N)),np.zeros((N,N)),np.zeros((N,N)),np.zeros((N,N))]
# plt.imshow(prev_weights[0][:][:])
# plt.show()
theta_weights=np.zeros(360)
theata_called_iters=0

def headDirection(theta_weights, angVel):
    global theata_called_iters
    N=360
    num_links,excite,activity_mag,inhibit_scale, iterations=16, 17, 2.16818183,  0.0281834545, 2
    net=attractorNetwork(N,num_links,excite, activity_mag,inhibit_scale)
    
    if theata_called_iters==0:
        theta_weights[net.activation(0)]=net.full_weights(num_links)
        theata_called_iters+=1


    for j in range(iterations):
        theta_weights=net.update_weights_dynamics(theta_weights,angVel)
        theta_weights[theta_weights<0]=0
    
    print(np.argmax(theta_weights))
    
    return theta_weights


def attractorGridcell():
    global prev_weights,x, y
    N=100
    num_links,excite,activity_mag,inhibit_scale=1,1,1,0.0005
    prev_weights=np.zeros((N,N))
    network=attractorNetwork2D(N,N,num_links,excite, activity_mag,inhibit_scale)
    prev_weights=network.excitations(50,50)
    x,y=50,50
    dirs=np.arange(0,90)
    speeds=np.linspace(0.1,1.1, 90)

    fig, axs = plt.subplots(1,1,figsize=(5, 5))
    def animate(i):
        axs.clear()
        global prev_weights, x, y
        
        prev_weights=network.update_weights_dynamics(prev_weights, dirs[i], speeds[i])

        print( np.argmax(np.max(prev_weights, axis=1)), np.argmax(np.max(prev_weights, axis=0)))
        x,y=x+speeds[i]*np.sin(np.deg2rad(dirs[i])), y+speeds[i]*np.cos(np.deg2rad(dirs[i]))
        print(round(x),round(y))
        print(' ')
        axs.imshow(prev_weights)
        axs.invert_yaxis()
    
    ani = FuncAnimation(fig, animate, interval=1,frames=len(speeds),repeat=False)
    plt.show()

def attractorGridcell_fitness():

    N=100
    num_links,excite,activity_mag,inhibit_scale=1,1,1,0.0005
    prev_weights=np.zeros((N,N))
    network=attractorNetwork2D(N,N,num_links,excite, activity_mag,inhibit_scale)
    prev_weights=network.excitations(50,50)
    x,y=50,50
    dirs=np.arange(0,90)
    speeds=np.linspace(0.1,1.1,90)


    for i in range(len(speeds)):

        prev_weights=network.update_weights_dynamics(prev_weights, dirs[i], speeds[i])

        print( np.argmax(np.max(prev_weights, axis=1)), np.argmax(np.max(prev_weights, axis=0)))
        x,y=x+speeds[i]*np.sin(np.deg2rad(dirs[i])), y+speeds[i]*np.cos(np.deg2rad(dirs[i]))
        print(round(x),round(y))
        print(' ')

    

    
# attractorGridcell()
attractorGridcell_fitness()
# for i in range(1,360):
#     theta_weights = headDirection(theta_weights, 1)
