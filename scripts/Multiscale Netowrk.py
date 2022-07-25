import matplotlib.pyplot as plt
import numpy as np
import math
import os
import pandas as pd
from matplotlib.animation import FuncAnimation
from matplotlib.artist import Artist
from mpl_toolkits.mplot3d import Axes3D 
from scipy import signal

from CAN import activityDecoding, activityDecodingAngle, attractorNetworkSettling, attractorNetwork, multiResolution,attractorNetworkScaling

'''Parameters'''
N=[30, 30, 30, 30, 30] #number of neurons
curr_Neuron=[0,0]
prev_weights=[np.zeros(N[0]), np.zeros(N[1])]
split_output=[0,0,0]
num_links=[3,17]
excite=[3,7]
activity_mag=[1,1]
inhibit_scale=[0.08,0.005]
curr_parameter=[0,0]
crossovers=np.zeros(5)

'''Initiliase Translating Weights'''
prev_weights_trans=[np.zeros(N[0]), np.zeros(N[1]), np.zeros(N[2]),np.zeros(N[3]), np.zeros(N[4])]
for n in range(len(prev_weights_trans)):
    net=attractorNetworkScaling(N[n],num_links[0],excite[0], activity_mag[0],inhibit_scale[0])
    prev_weights_trans[n][net.activation(0)]=net.full_weights(num_links[0])

'''Initiliase Rotating Weights'''
prev_weights_rot=[np.zeros(N[1]), np.zeros(N[1]), np.zeros(N[1]),np.zeros(N[1]), np.zeros(N[1])]
net=attractorNetworkScaling(N[1],num_links[0],excite[0], activity_mag[0],inhibit_scale[0])
for n in range(len(prev_weights_rot)):
    prev_weights_rot[n][net.activation(0)]=net.full_weights(num_links[0])

def visualiseMultiResolutionTranslation(data_x,data_y):
    # global prev_weights, num_links, excite, activity_mag,inhibit_scale, curr_parameter
    fig = plt.figure(figsize=(13, 6))
    ax0 =  plt.subplot2grid(shape=(6, 2), loc=(0, 0), rowspan=5,colspan=1)
    ax10 = plt.subplot2grid(shape=(6, 2), loc=(0, 1), rowspan=1,colspan=1)
    ax11 = plt.subplot2grid(shape=(6, 2), loc=(1, 1), rowspan=1,colspan=1)
    ax12 = plt.subplot2grid(shape=(6, 2), loc=(2, 1), rowspan=1,colspan=1)
    ax13 = plt.subplot2grid(shape=(6, 2), loc=(3, 1), rowspan=1,colspan=1)
    ax14 = plt.subplot2grid(shape=(6, 2), loc=(4, 1), rowspan=1,colspan=1)
    axtxt = plt.subplot2grid(shape=(6, 2), loc=(5, 0), rowspan=1,colspan=1)
    axtxt1 = plt.subplot2grid(shape=(6, 2), loc=(5, 1), rowspan=1,colspan=1)

    fig.tight_layout()

    def multiResolutionUpdate(input,prev_weights):
        '''Uptate the activity (previous weights) of scaled networks based on input velocity'''
        #initiliaise network and scale
        scale = [(1/(N[0]**2)), (1/N[0]), 1, N[0], N[0]**2]
        split_output=np.zeros((len(scale)))
        net=attractorNetwork(N[0],num_links[0],excite[0], activity_mag[0],inhibit_scale[0])

        #updating network
        prev_weights[0][:],crossovers[0]= net.update_weights_dynamics(prev_weights[0][:],(input/scale[0]),cross=True)    
        prev_weights[0][prev_weights[0][:]<0]=0
        split_output[0]=np.argmax(prev_weights[0][:])

        for n in range(1,len(scale)):
            net=attractorNetwork(N[0],num_links[0],excite[0], activity_mag[0],inhibit_scale[0])
            prev_weights[n][:],crossovers[n]= net.update_weights_dynamics(prev_weights[n][:],crossovers[n-1],cross=True)
            prev_weights[n][prev_weights[n][:]<0]=0
            split_output[n]=np.argmax(prev_weights[n][:])

        #decoding network and plotting values
        decoded=np.sum(split_output*scale)*np.sign(input)
        axtxt.axis('off'),axtxt1.axis('off')
        axtxt.text(0,0,"Input Velocity: " + str(input), c='r')
        axtxt1.text(0,0,"Decoded Position of Each Network: " + str(split_output), c='r')

        return decoded    

    def animate(i):
        global prev_weights_trans,prev_weights_rot, num_links, excite, activity_mag,inhibit_scale
        ax10.clear(),ax11.clear(),ax12.clear(),ax13.clear(),ax14.clear(), axtxt.clear(),axtxt1.clear()

        if i>=2:
            '''encoding mangnitude and direction of movement'''
            x0=data_x[i-2]
            x1=data_x[i-1]
            x2=data_x[i]
            y0=data_y[i-2]
            y1=data_y[i-1]
            y2=data_y[i]
            
            translation=np.sqrt(((x2-x1)**2)+((y2-y1)**2))#translation
            rotation=((np.rad2deg(math.atan2(y2-y1,x2-x1)) - np.rad2deg(math.atan2(y1-y0,x1-x0))))#%360     #angle

            decoded_translation=multiResolutionUpdate(translation,prev_weights_trans)
    
            curr_parameter[0]=curr_parameter[0]+decoded_translation
              
            # print(f"{str(i)}  velocity {str(x2)}  {str(decoded_translation )}  ")
            # print(f"{str(i)}  position {str(round(x2))}  decoded {str(round(decoded_translation ))}  ")
           
            '''decoding mangnitude and direction of movement'''
            
            ax0.set_title("Ground Truth Translation")
            ax0.plot(x2,0,"k.")
            ax0.plot(decoded_translation,1,'g.')

            ax10.set_title(str(1/(N[0]**2))+" Scale",fontsize=6)
            ax10.bar(np.arange(N[0]),prev_weights_trans[0][:],color='aqua')
            ax10.axis('off')

            ax11.set_title(str(1/N[0])+" Scale",fontsize=6)
            ax11.bar(np.arange(N[1]),prev_weights_trans[1][:],color='green')
            ax11.axis('off')

            ax12.set_title(str(1) + " Scale",fontsize=6)
            ax12.bar(np.arange(N[2]),prev_weights_trans[2][:],color='blue')
            ax12.axis('off')

            ax13.set_title(str(N[0]) + " Scale",fontsize=6)
            ax13.bar(np.arange(N[3]),prev_weights_trans[3][:],color='purple')
            ax13.axis('off')

            ax14.set_title(str(N[0]**2)+" Scale",fontsize=6)
            ax14.bar(np.arange(N[4]),prev_weights_trans[4][:],color='pink')
            ax14.axis('off')    

    ani = FuncAnimation(fig, animate, interval=1,frames=len(data_x),repeat=False)
    plt.show()



'''Translation Only'''
data_y=np.zeros(1000)
# data_x=np.concatenate([ np.linspace(0,10,100), np.linspace(10,110,100),np.linspace(110,1100,100), np.linspace(1000,10000,100)])
data_x=np.linspace(0,9990,1000)

visualiseMultiResolutionTranslation(data_x,data_y)
