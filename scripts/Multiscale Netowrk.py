import matplotlib.pyplot as plt
import numpy as np
import math
import os
import pandas as pd
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D 
from scipy import signal

from CAN import activityDecoding, activityDecodingAngle, attractorNetworkSettling, attractorNetwork, multiResolution,attractorNetworkScaling

'''Parameters'''
N=[60,360] #number of neurons
curr_Neuron=[0,0]
prev_weights=[np.zeros(N[0]), np.zeros(N[1])]
prev_weights_trans=[np.zeros(N[0]), np.zeros(N[0]), np.zeros(N[0]),np.zeros(N[0]), np.zeros(N[0])]
prev_weights_rot=[np.zeros(N[1]), np.zeros(N[1]), np.zeros(N[1]),np.zeros(N[1]), np.zeros(N[1])]
split_output=[0,0,0]
num_links=[5,17]
excite=[5,7]
activity_mag=[1,1]
inhibit_scale=[0.05,0.005]
curr_parameter=[0,0]

def multiResolutionModulus(input,split_output):
    rounded=np.round(input,2)*100

    hundreds=(rounded)%10
    tens=(rounded//10)%10
    ones=(rounded//100)

    scale=[1,0.1,0.01]
    return [ones,tens,hundreds], scale 

def multiResolutionUpdate(input,prev_weights): 
    # delta, scale = multiResolution(abs(input))
    
    scale = [(1/3600), (1/60), 1, 60, 3600]
    # scale = [0.0001, 0.01, 1, 100, 10000]
    delta = [(input/scale[0]), (input/scale[1]), (input/scale[2]),(input/scale[3]), (input/scale[4])]
    split_output=np.zeros((len(delta)))
    crossovers=np.zeros((len(delta)))
    
    net=attractorNetwork(N[0],num_links[0],excite[0], activity_mag[0],inhibit_scale[0])
    '''updating network'''
    prev_weights[0][:],crossovers[0]= net.update_weights_dynamics(prev_weights[0][:],delta[0])    
    # prev_weights[1][:],crossovers[1]= net.update_weights_dynamics(prev_weights[1][:],crossovers[0])
    # prev_weights[2][:],crossovers[2]= net.update_weights_dynamics(prev_weights[2][:],crossovers[1])
    for n in range(1,len(delta)):
        prev_weights[n][:],crossovers[n]= net.update_weights_dynamics(prev_weights[n][:],crossovers[n-1])
        prev_weights[n][prev_weights[n][:]<0]=0
        split_output[n]=np.argmax(prev_weights[n][:])

    '''decoding mangnitude and direction of movement'''
    print(crossovers)

    # print((np.argmax(prev_weights[1][:])+crossovers[0])//60)
    # print((np.argmax(prev_weights[2][:])+crossovers[1])//60)
    decoded=np.sum(split_output*scale)*np.sign(input)
    
    return decoded    

def visualiseMultiResolutionTranslation(data_x,data_y):
    global prev_weights, num_links, excite, activity_mag,inhibit_scale, curr_parameter
    # global curr_x,curr_y
    fig = plt.figure(figsize=(13, 4))
    ax0 =  plt.subplot2grid(shape=(5, 3), loc=(0, 0), rowspan=5,colspan=1)
    ax10 = plt.subplot2grid(shape=(5, 3), loc=(0, 1), rowspan=1,colspan=1)
    ax11 = plt.subplot2grid(shape=(5, 3), loc=(1, 1), rowspan=1,colspan=1)
    ax12 = plt.subplot2grid(shape=(5, 3), loc=(2, 1), rowspan=1,colspan=1)
    ax13 = plt.subplot2grid(shape=(5, 3), loc=(3, 1), rowspan=1,colspan=1)
    ax14 = plt.subplot2grid(shape=(5, 3), loc=(4, 1), rowspan=1,colspan=1)
    ax2 = plt.subplot2grid(shape=(5, 3), loc=(0, 2), rowspan=5,colspan=1)
    fig.tight_layout()

    
    # net=attractorNetworkSettling(N[1],num_links[1],excite[1], activity_mag[1],inhibit_scale[1])
    # prev_weights[1][net.activation(0)]=net.full_weights(num_links[1])
    net=attractorNetworkScaling(N[0],num_links[0],excite[0], activity_mag[0],inhibit_scale[0])
    for n in range(len(prev_weights_trans)):
        prev_weights_trans[n][net.activation(0)]=net.full_weights(num_links[0])


    def animate(i):
        global prev_weights_trans,prev_weights_rot, num_links, excite, activity_mag,inhibit_scale
        ax10.clear(),ax11.clear(),ax12.clear(),ax13.clear(),ax14.clear()
        
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

            net0=attractorNetworkScaling(N[0],num_links[0],excite[0], activity_mag[0],inhibit_scale[0])
            decoded_translation=multiResolutionUpdate(translation,prev_weights_trans)
    
            curr_parameter[0]=curr_parameter[0]+decoded_translation
              
            # print(f"{str(i)}  velocity {str(x2)}  {str(decoded_translation )}  ")
            '''decoding mangnitude and direction of movement'''
            
            ax0.set_title("Ground Truth Vel")
            ax0.plot(x2,0,"k.")
            # ax0.set_xlim([0,len(data_x)])
            # ax0.axis('equal')


            ax2.set_title("Decoded Translation Vel")
            ax2.plot(decoded_translation,0,'g.')
            # ax2.set_xlim([0,len(data_x)])
            # ax2.axis('equal')


            ax10.set_title("0.25 Scale",fontsize=6)
            ax10.bar(np.arange(N[0]),prev_weights_trans[0][:],color='aqua')
            ax10.axis('off')

            ax11.set_title("0.5 Scale",fontsize=6)
            ax11.bar(np.arange(N[0]),prev_weights_trans[1][:],color='green')
            ax11.axis('off')

            ax12.set_title("1 Scale",fontsize=6)
            ax12.bar(np.arange(N[0]),prev_weights_trans[2][:],color='blue')
            ax12.axis('off')

            ax13.set_title("2 Scale",fontsize=6)
            ax13.bar(np.arange(N[0]),prev_weights_trans[3][:],color='purple')
            ax13.axis('off')

            ax14.set_title("4 Scale",fontsize=6)
            ax14.bar(np.arange(N[0]),prev_weights_trans[4][:],color='pink')
            ax14.axis('off')

    ani = FuncAnimation(fig, animate, interval=1,frames=len(data_x),repeat=False)
    plt.show()

def visualiseMultiResolution(data_x,data_y):
    global prev_weights, num_links, excite, activity_mag,inhibit_scale, curr_parameter
    # global curr_x,curr_y
    fig = plt.figure(figsize=(13, 4))
    ax0 =  plt.subplot2grid(shape=(5, 3), loc=(0, 0), rowspan=5,colspan=1)
    ax3 = plt.subplot2grid(shape=(5, 3), loc=(0, 1), rowspan=5,colspan=1)
    ax2 = plt.subplot2grid(shape=(5, 3), loc=(0, 2), rowspan=5,colspan=1)
    fig.tight_layout()

    curr_x,curr_y=np.zeros((len(data_x))), np.zeros((len(data_y)))
    theta=np.zeros((len(data_x)))
    theta[0]=0
    theta[1]=90
    
    net=attractorNetwork(N[1],num_links[1],excite[1], activity_mag[1],inhibit_scale[1])
    prev_weights[1][net.activation(theta[1])]=net.full_weights(num_links[1])
    
    net=attractorNetwork(N[1],num_links[1],excite[1], activity_mag[1],inhibit_scale[1])
    for n in range(len(prev_weights_rot)):
        prev_weights_rot[n][net.activation(theta[1])]=net.full_weights(num_links[1])

    def animate(i):
        global prev_weights_trans,prev_weights_rot, num_links, excite, activity_mag,inhibit_scale
        # ax10.clear(),ax11.clear(),ax12.clear(),ax13.clear(),ax14.clear()
        
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

            net0=attractorNetwork(N[0],num_links[0],excite[0], activity_mag[0],inhibit_scale[0])
            decoded_translation=multiResolutionUpdate(translation,prev_weights_trans,net0)
    
            net1=attractorNetwork(N[1],num_links[1],excite[1], activity_mag[1],inhibit_scale[1])
            prev_weights[1][:]= net1.update_weights_dynamics(prev_weights[1][:],rotation)
            prev_weights[1][prev_weights[1][:]<0]=0
            decoded_rotation=activityDecodingAngle(prev_weights[1][:],num_links[1],N[1])#-prev_trans
            # decoded_rotation=np.argmax(prev_weights[1][:])
            net2=attractorNetworkSettling(N[1],num_links[1],excite[1], activity_mag[1],inhibit_scale[1])
            decoded_angVel=multiResolutionUpdate(rotation,prev_weights_rot,net2)

            "Adding translation and rotation to previous posisiton"
            curr_x[i]=curr_x[i-1]+ (decoded_translation*np.cos(np.deg2rad(decoded_rotation)))
            curr_y[i]=curr_y[i-1]+ (decoded_translation*np.sin(np.deg2rad(decoded_rotation)))
            
            '''Path integration with angular velocity'''
            theta[i]=(theta[i-1]+(decoded_angVel))%360#(2*np.pi)
            curr_parameter[0]=curr_parameter[0]+translation*np.cos(np.deg2rad(theta[i-1]))
            curr_parameter[1]=curr_parameter[1]+translation*np.sin(np.deg2rad(theta[i-1]))
            
            print(f"{str(i)}   {str(translation)}   {str(decoded_translation )}  ----- {rotation}   {decoded_angVel}")    
            
            ax0.plot(x2,y2,"k.")
            ax0.set_title("Ground Truth Position")
            ax0.axis('equal')
            # ax0.axis('equal')  

            ax3.set_title("Attractor Velocity Network")
            ax3.plot(curr_parameter[0],curr_parameter[1],'r.')
            ax3.axis('equal') 
            # 
            ax2.set_title("Attractor Rotation Network")
            ax2.plot(curr_x[i],curr_y[i],'g.')
            ax2.axis('equal')

            
    ani = FuncAnimation(fig, animate, interval=1,frames=len(data_x),repeat=False)
    plt.show()

def visualiseMultiResolutionModulus(data_x,data_y):
    global prev_weights, num_links, excite, activity_mag,inhibit_scale, curr_parameter
    # global curr_x,curr_y
    fig = plt.figure(figsize=(13, 4))
    ax0 =  plt.subplot2grid(shape=(3, 3), loc=(0, 0), rowspan=5,colspan=1)
    ax10 = plt.subplot2grid(shape=(3, 3), loc=(0, 1), rowspan=1,colspan=1)
    ax11 = plt.subplot2grid(shape=(3, 3), loc=(1, 1), rowspan=1,colspan=1)
    ax12 = plt.subplot2grid(shape=(3, 3), loc=(2, 1), rowspan=1,colspan=1)
    ax2 = plt.subplot2grid(shape=(3, 3), loc=(0, 2), rowspan=5,colspan=1)
    fig.tight_layout()

    curr_x,curr_y=np.zeros((len(data_x))), np.zeros((len(data_y)))
    theta=np.zeros((len(data_x)))
    theta[0]=0
    theta[1]=90
    
   
    net=attractorNetwork(N[1],num_links[1],excite[1], activity_mag[1],inhibit_scale[1])
    prev_weights[1][net.activation(theta[1])]=net.full_weights(num_links[1])
    
    for n in range(len(prev_weights_rot)):
        prev_weights_rot[n][net.activation(90)]=net.full_weights(num_links[1])


    def animate(i):
        global prev_weights_trans,prev_weights_rot, num_links, excite, activity_mag,inhibit_scale,split_output
        # ax10.clear(),ax11.clear(),ax12.clear(),ax13.clear(),ax14.clear()
        
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

            net=attractorNetwork(N[0],num_links[0],excite[0], activity_mag[0],inhibit_scale[0])
            decoded_translation=multiResolutionUpdate(translation,prev_weights_trans,net)
    
            # net=attractorNetwork(N[1],num_links[1],excite[1], activity_mag[1],inhibit_scale[1])
            # prev_weights[1][:]= net.update_weights_dynamics(prev_weights[1][:],rotation)
            # prev_weights[1][prev_weights[1][:]<0]=0
            # decoded_rotation=activityDecodingAngle(prev_weights[1][:],num_links[1],N[1])*1.09#-prev_trans
            # # decoded_rotation=np.argmax(prev_weights[1][:])
            
            delta, scale = multiResolutionModulus(abs(rotation),split_output)
            split_output=np.zeros((len(delta)))

            print(rotation,delta)

            '''updating network'''    
            net=attractorNetwork(N[1],num_links[1],excite[1], activity_mag[1],inhibit_scale[1])
            
            for n in range(len(delta)):
                prev_weights_rot[n][:]= net.update_weights_dynamics(prev_weights_rot[n][:],delta[n])
                prev_weights_rot[n][prev_weights_rot[n][:]<0]=0
                split_output[n]=np.argmax(prev_weights_rot[n][:])#-prev_trans
            '''decoding mangnitude and direction of movement'''
            decoded_rotation=np.sum(split_output*scale)*np.sign(rotation)
    
            # theta[i]=(theta[i-1]+rotation)%360
            # curr_x[i]=curr_x[i-1]+(translation*np.cos(np.deg2rad(theta[i-1])))
            # curr_y[i]=curr_y[i-1]+(translation*np.sin(np.deg2rad(theta[i-1])))

            
            # curr_x[i]=curr_x[i-1]+ (decoded_translation*np.cos(np.deg2rad(theta[i-1])))
            # curr_y[i]=curr_y[i-1]+ (decoded_translation*np.sin(np.deg2rad(theta[i-1])))
            curr_x[i]=curr_x[i-1]+ (decoded_translation*np.cos(np.deg2rad(decoded_rotation)))
            curr_y[i]=curr_y[i-1]+ (decoded_translation*np.sin(np.deg2rad(decoded_rotation)))

        
            # print(f"{str(i)}   {str(translation)}   {str(decoded_translation )}  ----- {np.rad2deg(math.atan2(y2-y1,x2-x1))}   {decoded_rotation}")
            '''decoding mangnitude and direction of movement'''
            
            ax0.plot(x2,y2,"k.")
            ax0.set_title("Ground Truth Position")
            ax0.axis('equal')
            # ax0.axis('equal')  
            # 
            ax2.set_title("Attractor Rotation Network")
            ax2.plot(curr_x[i],curr_y[i],'g.')
            ax2.axis('equal')

            # ax10.set_title("Whole Deg",fontsize=6)
            # ax10.bar(np.arange(N[1]),prev_weights_rot[0][:],color='blue')
            # ax10.axis('off')

            # ax11.set_title("1/10 Deg",fontsize=6)
            # ax11.bar(np.arange(N[1]),prev_weights_rot[1][:],color='purple')
            # ax11.axis('off')

            # ax12.set_title("1/100 Deg",fontsize=6)
            # ax12.bar(np.arange(N[1]),prev_weights_rot[2][:],color='pink')
            # ax12.axis('off')



            

    ani = FuncAnimation(fig, animate, interval=1,frames=len(data_x),repeat=False)
    plt.show()

'''Translation Only'''
data_y=np.zeros(1000)
# data_x=np.concatenate([np.linspace(0,0.1,10), np.linspace(0.2,1,10), np.linspace(2,10,10), np.linspace(20,100,10),np.linspace(110,1000,10)])
data_x=np.linspace(0,9990,1000)

visualiseMultiResolutionTranslation(data_x,data_y)
