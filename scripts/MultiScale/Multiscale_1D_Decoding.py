
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


def visualiseMultiResolution1D(velocities,scale,velocity_type, visualise=False):
    global prev_weights, num_links, excite, curr_parameter
    N=100
    # num_links,excite,activity_mag,inhibit_scale=6,3,1.01078180e-01,8.42457941e-01
    num_links,excite,activity_mag,inhibit_scale=3,3,0.6923,0.0064


    num_links,excite,activity_mag,inhibit_scale=1,1,9.37279639e-02,1.91339076e-02
    num_links,excite,activity_mag,inhibit_scale=1,5,0.920393357,0.04
    num_links,excite,activity_mag,inhibit_scale=4,7,9.17296828e-01,5.59935054e-03

    num_links,excite,activity_mag,inhibit_scale=1,1,4.33654032e-02,8.51548180e-02

    num_links,excite,activity_mag,inhibit_scale=1,3,1.21745813e-01,5.96673372e-02

    num_links,excite,activity_mag,inhibit_scale=1,3,0.0721745813,2.96673372e-02


    # num_links,excite,activity_mag,inhibit_scale=1,4,1.00221581e-01,1.29876096e-01
    # num_links,excite,activity_mag,inhibit_scale=9,7,8.66094143e-01,5.46047909e-02

    # num_links,excite,activity_mag,inhibit_scale=7,4,8.79135325e-03,4.64271510e-02
    # num_links,excite,activity_mag,inhibit_scale=1,1,5.90329937e-01,1.00934699e-02
    integratedPos=[0]
    decodedPos=[0]

    prev_weights=[np.zeros(N), np.zeros(N), np.zeros(N),np.zeros(N), np.zeros(N)]
    net=attractorNetwork(N,num_links,excite, activity_mag,inhibit_scale)
    for n in range(len(prev_weights)):
        prev_weights[n][net.activation(0)]=net.full_weights(num_links)


    if visualise==True:
        '''initlising network and animate figures'''
        fig = plt.figure(figsize=(6, 6))
        fig_cols=1
        ax10 = plt.subplot2grid(shape=(6, fig_cols), loc=(0, 0), rowspan=1,colspan=1)
        ax11 = plt.subplot2grid(shape=(6, fig_cols), loc=(1, 0), rowspan=1,colspan=1)
        ax12 = plt.subplot2grid(shape=(6, fig_cols), loc=(2, 0), rowspan=1,colspan=1)
        ax13 = plt.subplot2grid(shape=(6, fig_cols), loc=(3, 0), rowspan=1,colspan=1)
        ax14 = plt.subplot2grid(shape=(6, fig_cols), loc=(4, 0), rowspan=1,colspan=1)
        axtxt1 = plt.subplot2grid(shape=(6, fig_cols), loc=(5, 0), rowspan=1,colspan=1)
        fig.tight_layout()

        def animate(i):
            global prev_weights, num_links, excite,inhibit_scale
            ax10.clear(),ax11.clear(),ax12.clear(),ax13.clear(),ax14.clear(),axtxt1.clear()
            input=velocities[i]

            delta = [(input/scale[0]), (input/scale[1]), (input/scale[2]), (input/scale[3]), (input/scale[4])]
            split_output=np.zeros((len(delta)))
            '''updating network'''    
            for n in range(len(delta)):
                prev_weights[n][:]= net.update_weights_dynamics(prev_weights[n][:],delta[n])
                prev_weights[n][prev_weights[n][:]<0]=0
                split_output[n]=np.argmax(prev_weights[n][:])
            
            decoded_translation=np.sum(split_output*scale)

            integratedPos.append(integratedPos[-1]+input)
            decodedPos.append(decoded_translation)   

            print(f"{str(i)}  translation {input} integrated decoded {round(integratedPos[-1],3)}  {str(decoded_translation )}  ")
            
            # ax10.set_title(str(scale[0])+" Scale",fontsize=9)
            # ax11.set_title(str(scale[1])+" Scale",fontsize=9)
            # ax12.set_title(str(scale[2])+" Scale",fontsize=9)
            # ax13.set_title(str(scale[3])+" Scale",fontsize=9)
            # ax14.set_title(str(scale[4])+" Scale",fontsize=9)

            ax10.set_title("Centimeter (0.01m)",fontsize=9)
            ax11.set_title("Decimeter (0.1m)",fontsize=9)
            ax12.set_title("Meter",fontsize=9)
            ax13.set_title("Dekameter (10m)",fontsize=9)
            ax14.set_title("Hectometer (100m)",fontsize=9)
            
            axtxt1.text(0,0.5,f" Shift: {round(input,4)}, Decoded Trans: {round(decoded_translation,3)}", c='r')
            # axtxt.text(0,0,"Input Rot: " +str(round(rotation,3))+ " " + str(round(decoded_rotation,3)), c='m')
            axtxt1.axis('off')
            axtxt1.text(0,0,"Decoded Position of Each Network: " + str(split_output), c='r')

            ax10.bar(np.arange(N),prev_weights[0][:],color='aqua')
            ax10.get_xaxis().set_visible(False)
            ax10.spines[['top', 'bottom', 'right']].set_visible(False)

            ax11.bar(np.arange(N),prev_weights[1][:],color='green')
            ax11.get_xaxis().set_visible(False)
            ax11.spines[['top', 'bottom', 'right']].set_visible(False)

            ax12.bar(np.arange(N),prev_weights[2][:],color='blue')
            ax12.get_xaxis().set_visible(False)
            ax12.spines[['top', 'bottom', 'right']].set_visible(False)
        
            ax13.bar(np.arange(N),prev_weights[3][:],color='purple')
            ax13.get_xaxis().set_visible(False)
            ax13.spines[['top', 'bottom', 'right']].set_visible(False)
            
            ax14.bar(np.arange(N),prev_weights[4][:],color='pink')
            ax14.get_xaxis().set_visible(False)
            ax14.spines[['top', 'bottom', 'right']].set_visible(False)

        ani = FuncAnimation(fig, animate, interval=10,frames=len(velocities),repeat=False)
        plt.show()
    else: 
        fig = plt.figure(figsize=(13, 4))
        fig_rows,fig_cols=6,3
        ax0 = plt.subplot2grid(shape=(fig_rows, fig_cols), loc=(0, 0), rowspan=5,colspan=1)
        ax1 = plt.subplot2grid(shape=(fig_rows, fig_cols), loc=(0, 1), rowspan=5,colspan=1)
        ax2 = plt.subplot2grid(shape=(fig_rows, fig_cols), loc=(0, 2), rowspan=5,colspan=1)
        axtxt = plt.subplot2grid(shape=(fig_rows, fig_cols), loc=(5, 0), rowspan=1,colspan=1)
        fig.tight_layout()

        ax0.plot(velocities), ax0.set_title(f'{velocity_type} Velocity Profile')#, ax0.axis('equal')
        axtxt.axis('off'), 
        axtxt.text(0,0,f'Num_links: {num_links}, Excite_radius: {excite}, Activity_magnitude: {activity_mag}, Inhibition_scale: {inhibit_scale}', color='r',fontsize=12)

        for i in range(0,len(velocities)):
            if i>=2:
                input=velocities[i]

                delta = [(input/scale[0]), (input/scale[1]), (input/scale[2]), (input/scale[3]), (input/scale[4])]
                split_output=np.zeros((len(delta)))
                '''updating network'''    
                for n in range(len(delta)):
                    prev_weights[n][:]= net.update_weights_dynamics(prev_weights[n][:],delta[n])
                    prev_weights[n][prev_weights[n][:]<0]=0
                    split_output[n]=np.argmax(prev_weights[n][:])
                
                decoded_translation=np.sum(split_output*scale)
        
                integratedPos.append(integratedPos[-1]+input)
                decodedPos.append(decoded_translation)   

                print(f"{str(i)}  translation {input} input output {round(integratedPos[-1],3)}  {str(decoded_translation )}  ")

        fitness=np.sum(abs(np.array(integratedPos)-np.array(decodedPos)))*-1
        print(fitness), axtxt.text(0,-1,f'Error: {fitness}', fontsize=12, c='g')
        ax1.plot(integratedPos), ax1.set_title('Integrated Position')#, ax1.axis('equal'),
        ax2.plot(decodedPos), ax2.set_title('Decoded Position')#, ax2.axis('equal')
    plt.show()


def visualiseMultiResolution1DLandmark(velocities_1rep,scale,visulaise=False):
    global prev_weights, num_links, excite, curr_parameter
    N=240
    velocities=np.concatenate([velocities_1rep,velocities_1rep])
    # velocities=velocities_1rep

    # num_links,excite,activity_mag,inhibit_scale=6,3,1.01078180e-01,8.42457941e-01

    # num_links,excite,activity_mag,inhibit_scale=6,3,0.6923,0.0064 #good one 

    # num_links,excite,activity_mag,inhibit_scale=1,4,1.00221581e-01,1.29876096e-01
    # num_links,excite,activity_mag,inhibit_scale=9,7,8.66094143e-01,5.46047909e-02
    num_links,excite,activity_mag,inhibit_scale=5,10,7.79285135e-01,0.03 #0.0307657494
    num_links,excite,activity_mag,inhibit_scale=9,1,2.07046267e-03,4.45269855e-02

    num_links,excite,activity_mag,inhibit_scale=1,3,0.221745813,0.0496673372
    num_links,excite,activity_mag,inhibit_scale=1,3,0.0721745813,2.96673372e-02
    
    integratedPos=[0]
    decodedPos=[0]
    landmark_storage=[]

    prev_weights=[np.zeros(N), np.zeros(N), np.zeros(N),np.zeros(N), np.zeros(N)]
    net=attractorNetwork(N,num_links,excite, activity_mag,inhibit_scale)
    for n in range(len(prev_weights)):
        prev_weights[n][net.activation(N//2)]=net.full_weights(num_links)

    if visulaise==True:
        '''initlising network and animate figures'''
        fig = plt.figure(figsize=(12, 6))
        fig_cols=2
        ax10 = plt.subplot2grid(shape=(6, fig_cols), loc=(0, 0), rowspan=1,colspan=1)
        ax11 = plt.subplot2grid(shape=(6, fig_cols), loc=(1, 0), rowspan=1,colspan=1)
        ax12 = plt.subplot2grid(shape=(6, fig_cols), loc=(2, 0), rowspan=1,colspan=1)
        ax13 = plt.subplot2grid(shape=(6, fig_cols), loc=(3, 0), rowspan=1,colspan=1)
        ax14 = plt.subplot2grid(shape=(6, fig_cols), loc=(4, 0), rowspan=1,colspan=1)
        ax20 = plt.subplot2grid(shape=(6, fig_cols), loc=(0, 1), rowspan=1,colspan=1)
        ax21 = plt.subplot2grid(shape=(6, fig_cols), loc=(1, 1), rowspan=1,colspan=1)
        ax22 = plt.subplot2grid(shape=(6, fig_cols), loc=(2, 1), rowspan=1,colspan=1)
        ax23 = plt.subplot2grid(shape=(6, fig_cols), loc=(3, 1), rowspan=1,colspan=1)
        ax24 = plt.subplot2grid(shape=(6, fig_cols), loc=(4, 1), rowspan=1,colspan=1)
        axtxt1 = plt.subplot2grid(shape=(6, fig_cols), loc=(5, 0), rowspan=1,colspan=1)
        axtxt2 = plt.subplot2grid(shape=(6, fig_cols), loc=(5, 1), rowspan=1,colspan=1)
        fig.tight_layout()

        ax10.set_title(str(scale[0])+" Scale",fontsize=9)
        ax11.set_title(str(scale[1])+" Scale",fontsize=9)
        ax12.set_title(str(scale[2])+" Scale",fontsize=9)
        ax13.set_title(str(scale[3])+" Scale",fontsize=9)
        ax14.set_title(str(scale[4])+" Scale",fontsize=9)
        ax20.set_title(str(scale[0])+" Scale",fontsize=9)
        ax21.set_title(str(scale[1])+" Scale",fontsize=9)
        ax22.set_title(str(scale[2])+" Scale",fontsize=9)
        ax23.set_title(str(scale[3])+" Scale",fontsize=9)
        ax24.set_title(str(scale[4])+" Scale",fontsize=9)
        

        def animate(i):
            global prev_weights, num_links, excite
            ax10.clear(),ax11.clear(),ax12.clear(),ax13.clear(),ax14.clear(),axtxt1.clear(),axtxt2.clear(),axtxt1.axis('off'),axtxt2.axis('off')
            input=velocities[i]

            delta = [(input/scale[0]), (input/scale[1]), (input/scale[2]), (input/scale[3]), (input/scale[4])]
            split_output=np.zeros((len(delta)))
            '''updating network'''    
            for n in range(len(delta)):
                prev_weights[n][:]= net.update_weights_dynamics(prev_weights[n][:],delta[n])
                prev_weights[n][prev_weights[n][:]<0]=0
                split_output[n]=np.argmax(prev_weights[n][:]) - N//2
            
            decoded_translation=np.sum(split_output*scale) 
            integratedPos.append(integratedPos[-1]+input)
            decodedPos.append(decoded_translation)   

            
            if i < len(velocities_1rep) and i>1 and i%20==0:
                landmark_storage.append(prev_weights)
        
                ax20.bar(np.arange(N),landmark_storage[-1][0][:])
                ax20.get_xaxis().set_visible(False)
                ax20.spines[['top', 'bottom', 'right']].set_visible(False)

                ax21.bar(np.arange(N),landmark_storage[-1][1][:])
                ax21.get_xaxis().set_visible(False)
                ax21.spines[['top', 'bottom', 'right']].set_visible(False)

                ax22.bar(np.arange(N),landmark_storage[-1][2][:])
                ax22.get_xaxis().set_visible(False)
                ax22.spines[['top', 'bottom', 'right']].set_visible(False)
            
                ax23.bar(np.arange(N),landmark_storage[-1][3][:])
                ax23.get_xaxis().set_visible(False)
                ax23.spines[['top', 'bottom', 'right']].set_visible(False)
                
                ax24.bar(np.arange(N),landmark_storage[-1][4][:])
                ax24.get_xaxis().set_visible(False)
                ax24.spines[['top', 'bottom', 'right']].set_visible(False)
            
            elif i>len(velocities_1rep) and i%20==0:
                no_lnd=len(velocities_1rep)//20
                lndmark_id=(i//20)%no_lnd
                axtxt2.text(0,1,f'Landmark Id: {lndmark_id}', c='green')
                prev_weights+=landmark_storage[lndmark_id]

            print(f"{str(i)}  translation {input} input output {round(integratedPos[-1],3)}  {str(decoded_translation )}  ")
            
            ax10.set_title(str(scale[0])+" Scale",fontsize=9)
            ax11.set_title(str(scale[1])+" Scale",fontsize=9)
            ax12.set_title(str(scale[2])+" Scale",fontsize=9)
            ax13.set_title(str(scale[3])+" Scale",fontsize=9)
            ax14.set_title(str(scale[4])+" Scale",fontsize=9)
            
            axtxt1.text(0,0.5,f" Shift: {round(input,4)}, Decoded Trans: {round(decoded_translation,3)}", c='r')
            axtxt1.text(0,0,"Decoded Position of Each Network: " + str(split_output), c='r')

            ax10.bar(np.arange(N),prev_weights[0][:],color='aqua')
            ax10.get_xaxis().set_visible(False)
            ax10.spines[['top', 'bottom', 'right']].set_visible(False)

            ax11.bar(np.arange(N),prev_weights[1][:],color='green')
            ax11.get_xaxis().set_visible(False)
            ax11.spines[['top', 'bottom', 'right']].set_visible(False)

            ax12.bar(np.arange(N),prev_weights[2][:],color='blue')
            ax12.get_xaxis().set_visible(False)
            ax12.spines[['top', 'bottom', 'right']].set_visible(False)
        
            ax13.bar(np.arange(N),prev_weights[3][:],color='purple')
            ax13.get_xaxis().set_visible(False)
            ax13.spines[['top', 'bottom', 'right']].set_visible(False)
            
            ax14.bar(np.arange(N),prev_weights[4][:],color='pink')
            ax14.get_xaxis().set_visible(False)
            ax14.spines[['top', 'bottom', 'right']].set_visible(False)

        ani = FuncAnimation(fig, animate, interval=1,frames=len(velocities),repeat=False)
        plt.show()
    else: 
        fig = plt.figure(figsize=(13, 4))
        ax0 = fig.add_subplot(1, 3, 1)
        ax1 = fig.add_subplot(1, 3, 2)
        ax2 = fig.add_subplot(1, 3, 3)
        fig.tight_layout()

        ax0.plot(velocities), ax0.set_title('Velocity Profile')#, ax0.axis('equal')

        for i in range(0,len(velocities)):
            
            input=velocities[i]

            delta = [(input/scale[0]), (input/scale[1]), (input/scale[2]), (input/scale[3]), (input/scale[4])]
            split_output=np.zeros((len(delta)))
            '''updating network'''    
            for n in range(len(delta)):
                prev_weights[n][:]= net.update_weights_dynamics(prev_weights[n][:],delta[n])
                prev_weights[n][prev_weights[n][:]<0]=0
                split_output[n]=np.argmax(prev_weights[n][:])  - N//2

            if i < len(velocities_1rep) and i>1 and i%20==0:
                landmark_storage.append(prev_weights)

            elif i>len(velocities_1rep) and i%20==0:
                no_lnd=len(velocities_1rep)//20
                lndmark_id=(i//20)%no_lnd
                prev_weights+=landmark_storage[lndmark_id]
            
            decoded_translation=np.sum(split_output*scale)
    
            integratedPos.append(integratedPos[-1]+input)
            decodedPos.append(decoded_translation)   

            print(f"{str(i)}  translation {input} input output {round(integratedPos[-1],3)}  {str(decoded_translation )} ")

        fitness=np.sum(abs(np.array(integratedPos)-np.array(decodedPos)))*-1
        print(fitness)
        ax1.plot(integratedPos), ax1.set_title('Integrated Position')#, ax1.axis('equal'),
        ax2.plot(decodedPos), ax2.set_title('Decoded Position')#, ax2.axis('equal')
    plt.show()


def visualiseMultiResolution1DPLotAll(velocities,scale,visulaise=False):
    global prev_weights, num_links, excite
    N=200

    num_links,excite,activity_mag,inhibit_scale=1,3,0.0521745813,0.0496673372
    num_links,excite,activity_mag,inhibit_scale=9,2,1.25759298e-01,4.26232389e-02
    
    integratedPos=[0]
    decodedPos=[0]
    landmark_storage=[]

    prev_weights=[np.zeros(N), np.zeros(N), np.zeros(N),np.zeros(N), np.zeros(N)]
    net=attractorNetwork(N,num_links,excite, activity_mag,inhibit_scale)
    for n in range(len(prev_weights)):
        prev_weights[n][net.activation(N//2)]=net.full_weights(num_links)

    if visulaise==True:
        '''initlising network and animate figures'''
        fig, axs = plt.subplots(6,5, figsize=(15, 6))
        fig.subplots_adjust(hspace = .5, wspace=.001)
        axs = axs.ravel()
        from random import randint
        colors = []
        for i in range(5):
            colors.append('#%06X' % randint(0, 0xFFFFFF))

        def animate(i):
            global prev_weights, num_links, excite
            axs[25].clear(),axs[25].axis('off'),axs[26].axis('off'), axs[27].axis('off'),axs[28].axis('off'),axs[29].axis('off')
            input=velocities[i]

            delta = [(input/scale[0]), (input/scale[1]), (input/scale[2]), (input/scale[3]), (input/scale[4])]
            split_output=np.zeros((len(delta)))
            '''updating network'''    
            for n in range(len(delta)):
                prev_weights[n][:]= net.update_weights_dynamics(prev_weights[n][:],delta[n])
                prev_weights[n][prev_weights[n][:]<0]=0
                split_output[n]=np.argmax(prev_weights[n][:]) - N//2
            
            decoded_translation=np.sum(split_output*scale) 
            integratedPos.append(integratedPos[-1]+input)
            decodedPos.append(decoded_translation)   
            
            if i < len(velocities) and i>1 and i%20==0:
                landmark_storage.append(prev_weights)
                idx=(i//20)-1
        
                axs[0+idx].bar(np.arange(N),landmark_storage[-1][0][:], color=colors[idx])
                axs[0+idx].get_xaxis().set_visible(False)
                axs[0+idx].spines[['top', 'bottom', 'right']].set_visible(False)
                axs[0+idx].set_title(str(scale[0])+" Scale",fontsize=9)

                axs[5+idx].bar(np.arange(N),landmark_storage[-1][1][:],color=colors[idx])
                axs[5+idx].get_xaxis().set_visible(False)
                axs[5+idx].spines[['top', 'bottom', 'right']].set_visible(False)
                axs[5+idx].set_title(str(scale[1])+" Scale",fontsize=9)

                axs[10+idx].bar(np.arange(N),landmark_storage[-1][2][:],color=colors[idx])
                axs[10+idx].get_xaxis().set_visible(False)
                axs[10+idx].spines[['top', 'bottom', 'right']].set_visible(False)
                axs[10+idx].set_title(str(scale[2])+" Scale",fontsize=9)
            
                axs[15+idx].bar(np.arange(N),landmark_storage[-1][3][:],color=colors[idx])
                axs[15+idx].get_xaxis().set_visible(False)
                axs[15+idx].spines[['top', 'bottom', 'right']].set_visible(False)
                axs[15+idx].set_title(str(scale[3])+" Scale",fontsize=9)
                
                axs[20+idx].bar(np.arange(N),landmark_storage[-1][4][:],color=colors[idx])
                axs[20+idx].get_xaxis().set_visible(False)
                axs[20+idx].spines[['top', 'bottom', 'right']].set_visible(False)
                axs[20+idx].set_title(str(scale[4])+" Scale",fontsize=9)
            
            

            print(f"{str(i)}  translation {input} input output {round(integratedPos[-1],3)}  {str(decoded_translation )}  ")
            
            axs[25].text(0,-0.5,f"N: {N}, num_links: {num_links},excite: {excite},activity_mag: {activity_mag},inhibit_scale: {inhibit_scale}", c='b')
            axs[25].text(0,0.5,f" Shift: {round(input,4)}, Decoded Trans: {round(decoded_translation,3)}", c='r')
            axs[25].text(0,0,"Decoded Position of Each Network: " + str(split_output), c='g')



        ani = FuncAnimation(fig, animate, interval=1,frames=len(velocities),repeat=False)
        plt.show()
    else: 
        fig = plt.figure(figsize=(13, 4))
        ax0 = fig.add_subplot(1, 3, 1)
        ax1 = fig.add_subplot(1, 3, 2)
        ax2 = fig.add_subplot(1, 3, 3)
        fig.tight_layout()

        ax0.plot(velocities), ax0.set_title('Velocity Profile')#, ax0.axis('equal')

        for i in range(0,len(velocities)):
            if i>=2:
                input=velocities[i]

                delta = [(input/scale[0]), (input/scale[1]), (input/scale[2]), (input/scale[3]), (input/scale[4])]
                split_output=np.zeros((len(delta)))
                '''updating network'''    
                for n in range(len(delta)):
                    prev_weights[n][:]= net.update_weights_dynamics(prev_weights[n][:],delta[n])
                    prev_weights[n][prev_weights[n][:]<0]=0
                    split_output[n]=np.argmax(prev_weights[n][:])  - N//2
                
                decoded_translation=np.sum(split_output*scale)
        
                integratedPos.append(integratedPos[-1]+input)
                decodedPos.append(decoded_translation)   

                print(f"{str(i)}  translation {input} input output {round(integratedPos[-1],3)}  {str(decoded_translation )}  ")

        fitness=np.sum(abs(np.array(integratedPos)-np.array(decodedPos)))*-1
        print(fitness)
        ax1.plot(integratedPos), ax1.set_title('Integrated Position')#, ax1.axis('equal'),
        ax2.plot(decodedPos), ax2.set_title('Decoded Position')#, ax2.axis('equal')
    plt.show()


def pathIntegrationVelOnly(rot,velocities,velocity_type):
    N=100
    num_links,excite,activity_mag,inhibit_scale=1,3,0.0721745813,2.96673372e-02
    integratedMag=[0]
    decodedMag=[0]
    t=np.arange(len(velocities))
    x,y=np.zeros(len(velocities)),np.zeros(len(velocities))
    tru_x,tru_y=np.zeros(len(velocities)),np.zeros(len(velocities))
    
    prev_weights=[np.zeros(N), np.zeros(N), np.zeros(N),np.zeros(N), np.zeros(N)]
    net=attractorNetwork(N,num_links,excite, activity_mag,inhibit_scale)
    for n in range(len(prev_weights)):
        prev_weights[n][net.activation(0)]=net.full_weights(num_links)

    fig = plt.figure(figsize=(13, 4))
    fig_rows,fig_cols=6,3
    ax0 = plt.subplot2grid(shape=(fig_rows, fig_cols), loc=(0, 0), rowspan=5,colspan=1)
    ax1 = plt.subplot2grid(shape=(fig_rows, fig_cols), loc=(0, 1), rowspan=5,colspan=1)
    ax2 = plt.subplot2grid(shape=(fig_rows, fig_cols), loc=(0, 2), rowspan=5,colspan=1)
    axtxt = plt.subplot2grid(shape=(fig_rows, fig_cols), loc=(5, 0), rowspan=1,colspan=1)
    fig.tight_layout()
    ax1.axis('equal'),ax2.axis('equal')

    ax0.plot(velocities), ax0.set_title(f'{velocity_type} Velocity Profile')#, ax0.axis('equal')
    axtxt.axis('off'), 
    axtxt.text(0,0,f'Num_links: {num_links}, Excite_radius: {excite}, Activity_magnitude: {activity_mag}, Inhibition_scale: {inhibit_scale}', color='r',fontsize=12)

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

        x[i]=x[i-1]+(decodedMag[i]-decodedMag[i-1])*np.cos(rot[i])
        y[i]=y[i-1]+(decodedMag[i]-decodedMag[i-1])*np.sin(rot[i])

        tru_x[i]=tru_x[i-1]+input*np.cos(rot[i])
        tru_y[i]=tru_y[i-1]+input*np.sin(rot[i])

        print(f"{str(i)}  translation {input} input output {round(integratedMag[-1],3)}  {str(decoded_translation )}  ")

    fitness=np.sum(abs(np.array(integratedMag)-np.array(decodedMag)))*-1
    print(fitness), axtxt.text(0,-1,f'Error: {fitness}', fontsize=12, c='g')
    ax1.plot(t,integratedMag, t, decodedMag), ax1.set_title('Magnitudes'),ax1.legend(['Ground Truth','Decoded Magnitude'])
    ax2.plot(tru_x,tru_y,x,y), ax2.set_title('Path Integration'),ax2.legend(['Ground Truth','Integrated Mag&Rot'])
    plt.show()

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


'''Test Networks'''
gt,mag,rot=data_processing_groundTruth()
vn,ve,vf,vl,vu=data_processing_oxts()
scale=[0.25,0.5,1,2,4]

# increasing 
velocities=np.concatenate([np.array([0.01]*25), np.zeros(25), np.array([0.1]*25), np.zeros(25), np.array([1]*25), np.zeros(25), np.array([10]*25), np.zeros(25), np.array([100]*25)])
# velocities increasing then decreainf 
velocities=np.concatenate([np.array([scale[0]]*25), np.array([scale[1]]*25), np.array([scale[2]]*25), np.array([scale[3]]*25), np.array([scale[4]]*25), np.array([scale[3]]*25),  np.array([scale[2]]*25),  np.array([scale[1]]*25),  np.array([scale[0]]*25)])
# random uniform distribution 
# velocities=np.concatenate([np.random.uniform(0,10,20), np.random.uniform(0,1,20), np.random.uniform(0,0.1,21)])

# velocities=np.array([scale[0]]*2)

# pathIntegrationVelOnly(rot,mag,'KittiDataset')
# visualiseMultiResolution1D(velocities,scale,'Curated',visualise=True)

# visualiseMultiResolution1DLandmark(velocities,scale,visulaise=False)
# visualiseMultiResolution1DPLotAll(velocities,scale,visulaise=True)




