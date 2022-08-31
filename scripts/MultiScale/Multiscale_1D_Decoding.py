import matplotlib.pyplot as plt
import numpy as np
import math
import os
import pandas as pd
from matplotlib.animation import FuncAnimation
from matplotlib.artist import Artist
from mpl_toolkits.mplot3d import Axes3D 
from scipy import signal
import time 
from os import listdir
import sys
sys.path.append('./scripts')
from CAN import  attractorNetworkSettling, attractorNetwork, attractorNetworkScaling, attractorNetwork2D



def visualiseMultiResolution1D(velocities,scale,velocity_type, visualise=False):
    global prev_weights, num_links, excite, curr_parameter
    N=100
    # num_links,excite,activity_mag,inhibit_scale=6,3,1.01078180e-01,8.42457941e-01
    num_links,excite,activity_mag,inhibit_scale=3,3,0.6923,0.0064
    # num_links,excite,activity_mag,inhibit_scale=1,4,1.00221581e-01,1.29876096e-01
    # num_links,excite,activity_mag,inhibit_scale=9,7,8.66094143e-01,5.46047909e-02
    integratedPos=[0]
    decodedPos=[0]

    prev_weights=[np.zeros(N), np.zeros(N), np.zeros(N),np.zeros(N), np.zeros(N)]
    net=attractorNetwork(N,num_links,excite, activity_mag,inhibit_scale)
    for n in range(len(prev_weights)):
        prev_weights[n][net.activation(N//2)]=net.full_weights(num_links)


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
                split_output[n]=np.argmax(prev_weights[n][:])-N//2
            
            decoded_translation=np.sum(split_output*scale)

            integratedPos.append(integratedPos[-1]+input)
            decodedPos.append(decoded_translation)   

            print(f"{str(i)}  translation {input} integrated decoded {round(integratedPos[-1],3)}  {str(decoded_translation )}  ")
            
            ax10.set_title(str(scale[0])+" Scale",fontsize=9)
            ax11.set_title(str(scale[1])+" Scale",fontsize=9)
            ax12.set_title(str(scale[2])+" Scale",fontsize=9)
            ax13.set_title(str(scale[3])+" Scale",fontsize=9)
            ax14.set_title(str(scale[4])+" Scale",fontsize=9)
            
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
                    split_output[n]=np.argmax(prev_weights[n][:])-N//2
                
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
    N=200
    velocities=np.concatenate([velocities_1rep,velocities_1rep,velocities_1rep])
    # num_links,excite,activity_mag,inhibit_scale=6,3,1.01078180e-01,8.42457941e-01
    num_links,excite,activity_mag,inhibit_scale=6,3,0.6923,0.0064
    # num_links,excite,activity_mag,inhibit_scale=1,4,1.00221581e-01,1.29876096e-01
    # num_links,excite,activity_mag,inhibit_scale=9,7,8.66094143e-01,5.46047909e-02
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

'''test'''
vels=np.concatenate([np.array([0.01]*25), np.zeros(25), np.array([0.1]*25), np.zeros(25), np.array([1]*25), np.zeros(25), np.array([10]*25), np.zeros(25), np.array([100]*25)])

scale=[0.01,0.1,1,10,100]
path='./data/2011_09_26_2/2011_09_26_drive_0001_sync/oxts/data/'
filenames = [f for f in listdir(path)]
filenames.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
'''
vn:    velocity towards north (m/s)
ve:    velocity towards east (m/s)
vf:    forward velocity, i.e. parallel to earth-surface (m/s)
vl:    leftward velocity, i.e. parallel to earth-surface (m/s)
vu:    upward velocity, i.e. perpendicular to earth-surface (m/s)'''
vn,ve,vf,vl,vu=np.zeros(len(filenames)), np.zeros(len(filenames)), np.zeros(len(filenames)), np.zeros(len(filenames)), np.zeros(len(filenames))
for i in range(len(filenames)):
    cur_file=pd.read_csv(path+filenames[i], delimiter=' ', header=None)
    vn[i]=cur_file[7]
    ve[i]=cur_file[8]
    vf[i]=cur_file[9]
    vl[i]=cur_file[10]
    vu[i]=cur_file[11]
# print(vf)
# print(vl)


# visualiseMultiResolution1D(vu,scale,'East')
visualiseMultiResolution1DLandmark(ve,scale,visulaise=True)