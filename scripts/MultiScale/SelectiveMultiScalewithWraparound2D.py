
import matplotlib.pyplot as plt
import numpy as np
import random  
import math
import os
import pandas as pd
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap
from matplotlib.artist import Artist
from mpl_toolkits.mplot3d import Axes3D 
from scipy import signal
from scipy import ndimage
import time 
from os import listdir
import sys
sys.path.append('./scripts')
import CAN
from CAN import attractorNetwork2D, attractorNetwork, activityDecodingAngle, activityDecoding
import CAN as can
# import pykitti
import json 
from DataHandling import saveOrLoadNp  
# import scienceplots
# plt.style.use(['science','ieee'])
plt.style.use(['science','no-latex'])


def pathIntegration(speed, angVel):
    q=[0,0,0]
    x_integ,y_integ=[],[]
    for i in range(len(speed)):
        q[0],q[1]=q[0]+speed[i]*np.cos(q[2]), q[1]+speed[i]*np.sin(q[2])
        q[2]+=angVel[i]
        x_integ.append(round(q[0],4))
        y_integ.append(round(q[1],4))

    return x_integ, y_integ

def errorTwoCoordinateLists(x1, y1, x2, y2):
    x_error=np.sum(np.abs(np.array(x1) - np.array(x2)))
    y_error=np.sum(np.abs(np.array(y1) - np.array(y2)))

    return (x_error+y_error)

def scale_selection(input,scales, swap_val=1):
    if len(scales)==1:
        scale_idx=0
    else: 

        if input<=scales[0]*swap_val:
            scale_idx=0
        

        # elif input>scales[0]*swap_val and input<=scales[1]*swap_val:
        #         scale_idx=1 
        
        # elif input>scales[1]*swap_val and input<=scales[1]*swap_val:
        #         scale_idx=2
        
        # elif input>scales[2]*swap_val and input<=scales[1]*swap_val:
        #         scale_idx=3
        
        # elif input>scales[3]*swap_val and input<=scales[1]*swap_val:
        #         scale_idx=4
        for i in range(len(scales)-2):
            if input>scales[i]*swap_val and input<=scales[i+1]*swap_val:
                scale_idx=i+1
        
        if input>scales[-2]*swap_val:
            scale_idx=len(scales)-1
    # elif input>scales[2]*swap_val and input<=scales[3]*swap_val:
    #     scale_idx=3
    # elif input>scales[3]*swap_val:
    #     scale_idx=4
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


# outfile='./results/testEnvPathVelocities.npy'
# vel_x,vel_y=np.load(outfile)

# velocities=np.concatenate([np.random.uniform(0,0.25,20), np.random.uniform(0.25,1,20), np.random.uniform(1,4,20), np.random.uniform(4,16,20), np.random.uniform(16,100,20)])
# num_links,excite,activity_mag,inhibit_scale,iterations,wrap_iterations=1,1,1,0.0005,1, 1
# num_links,excite,activity_mag,inhibit_scale,iterations,wrap_iterations=8,1,0.27758052,0.08663314,3,6


# GIF_MultiResolutionFeedthrough2D(vel_x,vel_y,scales)
# MultiResolutionFeedthrough2D(vel_x, vel_y,scales)

# prev_weights=[np.zeros((N,N))+2,np.zeros((N,N))+1,np.zeros((N,N)),np.zeros((N,N)),np.zeros((N,N)),np.zeros((N,N))]
# plt.imshow(prev_weights[0][:][:])
# plt.show()
def shfitingPeakDecoding2D(prev_weights, peakRowIdx, peakColIdx, radius, N):
    diam=(radius*2)+1
    new_array=np.zeros((diam,diam))
    rows=np.arange(peakRowIdx-radius, peakRowIdx+radius+1)
    cols=np.arange(peakColIdx-radius, peakColIdx+radius+1)

    for j in range(diam):
        for k in range(diam):
            new_array[j,k]=(prev_weights[rows[j]%N,cols[k]%N])

    x,y=ndimage.center_of_mass(new_array)
    CM=(x+peakColIdx-radius, y+peakRowIdx-radius)

    return CM

def headDirection(theta_weights, angVel, init_angle):
    global theata_called_iters
    N=360
    # num_links,excite,activity_mag,inhibit_scale, iterations=16, 17, 2.16818183,  0.0281834545, 2
    num_links,excite,activity_mag,inhibit_scale, iterations=16, 17, 2.16818183,  0.0381834545, 2
    num_links,excite,activity_mag,inhibit_scale, iterations=13,4,2.70983783e+00,4.84668851e-02,2
    net=attractorNetwork(N,num_links,excite, activity_mag,inhibit_scale)
    
    if theata_called_iters==0:
        theta_weights[net.activation(init_angle)]=net.full_weights(num_links)
        theata_called_iters+=1

    for j in range(iterations):
        theta_weights=net.update_weights_dynamics(theta_weights,angVel)
        theta_weights[theta_weights<0]=0
    
    
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
    num_links,excite,activity_mag,inhibit_scale, iterations=7,2,1.96188442, 0.0420970698, 2
    num_links,excite,activity_mag,inhibit_scale, iterations=7,8,5.47157578e-01 ,3.62745653e-04, 2
    prev_weights=np.zeros((N,N))
    network=attractorNetwork2D(N,N,num_links,excite, activity_mag,inhibit_scale)
    prev_weights=network.excitations(50,50)
    x,y=50,50
    dirs=np.arange(0,90)
    speeds=np.linspace(0.1,1.1,90)
    x_integ, y_integ=[50],[50]
    x_grid, y_grid=[50], [50]

    for i in range(len(speeds)):
        for j in range(iterations):
            prev_weights=network.update_weights_dynamics(prev_weights, dirs[i], speeds[i])
            prev_weights[prev_weights<0]=0

        x_grid.append(np.argmax(np.max(prev_weights, axis=1)))
        y_grid.append(np.argmax(np.max(prev_weights, axis=0)))

        x,y=x+speeds[i]*np.sin(np.deg2rad(dirs[i])), y+speeds[i]*np.cos(np.deg2rad(dirs[i]))
        x_integ.append(round(x))
        y_integ.append(round(y))


    x_error=np.sum(np.abs(np.array(x_grid) - np.array(x_integ)))
    y_error=np.sum(np.abs(np.array(y_grid) - np.array(y_integ)))
    print(x_integ, y_integ)
    print(x_grid, y_grid)

    return (x_error+y_error)

# attractorGridcell()
# attractorGridcell_fitness()
# for i in range(1,360):
#     theta_weights = headDirection(theta_weights, 1)
# purePursuitFile='./results/vehiclePosisitionsPurePursuit.npy'
# vel_purePursuitFile='./results/vehicleVelocitiesPurePursuit.npy'
# vel,angVel=zip(*np.load(vel_purePursuitFile))



def TESTINGhierarchicalNetwork2DGrid(prev_weights, net,N, vel, direction, iterations, wrap_iterations, wrap_counter, x_grid_expect, y_grid_expect, scales):
    delta = [(vel/scales[0]), (vel/scales[1]), (vel/scales[2]), (vel/scales[3]), (vel/scales[4])]

    cs_idx=scale_selection(vel,scales)
    wrap_rows=np.zeros((len(scales)))
    wrap_cols=np.zeros((len(scales)))

    '''Update selected scale'''
    del_x_cs, del_y_cs= delta[cs_idx]*np.cos(np.deg2rad(direction)), delta[cs_idx]*np.sin(np.deg2rad(direction))
    x_grid_expect[cs_idx]=(x_grid_expect[cs_idx]+(del_x_cs *scales[cs_idx]))%(N*scales[cs_idx])
    y_grid_expect[cs_idx]=(y_grid_expect[cs_idx]+(del_y_cs *scales[cs_idx]))%(N*scales[cs_idx])

    for i in range(iterations):
        prev_weights[cs_idx][:], wrap_rows_cs, wrap_cols_cs= net.update_weights_dynamics(prev_weights[cs_idx][:],direction, delta[cs_idx])
        prev_weights[cs_idx][prev_weights[cs_idx][:]<0]=0
        wrap_rows[cs_idx]+=wrap_rows_cs
        wrap_cols[cs_idx]+=wrap_cols_cs
    if np.any(wrap_cols!=0):
        print(f"------------------------------------------------------------------------------------------------------wrap_cols {wrap_cols}")
    if np.any(wrap_rows!=0):
        print(f"------------------------------------------------------------------------------------------------------wrap_rows {wrap_rows}")
    

    '''Update the larger scale based on wraparound in smaller scales'''
    def feedthrough_update(wrap_scale, update_scale, prev_weights, wrap_rows, wrap_cols):
        if cs_idx!=wrap_scale and (wrap_rows[wrap_scale]!=0 or wrap_cols[wrap_scale]!=0 ): 
            del_rows, del_cols=(wrap_rows[wrap_scale]*scales[wrap_scale]*N)/scales[update_scale], (wrap_cols[wrap_scale]*scales[wrap_scale]*N)/scales[update_scale]  
            direction=np.rad2deg(math.atan2(del_rows, del_cols))
            distance=math.sqrt(del_cols**2 + del_rows**2)
   
            x_grid_expect[update_scale]=(x_grid_expect[update_scale]+(del_cols *scales[update_scale]))%(N*scales[update_scale])
            y_grid_expect[update_scale]=(y_grid_expect[update_scale]+(del_rows *scales[update_scale]))%(N*scales[update_scale])

            for i in range(wrap_iterations):
                prev_weights[update_scale][:], wrap_rows_update, wrap_cols_update= net.update_weights_dynamics(prev_weights[update_scale][:],direction, distance)
                prev_weights[update_scale][prev_weights[update_scale][:]<0]=0
                wrap_rows[update_scale]+=wrap_rows_update
                wrap_cols[update_scale]+=wrap_cols_update
            if np.any(wrap_cols!=0):
                print(f"------------------------------------------------------------------------------------------------------wrap_cols {wrap_cols}")
            if np.any(wrap_rows!=0):
                print(f"------------------------------------------------------------------------------------------------------wrap_rows {wrap_rows}")
    

    feedthrough_update(0, 2, prev_weights, wrap_rows, wrap_cols) # scale 0.1 wrap into scale 10
    feedthrough_update(1, 3, prev_weights, wrap_rows, wrap_cols) # scale 1 wrap into scale 100
    feedthrough_update(2, 4, prev_weights, wrap_rows, wrap_cols) # scale 10 wrap into scale 1000
    feedthrough_update(3, 5, prev_weights, wrap_rows, wrap_cols) # scale 100 wrap into scale 10000


    # if (cs_idx != 4) and (wrap_rows[cs_idx]!=0 or wrap_cols[cs_idx]!=0 ): 
    #     del_rows_100, del_cols_100=(wrap_rows[cs_idx]*scales[cs_idx]*N)/scales[4], (wrap_cols[cs_idx]*scales[cs_idx]*N)/scales[4]  
    #     direction_100=np.rad2deg(math.atan2(del_rows_100, del_cols_100))
    #     distance_100=math.sqrt(del_cols_100**2 + del_rows_100**2)
    #     print(f"delta row col {del_rows_100}, {del_cols_100}")
    #     print(f"dist, dir {distance_100}, {direction_100}")

    #     x_grid_expect[4]=(x_grid_expect[4]+(del_cols_100 *scales[4]))%(N*scales[4])
    #     y_grid_expect[4]=(y_grid_expect[4]+(del_rows_100 *scales[4]))%(N*scales[4])
    #     # wraparound[4]=(can.activityDecoding(prev_weights[4][:],4,N) + update_amount)//(N-1)
    #     for i in range(wrap_iterations):
    #         prev_weights[4][:], wrap_rows_100, wrap_cols_100= net.update_weights_dynamics(prev_weights[4][:],direction_100, distance_100,4,wrap_counter)
    #         prev_weights[4][prev_weights[4][:]<0]=0
    #         wrap_rows[4]+=wrap_rows_100
    #         wrap_cols[4]+=wrap_cols_100


    if np.any(wrap_cols!=0):
        print(f"------------------------------------------------------------------------------------------------------wrap_cols {wrap_cols}")
    if np.any(wrap_rows!=0):
        print(f"------------------------------------------------------------------------------------------------------wrap_rows {wrap_rows}")
    
    if np.any(wrap_cols!=0) or np.any(wrap_rows!=0):
        wrap=1
    else:
        wrap=0

       
    return prev_weights, wrap, x_grid_expect, y_grid_expect

def TESTINGheadDirectionAndPlace():
    '''change decoding so that very small fractional shifts can be made in the 100 scale, tune the iterations to get accurate tracking'''
    global theata_called_iters,theta_weights, prev_weights, q, wrap_counter, current_i, x_grid_expect, y_grid_expect 
    # scales=[0.01,1,10,100,1000, 10000]
    scales=[0.25,1,4,16,100,10000]

    theta_weights=np.zeros(360)
    theata_called_iters=0

    # start_x, start_y= 290, 547
    start_x, start_y=500000,500000
    N=100
    wrap_counter=[0,0,0,0,0,0]
    # num_links,excite,activity_mag,inhibit_scale, iterations, wrap_iterations=7,8,5.47157578e-01 ,3.62745653e-04, 2, 2 #good only at small scale
    # num_links,excite,activity_mag,inhibit_scale, iterations, wrap_iterations=72,1,9.05078199e-01,7.85317908e-04,4,1
    # num_links,excite,activity_mag,inhibit_scale, iterations, wrap_iterations=6,1,3.89338335e-01,1.60376324e-04, 3,3  #improved at larger scale 

    num_links,excite,activity_mag,inhibit_scale, iterations, wrap_iterations=7,1,2.59532708e-01 ,2.84252467e-04,4,3 #without decimals 1000 iters fitness -5000
    # num_links,excite,activity_mag,inhibit_scale, iterations, wrap_iterations=10,2,1.10262708e-01,6.51431074e-04,3,4 #with decimals 200 iters fitness -395
  
    network=attractorNetwork2D(N,N,num_links,excite, activity_mag,inhibit_scale)
    prev_weights=[np.zeros((N,N)),np.zeros((N,N)),np.zeros((N,N)),np.zeros((N,N)),np.zeros((N,N)), np.zeros((N,N))]
    for n in range(len(prev_weights)):
        prev_weights[n]=network.excitations(0,0)
        prev_weights[n]=network.update_weights_dynamics_row_col(prev_weights[n][:], 0, 0)

    start_idx=5#scale_selection(start_mag,scales)
   
    prev_weights[start_idx]=network.excitations(50,50)
    prev_weights[start_idx][:]= network.update_weights_dynamics_row_col(prev_weights[start_idx][:],0,0)
    prev_weights[start_idx][prev_weights[start_idx][:]<0]=0

    x_grid, y_grid=[], []
    x_grid_expect, y_grid_expect =[0,0,0,0,0,0],[0,0,0,0,0,0]
    x_grid_expect[start_idx], y_grid_expect[start_idx] = start_x,start_y
    x_integ, y_integ=[],[]
    x_integ_error, y_integ_error=[], []
    q=[start_x,start_y,0]
    q_e=[start_x,start_y,0]
    wrapPos=[]
    current_i=-1
    # q=[0,0,0]
    
    for i in range(test_length):   
    # fig, axs = plt.subplots(1,1,figsize=(5, 5)) 
    # def animate(i):
    #     global theta_weights, prev_weights, q, wrap_counter, current_i, x_grid_expect, y_grid_expect

        # error=0.01
        # if i%20 ==0:
        #     vel[i]+=error

        N_dir=360
        theta_weights=headDirection(theta_weights, np.rad2deg(angVel[i]), 0)
        direction=np.argmax(theta_weights)
        hD_x,hD_y=(theta_weights*np.cos(np.deg2rad(np.arange(N_dir)*360/N)))[::3], (theta_weights*np.sin(np.deg2rad(np.arange(N_dir)*360/N)))[::3]

        # prev_weights, wrap,x_grid_expect, y_grid_expect= TESTINGhierarchicalNetwork2DGrid(prev_weights, network, N, vel[i], direction, iterations,wrap_iterations, wrap_counter, x_grid_expect, y_grid_expect, scales)
        prev_weights, wrap= hierarchicalNetwork2DGrid(prev_weights, network, N, vel[i], direction, iterations,wrap_iterations, wrap_counter, scales)

        '''1D method for decoding'''
        maxXPerScale, maxYPerScale = np.array([np.argmax(np.max(prev_weights[m], axis=1)) for m in range(len(scales))]), np.array([np.argmax(np.max(prev_weights[m], axis=0)) for m in range(len(scales))])
        decodedXPerScale=[activityDecoding(prev_weights[m][maxXPerScale[m], :],5,N)*scales[m] for m in range(len(scales))]
        decodedYPerScale=[activityDecoding(prev_weights[m][:,maxYPerScale[m]],5,N)*scales[m] for m in range(len(scales))]
        print(decodedXPerScale, decodedYPerScale)
        # decodedXPerScale.append(x_grid_expect[-1])
        # decodedYPerScale.append(y_grid_expect[-1])
        x_multiscale_grid, y_multiscale_grid=np.sum(decodedXPerScale), np.sum(decodedYPerScale)

        '''Peak'''
        # x_multiscale_grid=np.sum(np.array([np.argmax(np.max(prev_weights[m], axis=1)) for m in range(len(scales))])*scales)
        # y_multiscale_grid=np.sum(np.array([np.argmax(np.max(prev_weights[m], axis=0)) for m in range(len(scales))])*scales)

        '''Determining Fractional shift from CoM of nearby activity'''
        # maxXPerScale, maxYPerScale = np.array([np.argmax(np.max(prev_weights[m], axis=1)) for m in range(len(scales))]), np.array([np.argmax(np.max(prev_weights[m], axis=0)) for m in range(len(scales))])
        # radius=5
        # CM =[shfitingPeak2D(prev_weights[m], maxYPerScale[m], maxXPerScale[m], radius, N) for m in range(len(scales))]
        # # print(CM)
        

        x_grid.append(x_multiscale_grid-start_x)
        y_grid.append(y_multiscale_grid-start_y)

        # q_e[0],q_e[1]=q_e[0]+vel[i]*np.cos(q_e[2]), q_e[1]+vel[i]*np.sin(q_e[2])
        # q_e[2]+=angVel[i]
        # x_integ_error.append(round(q_e[0],4))
        # y_integ_error.append(round(q_e[1],4))

        # if i%20 ==0:
        #     vel[i]-=error

        q[0],q[1]=q[0]+vel[i]*np.cos(q[2]), q[1]+vel[i]*np.sin(q[2])
        q[2]+=angVel[i]
        x_integ.append(q[0]-start_x)
        y_integ.append(q[1]-start_y)

        

        
        # if wrap==1:
        #     wrapPos.append((x_grid[-1], y_grid[-1]))
   

        print(x_integ[-1], y_integ[-1])
        print(x_grid[-1], y_grid[-1])
        print('')

        # for k in range(nrows-1):
        #     axs[0][k].imshow(prev_weights[k][:][:])#(np.arange(N),prev_weights[k][:],color=colors[k])
        #     axs[0][k].spines[['top', 'left', 'right']].set_visible(False)
        #     axs[0][k].invert_yaxis()

        # axs[0][nrows-1].clear()
        # axs[1][nrows-1].axis('off')
        # axs[0][nrows-1].set_xlim([-0.2*12,0.2*12])
        # axs[0][nrows-1].set_ylim([-0.2*12,0.2*12])

        # for j in range(len(hD_x)):
        #     axs[0][nrows-1].arrow(0,0,hD_x[j]*10,hD_y[j]*10, color='m')

        # for l in range(nrows-1): 
        #     axs[1][l].clear()
        #     # axs[1][l].set_title(f"Scale{l}")
        #     axs[1][l].text(0,1,f"Decode: {np.round(decodedXPerScale[l],2)},{np.round(decodedYPerScale[l],2)}")
        #     axs[1][l].text(0,0.5,f"Expect: {np.round(x_grid_expect[l],2)},{np.round(y_grid_expect[l],2)}")
        #     axs[1][l].axis('off')

    
    # ani = FuncAnimation(fig, animate, interval=1,frames=400,repeat=False)
    # plt.show()

    # f = "./results/GIFs/BerlinPathMultiscaleAttractor100ScaleTestingWRappingErrors.gif" 
    # writergif = animation.PillowWriter(fps=25) 
    # ani.save(f, writer=writergif)

    # outfile='./results/xGrid_yGrid7.npy'
    # np.save(outfile, np.array([x_grid, y_grid]))
    # outfile='./results/wrapPos7.npy'
    # np.save(outfile, np.array(wrapPos))

    x_error=np.sum(np.abs(np.array(x_grid) - np.array(x_integ)))
    y_error=np.sum(np.abs(np.array(y_grid) - np.array(y_integ)))

    print((x_error+y_error)*-1)

    # # wrap_x,wrap_y=zip(*wrapPos)
    plt.plot(x_integ, y_integ, 'g.')
    # plt.plot(x_integ_error, y_integ_error, 'm--')
    plt.plot(x_grid, y_grid, 'b.')
    # plt.plot(wrap_x, wrap_y,'r*')

    plt.axis('equal')
    plt.title('Test Environment 2D space')
    # plt.legend(('Path Integration', 'Path Integration with Error','Multiscale Grid Decoding (err inc)', 'Instances of Wraparound'))
    plt.legend(('Path Integration', 'Multiscale Grid Decoding'))
    plt.show()



def hierarchicalNetwork2DGridMultiParameter(prev_weights, net,N, vel, direction, iterations, wrap_iterations, wrap_counter, scales):
    delta = [(vel/scales[0]), (vel/scales[1]), (vel/scales[2]), (vel/scales[3]), (vel/scales[4])]

    cs_idx=scale_selection(vel,scales)
    wrap_rows=np.zeros((len(scales)))
    wrap_cols=np.zeros((len(scales)))


    '''Update selected scale'''
    for i in range(iterations[cs_idx]):
        prev_weights[cs_idx][:], wrap_rows_cs, wrap_cols_cs= net[cs_idx].update_weights_dynamics(prev_weights[cs_idx][:],direction, delta[cs_idx])
        prev_weights[cs_idx][prev_weights[cs_idx][:]<0]=0
        wrap_rows[cs_idx]+=wrap_rows_cs
        wrap_cols[cs_idx]+=wrap_cols_cs

    '''Update the 100 scale based on wraparound in any of the previous scales'''
    if (cs_idx != 4) and (wrap_rows[cs_idx]!=0 or wrap_cols[cs_idx]!=0 ): 
        del_rows_100, del_cols_100=(wrap_rows[cs_idx]*scales[cs_idx]*N)/scales[4], (wrap_cols[cs_idx]*scales[cs_idx]*N)/scales[4]  
        direction_100=np.rad2deg(math.atan2(del_rows_100, del_cols_100))
        distance_100=math.sqrt(del_cols_100**2 + del_rows_100**2)
        # wraparound[4]=(can.activityDecoding(prev_weights[4][:],4,N) + update_amount)//(N-1)
        for i in range(wrap_iterations[4]):
            prev_weights[4][:], wrap_rows_100, wrap_cols_100= net[4].update_weights_dynamics(prev_weights[4][:],direction_100, distance_100)
            prev_weights[4][prev_weights[4][:]<0]=0
            wrap_rows[4]+=wrap_rows_100
            wrap_cols[4]+=wrap_cols_100

    '''Update the 10000 scale based on wraparound in the 100 scale'''
    if (wrap_rows[-2]!=0 or wrap_cols[-2]!=0 ):
        del_rows_10000, del_cols_10000=(wrap_rows[-2]*scales[-2]*N)/scales[5], (wrap_cols[-2]*scales[-2]*N)/scales[5]  
        direction_10000=np.rad2deg(math.atan2(del_rows_10000, del_cols_10000))
        distance_10000=math.sqrt(del_cols_10000**2 + del_rows_10000**2)
        for i in range(wrap_iterations[-1]):
            prev_weights[-1][:], wrap_rows[-1], wrap_cols[-1]= net[-1].update_weights_dynamics(prev_weights[-1][:],direction_10000, distance_10000)
            prev_weights[-1][prev_weights[-1][:]<0]=0
    
    # if np.any(wrap_cols!=0):
    #     print(f"wrap_cols {wrap_cols}")
    # if np.any(wrap_rows!=0):
    #     print(f"wrap_rows {wrap_rows}")
    
    # if np.any(wrap_cols!=0) or np.any(wrap_rows!=0):
    #     wrap=1
    # else:
    #     wrap=0

       
    return prev_weights

def headDirectionAndPlaceMultiparameter():
    global theata_called_iters,theta_weights, prev_weights, q, wrap_counter

    filename=f'./results/GA_MultiScale/tuningGrid8.npy'
    with open(filename, 'rb') as f:
        data = np.load(f)

    genome=np.array(data[0,0,:-1])

    scales=[0.25,1,4,16,100,1000]
    test_length=500
    kinemVelFile='./results/TestEnvironmentFiles/TraverseInfo/testEnvPathVelocities.npy'
    kinemAngVelFile='./results/TestEnvironmentFiles/TraverseInfo/testEnvPathAngVelocities.npy'
    vel,angVel=np.load(kinemVelFile), np.load(kinemAngVelFile)
    vel=np.concatenate([np.linspace(0,scales[0]*5,test_length//5), np.linspace(scales[0]*5,scales[1]*5,test_length//5), np.linspace(scales[1]*5,scales[2]*5,test_length//5), np.linspace(scales[2]*5,scales[3]*5,test_length//5), np.linspace(scales[3]*5,scales[4]*5,test_length//5)])


    
    theta_weights=np.zeros(360)
    theata_called_iters=0
    start_x, start_y=5000,5000
    N=100
    wrap_counter=[0,0,0,0,0,0]

    networks=[]
    iterations=[]
    wrap_iterations=[]
    for i in range(0, 36, 6):
        networks.append(attractorNetwork2D(N,N,genome[i],genome[i+1], genome[i+2],genome[i+3]))
        iterations.append(int(genome[i+4]))
        wrap_iterations.append(int(genome[i+5]))

    # network=attractorNetwork2D(N,N,num_links,excite, activity_mag,inhibit_scale)
    prev_weights=[np.zeros((N,N)),np.zeros((N,N)),np.zeros((N,N)),np.zeros((N,N)),np.zeros((N,N)), np.zeros((N,N))]
    for n in range(len(prev_weights)):
        prev_weights[n]=networks[0].excitations(0,0)
        prev_weights[n]=networks[0].update_weights_dynamics_row_col(prev_weights[n][:], 0, 0)

    start_idx=5
    prev_weights[start_idx]=networks[start_idx].excitations(50,50)
    prev_weights[start_idx][:]= networks[start_idx].update_weights_dynamics_row_col(prev_weights[start_idx][:],0,0)
    prev_weights[start_idx][prev_weights[start_idx][:]<0]=0

    x_grid, y_grid=[], []
    x_integ, y_integ=[],[]
    q=[start_x,start_y,0]

    for i in range(test_length):
        theta_weights=headDirection(theta_weights, np.rad2deg(angVel[i]), 0)
        direction=np.argmax(theta_weights)

        prev_weights= hierarchicalNetwork2DGridMultiParameter(prev_weights, networks, N, vel[i], direction, iterations,wrap_iterations, wrap_counter, scales)
        
        maxXPerScale, maxYPerScale = np.array([np.argmax(np.max(prev_weights[m], axis=1)) for m in range(len(scales))]), np.array([np.argmax(np.max(prev_weights[m], axis=0)) for m in range(len(scales))])
        decodedXPerScale=[can.activityDecoding(prev_weights[m][maxXPerScale[m], :],5,N)*scales[m] for m in range(len(scales))]
        decodedYPerScale=[can.activityDecoding(prev_weights[m][:,maxYPerScale[m]],5,N)*scales[m] for m in range(len(scales))]
        x_multiscale_grid, y_multiscale_grid=np.sum(decodedXPerScale), np.sum(decodedYPerScale)

        x_grid.append(x_multiscale_grid-start_x)
        y_grid.append(y_multiscale_grid-start_y)

        q[0],q[1]=q[0]+vel[i]*np.cos(q[2]), q[1]+vel[i]*np.sin(q[2])
        q[2]+=angVel[i]
        x_integ.append(q[0]-start_x)
        y_integ.append(q[1]-start_y)

        print(x_integ[-1], y_integ[-1])
        print(x_grid[-1], y_grid[-1])
        print('')


    plt.plot(x_integ, y_integ, 'g.')
    plt.plot(x_grid, y_grid, 'b.')

    plt.axis('equal')
    plt.title('Test Environment 2D space')
    plt.legend(('Path Integration', 'Multiscale Grid Decoding'))
    plt.show()
     


def hierarchicalNetwork2DGrid(prev_weights, net,N, vel, direction, iterations, wrap_iterations, x_grid_expect, y_grid_expect,scales):
    '''Select scale and initilise wrap storage'''
    delta = [(vel/scales[0]), (vel/scales[1]), (vel/scales[2]), (vel/scales[3]), (vel/scales[4])]
    cs_idx=scale_selection(vel,scales)
    wrap_rows=np.zeros((len(scales)))
    wrap_cols=np.zeros((len(scales)))

    '''Update selected scale'''
    del_x_cs, del_y_cs= delta[cs_idx]*np.cos(np.deg2rad(direction)), delta[cs_idx]*np.sin(np.deg2rad(direction))
    x_grid_expect[cs_idx]=(x_grid_expect[cs_idx]+(del_x_cs *scales[cs_idx]))%(N*scales[cs_idx])
    y_grid_expect[cs_idx]=(y_grid_expect[cs_idx]+(del_y_cs *scales[cs_idx]))%(N*scales[cs_idx])
    for i in range(iterations):
        prev_weights[cs_idx][:], wrap_rows_cs, wrap_cols_cs= net.update_weights_dynamics(prev_weights[cs_idx][:],direction, delta[cs_idx])
        prev_weights[cs_idx][prev_weights[cs_idx][:]<0]=0
        wrap_rows[cs_idx]+=wrap_rows_cs
        wrap_cols[cs_idx]+=wrap_cols_cs
    
    '''Update the 16 scale based on wraparound in 0.25 scale'''
    if (cs_idx==0 and (wrap_rows[cs_idx]!=0 or wrap_cols[cs_idx]!=0 )):
        num_links,excite,activity_mag,inhibit_scale, wrap_iterations=10,2,1.10262708e-01,7.51431074e-04,2
        net16=attractorNetwork2D(N,N,num_links,excite, activity_mag,inhibit_scale)
        del_rows_16, del_cols_16=(wrap_rows[cs_idx]*scales[cs_idx]*N)/scales[3], (wrap_cols[cs_idx]*scales[cs_idx]*N)/scales[3]  
        direction_16=np.rad2deg(math.atan2(del_rows_16, del_cols_16))
        distance_16=math.sqrt(del_cols_16**2 + del_rows_16**2)

        x_grid_expect[3]=(x_grid_expect[3]+(del_cols_16 *scales[3]))%(N*scales[3])
        y_grid_expect[3]=(y_grid_expect[3]+(del_rows_16 *scales[3]))%(N*scales[3])
        # wraparound[4]=(can.activityDecoding(prev_weights[4][:],4,N) + update_amount)//(N-1)
        for i in range(wrap_iterations):
            prev_weights[3][:], wrap_rows_16, wrap_cols_16= net16.update_weights_dynamics(prev_weights[3][:],direction_16, distance_16)
            prev_weights[3][prev_weights[3][:]<0]=0
            wrap_rows[3]+=wrap_rows_16
            wrap_cols[3]+=wrap_cols_16

    '''Update the 100 scale based on wraparound in any of the previous scales'''
    if (cs_idx!=0 and (wrap_rows[cs_idx]!=0 or wrap_cols[cs_idx]!=0 )): 
        # tunedParms=3,5,1,0.000865888565,1
        # net100=attractorNetwork2D(N,N,tunedParms[0],tunedParms[1], tunedParms[2],tunedParms[3])
        del_rows_100, del_cols_100=(wrap_rows[cs_idx]*scales[cs_idx]*N)/scales[4], (wrap_cols[cs_idx]*scales[cs_idx]*N)/scales[4]  
        direction_100=np.rad2deg(math.atan2(del_rows_100, del_cols_100))
        distance_100=math.sqrt(del_cols_100**2 + del_rows_100**2)

        x_grid_expect[4]=(x_grid_expect[4]+(del_cols_100 *scales[4]))%(N*scales[4])
        y_grid_expect[4]=(y_grid_expect[4]+(del_rows_100 *scales[4]))%(N*scales[4])
        # wraparound[4]=(can.activityDecoding(prev_weights[4][:],4,N) + update_amount)//(N-1)
        for i in range(wrap_iterations):
            prev_weights[4][:], wrap_rows_100, wrap_cols_100= net.update_weights_dynamics(prev_weights[4][:],direction_100, distance_100)
            prev_weights[4][prev_weights[4][:]<0]=0
            wrap_rows[4]+=wrap_rows_100
            wrap_cols[4]+=wrap_cols_100

    '''Update the 10000 scale based on wraparound in the 100 scale'''
    if (wrap_rows[4]!=0 or wrap_cols[4]!=0 ):
        # tunedParms=3,5,1,0.000865888565,1
        # net100=attractorNetwork2D(N,N,tunedParms[0],tunedParms[1], tunedParms[2],tunedParms[3])

        del_rows_10000, del_cols_10000=(wrap_rows[4]*scales[4]*N)/scales[5], (wrap_cols[4]*scales[4]*N)/scales[5]  
        direction_10000=np.rad2deg(math.atan2(del_rows_10000, del_cols_10000))
        distance_10000=math.sqrt(del_cols_10000**2 + del_rows_10000**2)

        x_grid_expect[5]=(x_grid_expect[5]+(del_cols_10000 *scales[5]))%(N*scales[5])
        y_grid_expect[5]=(y_grid_expect[5]+(del_rows_10000 *scales[5]))%(N*scales[5])

        for i in range(wrap_iterations):
            prev_weights[-1][:], wrap_rows[-1], wrap_cols[-1]= net.update_weights_dynamics(prev_weights[-1][:],direction_10000, distance_10000)
            prev_weights[-1][prev_weights[-1][:]<0]=0
        
    wrap=0
    if np.any(wrap_cols!=0):
        wrap=1
        print(f"------------------------------------------------------------------------------------------------------wrap_cols {wrap_cols}")
    if np.any(wrap_rows!=0):
        wrap=1
        print(f"------------------------------------------------------------------------------------------------------wrap_rows {wrap_rows}")

       
    return prev_weights, wrap, x_grid_expect, y_grid_expect

def headDirectionAndPlace(index, outfile, plot=False, N=100):
    global theata_called_iters,theta_weights, prev_weights, q, wrap_counter, current_i, x_grid_expect, y_grid_expect 

    # num_links,excite,activity_mag,inhibit_scale, iterations, wrap_iterations=7,8,5.47157578e-01 ,3.62745653e-04, 2, 2 #good only at small scale
    # num_links,excite,activity_mag,inhibit_scale, iterations, wrap_iterations=72,1,9.05078199e-01,7.85317908e-04,4,1
    # num_links,excite,activity_mag,inhibit_scale, iterations, wrap_iterations=6,1,3.89338335e-01,1.60376324e-04, 3,3  #improved at larger scale 

    # num_links,excite,activity_mag,inhibit_scale, iterations, wrap_iterations=7,1,2.59532708e-01 ,2.84252467e-04,4,3 #without decimals 1000 iters fitness -5000
    # num_links,excite,activity_mag,inhibit_scale, iterations, wrap_iterations=10,2,1.10262708e-01,6.51431074e-04,3,2 #with decimals 200 iters fitness -395
    # num_links,excite,activity_mag,inhibit_scale, iterations, wrap_iterations=10,2,1.10262708e-01,7.51431074e-04,2,2 #with decimals 200 iters fitness -395 modified
    
    num_links,excite,activity_mag,inhibit_scale, iterations, wrap_iterations=5,10,7.59471889e-01,3.93846361e-04,1,1 #tuned to reduce error 
    
    network=attractorNetwork2D(N,N,num_links,excite, activity_mag,inhibit_scale)

    '''__________________________Storage and initilisation parameters______________________________'''
    scales=[0.25,1,4,16,100,10000]
    theta_weights=np.zeros(360)
    theata_called_iters=0
    start_x, start_y=(50*scales[3])+(50*scales[4])+(50*scales[5]),(50*scales[3])+(50*scales[4])+(50*scales[5])
    wrap_counter=[0,0,0,0,0,0]
    x_grid, y_grid=[], []
    x_grid_expect, y_grid_expect =[0,0,0,50*scales[3],50*scales[4],50*scales[5]],[0,0,0,50*scales[3],50*scales[4],50*scales[5]]
    x_integ, y_integ=[],[]
    q=[start_x,start_y,0]
    x_integ_err, y_integ_err=[],[]
    q_err=[start_x,start_y,0]

    '''__________________________Initilising scales in the center and at the edge_____________________________'''
    prev_weights=[np.zeros((N,N)),np.zeros((N,N)),np.zeros((N,N)),np.zeros((N,N)),np.zeros((N,N)), np.zeros((N,N))]
    for n in range(3):
        for m in range(iterations):
            prev_weights[n]=network.excitations(0,0)
            prev_weights[n]=network.update_weights_dynamics_row_col(prev_weights[n][:], 0, 0)
            prev_weights[n][prev_weights[n][:]<0]=0
    
    for start_idx in range(3,6):
        prev_weights[start_idx]=network.excitations(50,50)
        prev_weights[start_idx][:]= network.update_weights_dynamics_row_col(prev_weights[start_idx][:],0,0)
        prev_weights[start_idx][prev_weights[start_idx][:]<0]=0


    '''_______________________________Iterating through simulation velocities_______________________________'''
    
    for i in range(test_length):   
    # fig, axs = plt.subplots(1,1,figsize=(5, 5)) 
    # def animate(i):
    #     global theta_weights, prev_weights, q, wrap_counter, current_i, x_grid_expect, y_grid_expect
        
        '''Path integration'''
        q[2]+=angVel[i]
        q[0],q[1]=q[0]+vel[i]*np.cos(q[2]), q[1]+vel[i]*np.sin(q[2])
        x_integ.append(q[0]-start_x)
        y_integ.append(q[1]-start_y)

        '''Mutliscale CAN update'''
        noise = np.random.uniform(0,1,1)
        vel[i]=np.float64(vel[i]+noise)
        N_dir=360
        theta_weights=headDirection(theta_weights, np.rad2deg(angVel[i]), 0)
        direction=activityDecodingAngle(theta_weights,5,N_dir)
        prev_weights, wrap, x_grid_expect, y_grid_expect= hierarchicalNetwork2DGrid(prev_weights, network, N, vel[i], direction, iterations,wrap_iterations, x_grid_expect, y_grid_expect, scales)

        '''1D method for decoding'''
        maxXPerScale, maxYPerScale = np.array([np.argmax(np.max(prev_weights[m], axis=1)) for m in range(len(scales))]), np.array([np.argmax(np.max(prev_weights[m], axis=0)) for m in range(len(scales))])
        decodedXPerScale=[activityDecoding(prev_weights[m][maxXPerScale[m], :],5,N)*scales[m] for m in range(len(scales))]
        decodedYPerScale=[activityDecoding(prev_weights[m][:,maxYPerScale[m]],5,N)*scales[m] for m in range(len(scales))]
        # x_multiscale_grid, y_multiscale_grid=np.sum(decodedXPerScale), np.sum(decodedYPerScale)
        x_multiscale_grid, y_multiscale_grid=np.sum(decodedXPerScale[0:3]+x_grid_expect[3:6]), np.sum(decodedYPerScale[0:3]+y_grid_expect[3:6])
        x_grid.append(x_multiscale_grid-start_x)
        y_grid.append(y_multiscale_grid-start_y)

        '''Error integrated path'''
        q_err[2]+=angVel[i]
        q_err[0],q_err[1]=q_err[0]+vel[i]*np.cos(q_err[2]), q_err[1]+vel[i]*np.sin(q_err[2])
        x_integ_err.append(q_err[0]-start_x)
        y_integ_err.append(q_err[1]-start_y)



        print(np.rad2deg(q[2]), direction)
        print(f'decoded: {decodedXPerScale}, {decodedYPerScale}')
        print(f'expected: {x_grid_expect}, {y_grid_expect}')
        print(f'integ: {x_integ[-1]}, {y_integ[-1]}')
        print(f'CAN: {x_grid[-1]}, {y_grid[-1]}')
        print('')

        # for k in range(nrows-1):
        #     axs[0][k].imshow(prev_weights[k][:][:])#(np.arange(N),prev_weights[k][:],color=colors[k])
        #     axs[0][k].spines[['top', 'left', 'right']].set_visible(False)
        #     axs[0][k].invert_yaxis()

        # axs[0][nrows-1].clear()
        # axs[1][nrows-1].axis('off')
        # axs[0][nrows-1].set_xlim([-0.2*12,0.2*12])
        # axs[0][nrows-1].set_ylim([-0.2*12,0.2*12])

        # for j in range(len(hD_x)):
        #     axs[0][nrows-1].arrow(0,0,hD_x[j]*10,hD_y[j]*10, color='m')

        # for l in range(nrows-1): 
        #     axs[1][l].clear()
        #     # axs[1][l].set_title(f"Scale{l}")
        #     axs[1][l].text(0,1,f"Decode: {np.round(decodedXPerScale[l],2)},{np.round(decodedYPerScale[l],2)}")
        #     axs[1][l].text(0,0.5,f"Expect: {np.round(x_grid_expect[l],2)},{np.round(y_grid_expect[l],2)}")
        #     axs[1][l].axis('off')

    
    # ani = FuncAnimation(fig, animate, interval=1,frames=400,repeat=False)
    # plt.show()

    # f = "./results/GIFs/BerlinPathMultiscaleAttractor100ScaleTestingWRappingErrors.gif" 
    # writergif = animation.PillowWriter(fps=25) 
    # ani.save(f, writer=writergif)

    np.save(outfile, np.array([x_grid, y_grid, x_integ, y_integ]))
    x_error=np.sum(np.abs(np.array(x_grid) - np.array(x_integ)))
    y_error=np.sum(np.abs(np.array(y_grid) - np.array(y_integ)))

    x_error_integ=np.sum(np.abs(np.array(x_integ_err) - np.array(x_integ)))
    y_error_integ=np.sum(np.abs(np.array(y_integ_err) - np.array(y_integ)))

    print(f'Integrated error: {(x_error_integ+y_error_integ)*-1}')
    print(f'CAN error: {(x_error+y_error)*-1}')

    if plot ==True:
        
        plt.plot(x_integ, y_integ, 'g.')
        plt.plot(x_integ_err, y_integ_err, 'r.')
        plt.plot(x_grid, y_grid, 'b.')
        plt.axis('equal')
        plt.title('Test Environment 2D space')
        plt.legend(('Path Integration without Error', 'Path Integration with Error', 'Multiscale Grid Decoding'))
        plt.show()



def hierarchicalNetwork2DGridNowrapNet(prev_weights, net,N, vel, direction, iterations, wrap_iterations, x_grid_expect, y_grid_expect,scales):
    '''Select scale and initilise wrap storage'''
    delta = [(vel/scales[i]) for i in range(len(scales))]
    cs_idx=scale_selection(vel,scales)
    # print(vel, scales, cs_idx)
    wrap_rows=np.zeros((len(scales)))
    wrap_cols=np.zeros((len(scales)))

    '''Update selected scale'''

    for i in range(iterations):
        prev_weights[cs_idx][:], wrap_rows_cs, wrap_cols_cs= net.update_weights_dynamics(prev_weights[cs_idx][:],direction, delta[cs_idx])
        prev_weights[cs_idx][prev_weights[cs_idx][:]<0]=0
        # wrap_rows[cs_idx]+=wrap_rows_cs
        # wrap_cols[cs_idx]+=wrap_cols_cs
        x_grid_expect+=wrap_cols_cs*N*scales[cs_idx]
        y_grid_expect+=wrap_rows_cs*N*scales[cs_idx]
    
    # for i in range(len(wrap_cols)):
    #     x_grid_expect+=wrap_cols[i]*N*scales[i]
    # for i in range(len(wrap_rows)):
    #     y_grid_expect+=wrap_rows[i]*N*scales[i]

    
        if np.any(wrap_cols_cs!=0):
            print(f"------------------------------------------------------------------------------------------------------wrap_cols {wrap_cols_cs}, {scales[cs_idx]}")
        if np.any(wrap_rows_cs!=0):
            print(f"------------------------------------------------------------------------------------------------------wrap_rows {wrap_rows_cs}, {scales[cs_idx]}")

    wrap=0   
    return prev_weights, wrap, x_grid_expect, y_grid_expect

def headDirectionAndPlaceNoWrapNet(scales, test_length, vel, angVel,savePath, plot=False, printing=True, N=100):
    global theata_called_iters,theta_weights, prev_weights, q, wrap_counter, current_i, x_grid_expect, y_grid_expect 

    # num_links,excite,activity_mag,inhibit_scale, iterations, wrap_iterations=7,8,5.47157578e-01 ,3.62745653e-04, 2, 2 #good only at small scale
    # num_links,excite,activity_mag,inhibit_scale, iterations, wrap_iterations=72,1,9.05078199e-01,7.85317908e-04,4,1
    # num_links,excite,activity_mag,inhibit_scale, iterations, wrap_iterations=6,1,3.89338335e-01,1.60376324e-04, 3,3  #improved at larger scale 

    # num_links,excite,activity_mag,inhibit_scale, iterations, wrap_iterations=7,1,2.59532708e-01 ,2.84252467e-04,4,3 #without decimals 1000 iters fitness -5000
    # num_links,excite,activity_mag,inhibit_scale, iterations, wrap_iterations=10,2,1.10262708e-01,6.51431074e-04,3,2 #with decimals 200 iters fitness -395
    # num_links,excite,activity_mag,inhibit_scale, iterations, wrap_iterations=10,2,1.10262708e-01,6.51431074e-04,3,2 #with decimals 200 iters fitness -395 modified
    
    num_links,excite,activity_mag,inhibit_scale, iterations, wrap_iterations=10,10,1,0.0008,2,1 #with decimals 200 iters fitness -395 modified
    # num_links,excite,activity_mag,inhibit_scale, iterations, wrap_iterations=5,7,9.59471889e-01,2.93846361e-04,1,1 #tuned to reduce error 
    # num_links,excite,activity_mag,inhibit_scale, iterations, wrap_iterations=3,5,0.015,0.000865888565,1,1 #0.25 scale input, np.random.uniform(0,1,1) error
    # num_links,excite,activity_mag,inhibit_scale, iterations, wrap_iterations=3,5,0.05,0.000565888565,1,1 #16 scale input, np.random.uniform(10,20,1) error
    # num_links,excite,activity_mag,inhibit_scale, iterations, wrap_iterations=3,5,0.04,0.000965888565,1,1 #1 scale input, np.random.uniform(1,2,1) error
    # num_links,excite,activity_mag,inhibit_scale, iterations, wrap_iterations=3,5,0.05,0.000665888565,1,1 #4 scale input, np.random.uniform(2,10,1) error
    network=attractorNetwork2D(N,N,num_links,excite, activity_mag,inhibit_scale)

    

    '''__________________________Storage and initilisation parameters______________________________'''
    # scales=[0.25,1,4,16]
    theta_weights=np.zeros(360)
    theata_called_iters=0
    # start_x, start_y=(50*scales[3])+(50*scales[4])+(50*scales[5]),(50*scales[3])+(50*scales[4])+(50*scales[5])
    wrap_counter=[0,0,0,0,0,0]
    x_grid, y_grid=[], []
    x_grid_expect, y_grid_expect =0,0
    x_integ, y_integ=[],[]
    q=[0,0,0]
    x_integ_err, y_integ_err=[],[]
    q_err=[0,0,0]

    '''__________________________Initilising scales in the center and at the edge_____________________________'''
    prev_weights=[np.zeros((N,N)) for _ in range(len(scales))]
    for n in range(len(scales)):
        for m in range(iterations):
            prev_weights[n]=network.excitations(0,0)
            prev_weights[n]=network.update_weights_dynamics_row_col(prev_weights[n][:], 0, 0)
            prev_weights[n][prev_weights[n][:]<0]=0
    

    '''_______________________________Iterating through simulation velocities_______________________________'''
    for i in range(1,test_length):   
        '''Path integration'''
        q[2]+=angVel[i]
        q[0],q[1]=q[0]+vel[i]*np.cos(q[2]), q[1]+vel[i]*np.sin(q[2])
        x_integ.append(q[0])
        y_integ.append(q[1])

        '''Dynamic network tuning'''
        # swap_val=5
        # if vel[i]<=scales[0]*swap_val:
        #     num_links,excite,activity_mag,inhibit_scale, iterations, wrap_iterations=3,5,0.015,0.000865888565,1,1 #0.25 scale input, np.random.uniform(0,1,1) error
        #     network=attractorNetwork2D(N,N,num_links,excite, activity_mag,inhibit_scale)
        # elif vel[i]>scales[0]*swap_val and vel[i]<=scales[1]*swap_val:
        #     num_links,excite,activity_mag,inhibit_scale, iterations, wrap_iterations=3,5,0.04,0.000865888565,1,1 #1 scale input, np.random.uniform(1,2,1) error
        #     network=attractorNetwork2D(N,N,num_links,excite, activity_mag,inhibit_scale)
        # elif vel[i]>scales[1]*swap_val and vel[i]<=scales[2]*swap_val:
        #     num_links,excite,activity_mag,inhibit_scale, iterations, wrap_iterations=3,5,0.05,0.000665888565,1,1 #4 scale input, np.random.uniform(2,10,1) error
        #     network=attractorNetwork2D(N,N,num_links,excite, activity_mag,inhibit_scale)
        # elif vel[i]>scales[2]*swap_val:
        #     num_links,excite,activity_mag,inhibit_scale, iterations, wrap_iterations=3,5,0.05,0.000565888565,1,1 #16 scale input, np.random.uniform(10,20,1) error
        #     network=attractorNetwork2D(N,N,num_links,excite, activity_mag,inhibit_scale)   

        '''Mutliscale CAN update'''
        N_dir=360
        theta_weights=headDirection(theta_weights, np.rad2deg(angVel[i]), 0)
        direction=activityDecodingAngle(theta_weights,5,N_dir)
        prev_weights, wrap, x_grid_expect, y_grid_expect= hierarchicalNetwork2DGridNowrapNet(prev_weights, network, N, vel[i], direction, iterations,wrap_iterations, x_grid_expect, y_grid_expect, scales)

        '''1D method for decoding'''
        maxXPerScale, maxYPerScale = np.array([np.argmax(np.max(prev_weights[m], axis=1)) for m in range(len(scales))]), np.array([np.argmax(np.max(prev_weights[m], axis=0)) for m in range(len(scales))])
        decodedXPerScale=[activityDecoding(prev_weights[m][maxXPerScale[m], :],5,N)*scales[m] for m in range(len(scales))]
        decodedYPerScale=[activityDecoding(prev_weights[m][:,maxYPerScale[m]],5,N)*scales[m] for m in range(len(scales))]
        x_multiscale_grid, y_multiscale_grid=np.sum(decodedXPerScale), np.sum(decodedYPerScale)
        # x_multiscale_grid, y_multiscale_grid=np.sum(decodedXPerScale[0:3]+x_grid_expect[3:6]), np.sum(decodedYPerScale[0:3]+y_grid_expect[3:6])
        x_grid.append(x_multiscale_grid+x_grid_expect)
        y_grid.append(y_multiscale_grid+y_grid_expect)

        '''Error integrated path'''
        q_err[2]+=angVel[i]
        q_err[0],q_err[1]=q_err[0]+vel[i]*np.cos(np.deg2rad(direction)), q_err[1]+vel[i]*np.sin(np.deg2rad(direction))
        x_integ_err.append(q_err[0])
        y_integ_err.append(q_err[1])

        if printing==True:
            print(f'dir: {np.rad2deg(q[2])}, {direction}')
            print(f'vel: {vel[i]}')
            print(f'decoded: {decodedXPerScale}, {decodedYPerScale}')
            print(f'expected: {x_grid_expect}, {y_grid_expect}')
            print(f'integ: {x_integ[-1]}, {y_integ[-1]}')
            print(f'CAN: {x_grid[-1]}, {y_grid[-1]}')
            print('')

    if savePath != None:
        np.save(savePath, np.array([x_grid, y_grid, x_integ, y_integ, x_integ_err, y_integ_err]))
    
    print(f'CAN error: {errorTwoCoordinateLists(x_integ,y_integ, x_grid, y_grid)}')

    if plot ==True:    
        plt.plot(x_integ, y_integ, 'g.')
        # plt.plot(x_integ_err, y_integ_err, 'y.')
        plt.plot(x_grid, y_grid, 'b.')
        plt.axis('equal')
        plt.title('Test Environment 2D space')
        plt.legend(('Path Integration without Error','Multiscale Grid Decoding'))
        plt.show()
    else:
        return x_grid, y_grid



def plotFromSavedArray(outfile,savePath):
    x_grid,y_grid, x_integ, y_integ, x_integ_err, y_integ_err= np.load(outfile)
    '''Compute distance'''
    dist=np.sum(vel)
   
    '''Compute error'''
    x_error=np.sum(np.abs(np.array(x_grid) - np.array(x_integ)))
    y_error=np.sum(np.abs(np.array(y_grid) - np.array(y_integ)))
    error=((x_error+y_error)*-1)/len(x_grid)

    '''Plot'''
    fig, axs = plt.subplots(1,1,figsize=(8, 8))
    plt.title('Kitti Dataset Trajectory Tracking')
    plt.plot(x_integ, y_integ, 'g--')
    plt.plot(x_grid, y_grid, 'm.')
    # plt.plot(wrap_x, wrap_y,'r*')
    plt.axis('equal')
    # plt.title(f'Distance:{round(dist)}m   Error:{round(error)}m/iter   Iterations:{len(x_integ)}')
    plt.legend(('Path Integration', 'Multiscale CAN'))
    plt.savefig(savePath)

def plotSavedMultiplePaths():
    fig, axs = plt.subplots(6,3,figsize=(10, 8))
    fig.legend(['MultiscaleCAN', 'Grid'])
    fig.tight_layout(pad=2.0)
    fig.suptitle('Tracking Simulated Trajectories through Berlin')
    # handles, labels = axs.get_legend_handles_labels()
    # fig.legend(handles, labels, loc='upper center')
    axs=axs.ravel()
    for i in range(18):
        '''distance'''
        outfile=f'./results/TestEnvironmentFiles/TraverseInfo/BerlineEnvPath{i}.npz'
        traverseInfo=np.load(outfile, allow_pickle=True)
        dist=np.sum(traverseInfo['speeds'])


        '''error'''
        # outfile=f'./results/TestEnvironmentFiles/MultiscaleCAN/TestMultiscalePath{i}.npy'         
        # outfile=f'./results/TestEnvironmentFiles/MultiscaleCAN/TestMultiscalePathChangedFeedthrough_{i}.npy'
        # outfile=f'./results/TestEnvironmentFiles/MultiscaleCAN/TestMultiscalePathTuned_{i}.npy'
        # outfile=f'./results/TestEnvironmentFiles/MultiscaleCAN/TestMultiscalePathLonger_{i}.npy'
        # outfile=f'./results/TestEnvironmentFiles/MultiscaleCAN/TestMultiscalePathwithUniformErr_{i}.npy'
        outfile=f'./results/TestEnvironmentFiles/MultiscaleCAN/TestMultiscalePathwithSpeeds0to20_{i}.npy'
        x_grid,y_grid,x_integ, y_integ, x_integ_err, y_integ_err = np.load(outfile)
        # print(np.load(outfile)[0])


        x_error=np.sum(np.abs(np.array(x_grid) - np.array(x_integ)))
        y_error=np.sum(np.abs(np.array(y_grid) - np.array(y_integ)))
        errorCAN=((x_error+y_error)*-1)

        x_error_integ=np.sum(np.abs(np.array(x_integ_err) - np.array(x_integ)))
        y_error_integ=np.sum(np.abs(np.array(y_integ_err) - np.array(y_integ)))
        errorPathIntegration=((x_error_integ+y_error_integ)*-1)


        '''plot'''
        l1=axs[i].plot(x_grid,y_grid, 'm-',label='MultiscaleCAN')
        l2=axs[i].plot(x_integ, y_integ, 'g--', label='Naiive Integration')
        # axs[i].plot(x_integ_err, y_integ_err, 'r.')
        axs[i].axis('equal')
        # axs[i].set_title(f'CAN Err:{round(errorCAN)}m   Integ Err:{round(errorPathIntegration)}')
        # axs[i].legend(['MultiscaleCAN', 'Naiive Integration'])
    plt.subplots_adjust(top=0.93)
    fig.legend([l1, l2], labels=['Multiscale CAN', 'Naiive Integration'],loc="upper right")
    plt.savefig('./results/TestEnvironmentFiles/MultipathTrackingSpeeds0to20.png')


# particle_pos=particle_filter( vel[1:test_length], angVel[1:test_length])
# particle_x,particle_y=zip(*particle_pos)
# x_integ, y_integ=pathIntegration(vel[1:test_length],angVel[1:test_length])
# ekf_pos=EKF_main(vel[1:test_length], angVel[1:test_length])
# ekf_x,ekf_y=zip(*ekf_pos)


# plt.plot(x_integ, y_integ,'g.-')
# plt.plot(ekf_x,ekf_y, 'm-.')
# # plt.plot(particle_x,particle_y,'b.-')
# plt.axis('equal')
# plt.legend(('Path Integration', 'EKF', 'Particle Filter'))
# plt.show()



# kinemVelFile='./results/TestEnvironmentFiles/TraverseInfo/testEnvPathVelocities2.npy'
# kinemAngVelFile='./results/TestEnvironmentFiles/TraverseInfo/testEnvPathAngVelocities2.npy'
# vel,angVel=np.load(kinemVelFile), np.load(kinemAngVelFile)
# vel=np.concatenate([np.linspace(0,scales[0]*5,test_length//5), np.linspace(scales[0]*5,scales[1]*5,test_length//5), np.linspace(scales[1]*5,scales[2]*5,test_length//5), np.linspace(scales[2]*5,scales[3]*5,test_length//5), np.linspace(scales[3]*5,scales[4]*5,test_length//5)])

'''Running 18 paths with Multiscale CAN'''
# for index in range(18):
#     outfile=f'./results/TestEnvironmentFiles/TraverseInfo/BerlineEnvPath{index}.npz'
#     traverseInfo=np.load(outfile, allow_pickle=True)
#     vel,angVel,truePos, startPose=traverseInfo['speeds'], traverseInfo['angVel'], traverseInfo['truePos'], traverseInfo['startPose']

#     scales=[0.25,1,4,16]
#     if len(vel)<500:
#         test_length=len(vel)
#     else:
#         test_length=500

#     # iterPerScale=int(np.ceil(test_length/4))
#     # vel=np.concatenate([np.linspace(0,scales[0]*5,iterPerScale), np.linspace(scales[0]*5,scales[1]*5,iterPerScale), np.linspace(scales[1]*5,scales[2]*5,iterPerScale), np.linspace(scales[2]*5,scales[3]*5,iterPerScale)])
#     vel=np.random.uniform(0,20,test_length) 
#     headDirectionAndPlaceNoWrapNet(scales, test_length, vel, angVel,f'./results/TestEnvironmentFiles/MultiscaleCAN/TestMultiscalePathwithSpeeds0to20_{index}.npy', printing=False)
# plotSavedMultiplePaths()

'''Running single path'''

# scales=[0.25,0.5,1,2,4,8,16]
# scales=[0.25,1,4,16]
# scales=[1]

# vel=[0.5]*test_length
# iterPerScale=int(np.ceil(test_length/4))
# vel=np.concatenate([np.linspace(0,scales[0]*5,iterPerScale), np.linspace(scales[0]*5,scales[1]*5,iterPerScale), np.linspace(scales[1]*5,scales[2]*5,iterPerScale), np.linspace(scales[2]*5,scales[3]*5,iterPerScale)])
# vel=np.linspace(scales[1]*5,scales[2]*5,test_length)
# headDirectionAndPlaceMultiparameter()
# headDirectionAndPlace(index)
# plotFromSavedArray(f'./results/TestEnvironmentFiles/MultiscaleCAN/TestMultiscalePathwithUniformErr_{index}.npy')
# plotFromSavedArray(f'./results/TestEnvironmentFiles/MultiscaleCAN/TestMultiscalePathTesting{index}.npy')



''' Benefits of Multiscale '''
def mutliVs_single(filepath, index, desiredTestLength):
    outfile=f'./results/TestEnvironmentFiles/TraverseInfo/BerlineEnvPath{index}.npz'
    traverseInfo=np.load(outfile, allow_pickle=True)
    angVel= traverseInfo['angVel']
  
    if len(angVel)<desiredTestLength:
        test_length=len(angVel)
    else:
        test_length=desiredTestLength

    errors=[]
    for i in range(1,21):
        vel=np.random.uniform(0,i,test_length)
        true_x,true_y=pathIntegration(vel,angVel)

        scales=[1]
        single_x,single_y=headDirectionAndPlaceNoWrapNet(scales,test_length, vel, angVel,None,plot=False, printing=False)
        singleError=errorTwoCoordinateLists(true_x,true_y, single_x, single_y)

        scales=[0.25,1,4,16]
        multi_x,multi_y=headDirectionAndPlaceNoWrapNet(scales, test_length, vel, angVel,None,plot=False, printing=False)
        multipleError=errorTwoCoordinateLists(true_x,true_y, multi_x, multi_y)

        errors.append([singleError,multipleError])

    np.save(filepath, errors)

# index=0
# filepath=f'./results/TestEnvironmentFiles/MultiscaleVersus SingleScale/Path{index}_singleVSmultiErrors2.npy'
# mutliVs_single(filepath, index, 500)


# plt.figure()
# singleErrors, multipleErrors = zip(*np.load(filepath))
# plt.plot(singleErrors, 'b')
# plt.plot(multipleErrors, 'm')
# plt.legend(['Single Network Error', 'Multiscale Networks Error'])
# plt.xlabel('Upper Limit of Velocity Range [m/s]')
# plt.ylabel('Error [Sum of Absolute Differences]')
# plt.title('Comparison of Single versus Multiscale Networks', y=1.08)
# plt.tight_layout()
# plt.show()



''' Kitti Odometry'''
def data_processing():
    poses = pd.read_csv('./data/dataset/poses/00.txt', delimiter=' ', header=None)
    gt = np.zeros((len(poses), 3, 4))
    for i in range(len(poses)):
        gt[i] = np.array(poses.iloc[i]).reshape((3, 4))
    return gt

def testing_Conversion(sparse_gt):
    # length=len(sparse_gt[:,:,3][:,0])
    data_x=sparse_gt[:, :, 3][:,0]#[:200]
    data_y=sparse_gt[:, :, 3][:,2]#[:200]
    delta1,delta2=[],[]
    for i in range(1,len(data_x)):
        x0=data_x[i-2]
        x1=data_x[i-1]
        x2=data_x[i]
        y0=data_y[i-2]
        y1=data_y[i-1]
        y2=data_y[i]

        delta1.append(np.sqrt(((x2-x1)**2)+((y2-y1)**2))) #translation
        delta2.append((math.atan2(y2-y1,x2-x1)) - (math.atan2(y1-y0,x1-x0)))  

    print(np.array(delta1).shape,np.array(delta2).shape)

    # print(dirs)
    # delta2=np.transpose(np.diff(dirs))
    # print(delta1,delta2)
    np.save('./results/TestEnvironmentFiles/kittiVels.npy', np.array([delta1,delta2]))

sparse_gt=data_processing()#[0::4]
testing_Conversion(sparse_gt)
# scales=[1,2,4,8,16]
scales=[0.25,1,4,16]
scales=[1]
vel,angVel=np.load('./results/TestEnvironmentFiles/kittiVels.npy')
if len(vel)<500:
    test_length=len(vel)
else:
    test_length=500

test_length=len(vel)
# headDirectionAndPlaceNoWrapNet(scales, test_length, vel, angVel,f'./results/TestEnvironmentFiles/kittiPath_nosparse_singleScale.npy', printing=False)
plotFromSavedArray(f'./results/TestEnvironmentFiles/kittiPath_nosparse_singleScale.npy','./results/TestEnvironmentFiles/KittiPath9.png')

# def angdiff( th1, th2):
#     d = th1 - th2
#     d = np.mod(d+np.pi, 2*np.pi) - np.pi
#     return d

# def testing_Conversion_kitti(sparse_gt):
#     length=len(sparse_gt[:,:,3][:,0])
#     curr_x, curr_y, x, y=np.zeros(length),np.zeros(length),np.zeros(length),np.zeros(length)
#     curr_x, curr_y, x[0], y[0]= [0],[0],0,0
#     curr_theta=np.deg2rad(90)
#     data_x=sparse_gt[:, :, 3][:,0]#[:200]
#     data_y=sparse_gt[:, :, 3][:,2]#[:200]
#     for i in range(2,len(data_x)):
#         x0=data_x[i-2]
#         x1=data_x[i-1]
#         x2=data_x[i]
#         y0=data_y[i-2]
#         y1=data_y[i-1]
#         y2=data_y[i]

#         # delta1=np.sqrt(((x2-x1)**2)+((y2-y1)**2)) #translation
#         # # if (x2-x1)==0:
#         # #     assert False 
#         # #     # delta2=np.pi/2
#         # # else:
#         # angle2=(math.atan2(y2-y1,x2-x1)) #angle
#         # angle1=(math.atan2(y1-y0,x1-x0)) 

#         delta1=np.sqrt(((x2-x1)**2)+((y2-y1)**2)) #translation
#         delta2=((math.atan2(y2-y1,x2-x1)) - (math.atan2(y1-y0,x1-x0)))  

        
#         curr_x.append(curr_x[-1]+(delta1*np.cos(curr_theta)))
#         curr_y.append(curr_y[-1]+(delta1*np.sin(curr_theta)))
#         curr_theta+=delta2

#         x[i]=x[i-1]+(x2-x1)
#         y[i]=y[i-1]+(y2-y1)

#         print(delta1, curr_theta)

#     fig = plt.figure(figsize=(13, 4))
#     ax0 = fig.add_subplot(1, 2, 1)
#     ax1 = fig.add_subplot(1, 2, 2)

#     ax1.set_title('Converted')
#     ax1.scatter(curr_x, curr_y,c='b',s=15)
#     # ax1.set_xlim([-300,300])
#     # ax1.set_ylim([-100,500])

#     ax0.set_title('Original')
#     ax0.scatter(x, y,c='b',s=15)
#     # ax0.set_xlim([-300,300])
#     # ax0.set_ylim([-100,500])

    
#     plt.savefig('./results/TestEnvironmentFiles/testingKitti')

# testing_Conversion_kitti(data_processing())
