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
prev_weights_trans=[np.zeros(N[0]), np.zeros(N[0]), np.zeros(N[0]), np.zeros(N[0]),np.zeros(N[0])]
prev_weights_rot=[np.zeros(N[1]), np.zeros(N[1]), np.zeros(N[1]),np.zeros(N[1]), np.zeros(N[1])]
split_output=[0,0,0]
num_links=[4,17]
excite=[5,7]
activity_mag=[1,1]
inhibit_scale=[0.05,0.005]
curr_parameter=[0,0,0]
# curr_x,curr_y=0,0
x,y=0,0
SCALING_FACTOR=50
SCALING_FACTOR2=360


def angdiff( th1, th2):
    d = th1 - th2
    d = np.mod(d+np.pi, 2*np.pi) - np.pi
    return d

def data_processing():
    poses = pd.read_csv('./data/dataset/poses/00.txt', delimiter=' ', header=None)
    gt = np.zeros((len(poses), 3, 4))
    for i in range(len(poses)):
        gt[i] = np.array(poses.iloc[i]).reshape((3, 4))
    return gt

#this function works
def testing_Conversion(sparse_gt):
    length=len(sparse_gt[:,:,3][:,0])
    curr_x, curr_y, x, y=np.zeros(length),np.zeros(length),np.zeros(length),np.zeros(length)
    curr_x[0], curr_y[0], x[0], y[0]= 0,0,0,0
    for i in range(1,length):
        x1=sparse_gt[:,:,3][i-1,0]
        y1=sparse_gt[:,:,3][i-1,2]

        x2=sparse_gt[:,:,3][i,0]
        y2=sparse_gt[:,:,3][i,2]

        delta1=np.sqrt(((x2-x1)**2)+((y2-y1)**2)) #translation
        if (x2-x1)==0:
            assert False 
            # delta2=np.pi/2
        else:
            delta2=(math.atan2(y2-y1,x2-x1)) #angle

        curr_x[i]=curr_x[i-1] + (delta1*np.cos(delta2))
        curr_y[i]=curr_y[i-1] + (delta1*np.sin(delta2))

        x[i]=x[i-1]+(x2-x1)
        y[i]=y[i-1]+(y2-y1)

        print(delta1, delta2)

    fig = plt.figure(figsize=(13, 4))
    ax0 = fig.add_subplot(1, 2, 1)
    ax1 = fig.add_subplot(1, 2, 2)

    ax1.set_title('Converted')
    ax1.scatter(curr_x, curr_y,c='b',s=15)
    ax1.set_xlim([-300,300])
    ax1.set_ylim([-100,500])

    ax0.set_title('Original')
    ax0.scatter(x, y,c='b',s=15)
    ax0.set_xlim([-300,300])
    ax0.set_ylim([-100,500])

    
    plt.show()

# this fucntion runs but decoding is inaccurate due to wrong angular velocity decoding (angle curr_parameter [2] initliasied at 90)
def visualise(data_x,data_y):
    ''' 2 x 1D attractor networks representing change in translation and roation while seting on the 
    delta value which is decoded and used to update the exstimated x y position. The animation displays
    one iteration at a time'''

    fig = plt.figure(figsize=(13, 4))
    ax0 = fig.add_subplot(1, 3, 1)
    ax1 = fig.add_subplot(1, 3, 2)
    ax2 = fig.add_subplot(1, 3, 3)

    '''Initalise network'''            
    delta=[0,0]
    curr_parameter[2]=90
    for i in range(len(delta)):
        net=attractorNetwork(N[i],num_links[i],excite[i], activity_mag[i],inhibit_scale[i])
        prev_weights[i][net.activation(delta[i])]=net.full_weights(num_links[i])
        prev_weights[i][prev_weights[i][:]<0]=0

    
    def animate(i):
        ax0.set_title("Ground Truth Pose")
        ax0.scatter(data_x[i],data_y[i],s=15)
        ax0.axis('equal')
        # ax0.set_xlim([-300,300])
        # ax0.set_ylim([-100,500])
        # ax0.set_zlim([-50,50])
        # ax0.invert_yaxis()
        # ax0.view_init(elev=39, azim=140)
        
        global prev_weights, num_links, excite, activity_mag,inhibit_scale, curr_parameter
        ax1.clear()
        if i>=2:
            '''encoding mangnitude and direction of movement'''
            x0=data_x[i-2]
            x1=data_x[i-1]
            x2=data_x[i]
            y0=data_y[i-2]
            y1=data_y[i-1]
            y2=data_y[i]
            
            delta[0]=np.sqrt(((x2-x1)**2)+((y2-y1)**2)) *SCALING_FACTOR#translation
            delta[1]=((np.rad2deg(math.atan2(y2-y1,x2-x1)) - np.rad2deg(math.atan2(y1-y0,x1-x0))))         #angle
       
            '''updating network'''
            for k in range(10): 
                net=attractorNetworkSettling(N[0],num_links[0],excite[0], activity_mag[0],inhibit_scale[0])
                prev_weights[0][:]= net.update_weights_dynamics(prev_weights[0][:],delta[0])
                prev_weights[0][prev_weights[0][:]<0]=0

                net=attractorNetworkSettling(N[1],num_links[1],excite[1], activity_mag[1],inhibit_scale[1])
                prev_weights[1][:]= net.update_weights_dynamics(prev_weights[1][:],delta[1])
                prev_weights[1][prev_weights[1][:]<0]=0

            '''decoding mangnitude and direction of movement'''
            trans=activityDecoding(prev_weights[0][:],num_links[0],N[0])/SCALING_FACTOR#-prev_trans
            angVel=np.deg2rad(activityDecodingAngle(prev_weights[1][:],num_links[1],N[1]))#-prev_angle

            curr_parameter[2]=(curr_parameter[2]+angVel)%(2*np.pi)
            curr_parameter[0]=curr_parameter[0] + (trans*np.cos(curr_parameter[2]))
            curr_parameter[1]=curr_parameter[1]+ (trans*np.sin(curr_parameter[2]))
            # curr_z=curr_z+del_z
            
            ax1.set_title("Velocity Attractor Network")
            im=np.outer(prev_weights[0][:],prev_weights[1][:])
            ax1.imshow(im,interpolation='nearest', aspect='auto')

            # print(delta1, delta2, del_y,del_x)
            ax2.set_title("Decoded Pose")
            ax2.scatter(curr_parameter[0], curr_parameter[1],s=15)
            ax2.axis('equal')
            # ax2.set_xlim([-100,100])
            # ax2.set_ylim([-100,100])
            # ax2.set_zlim([0,N])


            print(str(delta[0])+"__"+str( delta[1])+ "------"+str(trans )+"__"+str(np.rad2deg(curr_parameter[2])))
            # print(x2-x1, y2-y1)
            # print(len(signal.find_peaks(prev_weights_trans)[0]),len(signal.find_peaks(prev_weights_angle)[0]) )
            

    ani = FuncAnimation(fig, animate, interval=1,frames=len(data_x),repeat=False)
    plt.show()

# this function runs but tuning necessary for accurate results (angle initiliased at 90)
def encodingDecodingMotion(data_x,data_y):
    '''Encoding change in translation and rotation (using scale factor to avoid wraparound and very small changes)
       Decoding Attractor Network activity and integrating change in x and y positions 
       Comapring expected and decoded translation, angular velocity and angle'''
    global prev_weights, num_links, excite, activity_mag,inhibit_scale, curr_parameter
    N[1]=720
    prev_weights[1]=np.zeros(N[1])
    # num_links[1]=120
    excite[1]=67
    inhibit_scale[1]=0.005
    '''Initalise network'''            
    delta=[0,0]
    for i in range(len(delta)):
        net=attractorNetwork(N[i],num_links[i],excite[i], activity_mag[i],inhibit_scale[i])
        prev_weights[i][net.activation(delta[i])]=net.full_weights(num_links[i])
    
    '''Data Storage Parameters'''
    curr_x,curr_y=np.zeros((len(data_x))), np.zeros((len(data_y)))
    theta=np.zeros((len(data_x)))
    delta_ang_out,delta_ang=np.zeros((len(data_x))), np.zeros((len(data_x)))
    theta[0]=0
    theta[1]=90 #np.rad2deg(math.atan2(data_y[1]-data_y[0],data_x[1]- data_y[0]))
    tran,rot=np.zeros((len(data_x))), np.zeros((len(data_y)))
    rot[0]=0
    rot[1]=90#np.rad2deg(math.atan2(data_y[1]-data_y[0],data_x[1]- data_y[0]))
    tran_out,rot_out=np.zeros((len(data_x))), np.zeros((len(data_y)))

    for i in range(len(data_x)):
        if i>=2:
            '''encoding mangnitude and direction of movement'''
            x0=data_x[i-2]
            x1=data_x[i-1]
            x2=data_x[i]
            y0=data_y[i-2]
            y1=data_y[i-1]
            y2=data_y[i]
            
            delta[0]=np.sqrt(((x2-x1)**2)+((y2-y1)**2)) *SCALING_FACTOR#translation
            
            delta[1]=((np.rad2deg(math.atan2(y2-y1,x2-x1)) - np.rad2deg(math.atan2(y1-y0,x1-x0))))+SCALING_FACTOR2#%360     #angle
           
            '''updating network'''
            net0=attractorNetworkSettling(N[0],num_links[0],excite[0], activity_mag[0],inhibit_scale[0])
            net1=attractorNetworkSettling(N[1],num_links[1],excite[1], activity_mag[1],inhibit_scale[1])

            for n in range(3):
                prev_weights[0][:]= net0.update_weights_dynamics(prev_weights[0][:],delta[0])
                prev_weights[0][prev_weights[0][:]<0]=0
                
                prev_weights[1][:]= net1.update_weights_dynamics(prev_weights[1][:],delta[1])
                prev_weights[1][prev_weights[1][:]<0]=0

            '''decoding mangnitude and direction of movement'''
            trans=activityDecoding(prev_weights[0][:],num_links[0],N[0])/SCALING_FACTOR#-prev_trans
            del_angle=(activityDecoding(prev_weights[1][:],num_links[1],N[1]))#-prev_angle
            
            theta[i]=(theta[i-1]+(del_angle-SCALING_FACTOR2))%360#(2*np.pi)
            curr_x[i]=curr_x[i-1]+ (trans*np.cos(np.deg2rad(theta[i-1])))
            curr_y[i]=curr_y[i-1]+ (trans*np.sin(np.deg2rad(theta[i-1])))
        

            tran[i]=delta[0]/SCALING_FACTOR
            delta_ang[i]=(delta[1]-SCALING_FACTOR2) #%360
            rot[i]=(rot[i-1]+(delta[1]-SCALING_FACTOR2)) % 360

            tran_out[i]=trans 
            delta_ang_out[i]=(del_angle-SCALING_FACTOR2) #% 360
            rot_out[i]=theta[i]
            # (delta[1]/SCALING_FACTOR)%360
            print(str(i)+ "  "+str(delta[0]/SCALING_FACTOR)+"  "+str(delta[1] )+ "_______"+str(trans )+"  "+str(del_angle))
            
            

    print(f"Final Error Translate: {np.sum(abs(tran-tran_out))}")
    print(f"Final Error Rotate: {np.sum(abs(delta_ang-delta_ang_out))}")
    fig = plt.figure(figsize=(7, 7))
    ax0 = fig.add_subplot(4, 2, 1)
    ax1 = fig.add_subplot(4, 2, 2)
    ax2 = fig.add_subplot(4, 2, 3)
    ax3 = fig.add_subplot(4, 2, 4)
    ax4 = fig.add_subplot(4, 2, 5)
    ax5 = fig.add_subplot(4, 2, 6)
    ax6 = fig.add_subplot(4, 2, 7)
    ax7 = fig.add_subplot(4, 2, 8)

    ax1.set_title('Converted')
    ax1.scatter(curr_x, curr_y,c='b',s=5)
    ax1.axis('equal')
    # ax1.set_xlim([-300,300])
    # ax1.set_ylim([-100,500])

    ax0.set_title('Original')
    ax0.scatter(data_x, data_y,c='b',s=5)
    ax0.axis('equal')
    # ax0.set_xlim([-300,300])
    # ax0.set_ylim([-100,500])

    ax2.set_title('Traslation Input')
    ax2.plot(tran,'-o',markersize=0.5,linewidth=0.05)
    # ax2.axis('equal')
    ax3.set_title('Traslation Output')
    ax3.plot(tran_out,'-o',markersize=0.5,linewidth=0.05)
    # ax3.axis('equal')

    ax4.set_title('Delta Rotation Input')
    ax4.plot(delta_ang,'g.',markersize=2)
    ax4.set_ylim([-360,360])
    # ax4.axis('equal')

    ax5.set_title('Delta Rotation Output')
    ax5.plot(delta_ang_out,'g.',markersize=2)
    ax5.set_ylim([-360,360])
    # ax5.axis('equal')

    ax6.set_title('Rotation Input')
    ax6.plot(rot,'-k.',markersize=1,linewidth=0.05)
    ax6.set_ylim([0,360])
    # ax6.axis('equal')

    ax7.set_title('Rotation Output')
    ax7.plot(rot_out,'-k.',markersize=1,linewidth=0.05)
    ax7.set_ylim([0,360])

    
    fig.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    plt.gcf().text(0.02,0.02,"N= " + str(N[1]) +",  num links= " + str(num_links[1]) + ",  excite links= " + str(excite[1]) + ", inhibition=" + str(inhibit_scale[1]),  fontsize=8)
    

    plt.show()

#note: need to fix the angular velocity network decoding 
def CompareState_Velocity_Networks(data_x,data_y):
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
        global prev_weights,prev_weights_rot, num_links, excite, activity_mag,inhibit_scale
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

    
            net1=attractorNetwork(N[1],num_links[1],excite[1], activity_mag[1],inhibit_scale[1])
            prev_weights[1][:]= net1.update_weights_dynamics(prev_weights[1][:],rotation)
            prev_weights[1][prev_weights[1][:]<0]=0
            decoded_rotation=activityDecodingAngle(prev_weights[1][:],num_links[1],N[1])#-prev_trans

            net2=attractorNetworkSettling(N[0],num_links[0],excite[0], activity_mag[0],inhibit_scale[0])
            prev_weights[0][:]= net2.update_weights_dynamics(prev_weights[0][:],rotation)
            prev_weights[0][prev_weights[0][:]<0]=0
            decoded_angVel=activityDecodingAngle(prev_weights[0][:],num_links[0],N[0])#-prev_trans
            

            "Adding translation and rotation to previous posisiton"
            curr_x[i]=curr_x[i-1]+ (translation*np.cos(np.deg2rad(decoded_rotation)))
            curr_y[i]=curr_y[i-1]+ (translation*np.sin(np.deg2rad(decoded_rotation)))
            
            '''Path integration with angular velocity'''
            theta[i]=(theta[i-1]+(decoded_angVel))%360#(2*np.pi)
            curr_parameter[0]=curr_parameter[0]+translation*np.cos(np.deg2rad(theta[i-1]))
            curr_parameter[1]=curr_parameter[1]+translation*np.sin(np.deg2rad(theta[i-1]))
            
            print(f"{str(i)}   {str(translation)}   {str(rotation )}  ----- {decoded_rotation}   {decoded_angVel}")
            
            
            ax0.plot(x2,y2,"k.")
            ax0.set_title("Ground Truth Position")
            ax0.axis('equal')
            # ax0.axis('equal')  

            ax3.set_title("Attractor Angular Velocity Network")
            ax3.plot(curr_parameter[0],curr_parameter[1],'r.')
            ax3.axis('equal') 
            # 
            ax2.set_title("Attractor Rotation Network")
            ax2.plot(curr_x[i],curr_y[i],'g.')
            ax2.axis('equal')

            
    ani = FuncAnimation(fig, animate, interval=1,frames=len(data_x),repeat=False)
    plt.show()


'''Test Area'''
sparse_gt=data_processing()#[0::4]
data_x=sparse_gt[:, :, 3][:,0][:200]
data_y=sparse_gt[:, :, 3][:,2][:200]

# data_y=np.concatenate([np.zeros(100), np.arange(100), np.ones(100)*100, np.arange(100,5,-1)])
# data_x=np.concatenate([np.arange(100), np.ones(100)*100, np.arange(100,0,-1), np.zeros(95)])

# data_x=np.arange(200)
# data_y=np.zeros(200)

data_y=np.zeros(100)
data_x=np.arange(100)


# data_y=np.zeros(200)
# data_x=np.concatenate([np.linspace(0,0.1,10), np.linspace(0.2,1,10), np.linspace(2,10,10), np.linspace(20,100,10),np.linspace(110,1000,10)])
# data_x=np.linspace(0,796,200)


'''Functions'''
# testing_Conversion(sparse_gt)
# visualise(data_x,data_y)
encodingDecodingMotion(data_x,data_y)
# CompareState_Velocity_Networks(data_x,data_y)


