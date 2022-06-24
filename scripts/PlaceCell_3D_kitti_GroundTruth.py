
   
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D 

from CAN import activityDecoding, activityDecodingAngle, attractorNetworkSettling, attractorNetwork



'''Parameters'''
N=[200,200,200]
prev_weights=[np.zeros(N[0]), np.zeros(N[1]), np.zeros(N[2])]
num_links=[40,40,40]
excite=[20,20,20]
activity_mag=[1,1,1]
inhibit_scale=[0.005,0.005, 0.005]
curr_parameter=[0,0,0]


def data_processing():
    poses = pd.read_csv('./data/dataset/poses/08.txt', delimiter=' ', header=None)
    gt = np.zeros((len(poses), 3, 4))
    for i in range(len(poses)):
        gt[i] = np.array(poses.iloc[i]).reshape((3, 4))
    return gt

def visualise(sparse_gt):
    fig = plt.figure(figsize=(13, 4))
    ax0 = fig.add_subplot(1, 2, 1)
    ax1 = fig.add_subplot(1, 2, 2)
    # ax2 = fig.add_subplot(1, 3, 3)

    '''Initalise network'''            
    current,prediction, velocity=[],[],[]
    # delta1=sparse_gt[:, :, 3][0,2] #y_axis
    # delta2=sparse_gt[:, :, 3][0,0] #x_axis
    # delta3=sparse_gt[:, :, 3][0,1] #x_axis

    delta=[sparse_gt[:, :, 3][0,2],sparse_gt[:, :, 3][0,0],sparse_gt[:, :, 3][0,1]] # y , x, z axis

    for i in range(len(delta)):
        net=attractorNetwork(delta[i],N[i],num_links[i],excite[i], activity_mag[i],inhibit_scale[i])
        prev_weights[i][net.activation(delta[i])]=net.full_weights(num_links[i])
        prev_weights[i][prev_weights[i][:]<0]=0

    # net=attractorNetwork(int(delta1),int(delta2),int(delta3),N,num_links,int(excite), activity_mag,inhibit_scale)
    # prev_weights_x[net.activation(int(delta1))]=net.full_weights(num_links)
    # prev_weights_y[net.activation(int(delta2))]=net.full_weights(num_links)
    # prev_weights_z[net.activation(int(delta3))]=net.full_weights(num_links)

    
    def animate(i):
        if i==len(sparse_gt)-1:
            ax0.clear()
        ax0.set_title("Ground Truth Pose")
        ax0.set_ylim([-100,500])
        ax0.set_xlim([-400,500])
        ax0.invert_yaxis()
        ax0.scatter(sparse_gt[:, :, 3][i, 0],sparse_gt[:, :, 3][i, 2],s=15)
        
        # ax0.set_xlim([-300,300])
        # ax0.set_ylim([-100,500])
        # ax0.set_zlim([-50,50])
        # ax0.set_ylim(ax0.get_ylim()[::-1])
        
        # ax0.view_init(elev=39, azim=140)

        global prev_weights, num_links, excite, activity_mag,inhibit_scale, curr_parameter
        ax1.clear()
        if i>=1:
            '''distributed weights with excitations and inhibitions'''
            delta[0]=sparse_gt[:, :, 3][i,0]-sparse_gt[:, :, 3][i-1,0] #x_axis
            delta[1]=(sparse_gt[:, :, 3][i,2]-sparse_gt[:, :, 3][i-1,2]) #y_axis
            delta[2]=sparse_gt[:, :, 3][i,1]-sparse_gt[:, :, 3][i-1,1] #z_axis
            prev_x=np.argmax(prev_weights[0][:])
            prev_y=np.argmax(prev_weights[1][:])
            prev_z=np.argmax(prev_weights[2][:])
            

            '''updating network'''
            for j in range(len(delta)):
               net=attractorNetwork(delta[j],N[j],num_links[j],excite[j], activity_mag[j],inhibit_scale[j])
               prev_weights[j][:]= net.update_weights_dynamics(prev_weights[j][:],delta[j])
               prev_weights[j][prev_weights[j][:]<0]=0

            ax1.set_title("2D Attractor Network")
            im=np.outer(prev_weights[1][:],prev_weights[0][:])
            ax1.imshow(im,interpolation='nearest', aspect='auto')
            # ax1.imshow(np.tile(prev_weights_x,(N,1)).T*np.tile(prev_weights_y,(N,1)))
            # ax1.pts = mlab.points3d(prev_weights_x, prev_weights_y, prev_weights_z, scale_mode='none', scale_factor=0.07)
            # ax1.scatter(prev_weights_x, prev_weights_y, prev_weights_z)

    
            del_y=np.argmax(prev_weights[0][:])-prev_x
            del_x=np.argmax(prev_weights[1][:])-prev_y
            del_z=np.argmax(prev_weights[2][:])-prev_z
            curr_parameter[0]=curr_parameter[0]+del_x
            curr_parameter[1]=curr_parameter[1]+del_y
            curr_parameter[2]=curr_parameter[2]+del_z

            print(delta[0], delta[1], del_x,del_y)
            # ax2.set_title("Decoded Pose")
            
            # ax2.scatter(curr_x, curr_y,c='b',s=15)
            # ax2.set_xlim([0,N])
            # ax2.set_ylim([0,N])
            # # ax2.set_zlim([0,N])
            # ax2.invert_yaxis()
            # # ax2.view_init(elev=39, azim=140)
            

    ani = FuncAnimation(fig, animate, interval=1,frames=len(sparse_gt),repeat=True)
    plt.show()


'''Test Area'''
sparse_gt=data_processing()[0::20]
visualise(sparse_gt)
