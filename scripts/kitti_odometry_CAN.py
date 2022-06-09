import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D 
from scipy import signal

# from mayavi import mlab

'''Parameters'''
N=200 #number of neurons
neurons=np.arange(0,N)
curr_Neuron=0
prev_weights_trans=np.zeros(N)
prev_weights_angle=np.zeros(N)
# prev_weights_z=np.zeros(N)
num_links=40
excite=20
activity_mag=1
inhibit_scale=0.005
curr_x,curr_y,curr_z=0,0,0 
x,y=0,0

class attractorNetwork:
    '''defines 1D attractor network with N neurons, angles associated with each neurons 
    along with inhitory and excitatory connections to update the weights'''
    def __init__(self, delta1, delta2, N, num_links, excite_radius, activity_mag,inhibit_scale):
        self.delta1=delta1
        self.delta2=delta2
        # self.delta3=delta3
        self.excite_radius=excite_radius
        self.N=N  
        self.num_links=num_links
        self.activity_mag=activity_mag
        self.inhibit_scale=inhibit_scale

    def neuron_update(self,prev_weights_x,prev_weights_y):
        indexes=np.arange(N)
        non_zero_idxs_x=indexes[prev_weights_x>0]
        non_zero_idxs_y=indexes[prev_weights_y>0]
        # non_zero_idxs_z=indexes[prev_weights_z>0]
        return (non_zero_idxs_x+self.delta1) % N, (non_zero_idxs_y+self.delta2) % N
        
    def inhibitions(self,id):
        ''' each nueuron inhibits all other nueurons but itself'''
        return np.delete(np.arange(self.N),self.excitations(id))

    def excitations(self,id):
        '''each neuron excites itself and num_links neurons left and right with wraparound connections'''
        excite=[]
        for i in range(-self.excite_radius,self.excite_radius+1):
            excite.append((id + i) % N)
        return np.array(excite)

    def activation(self,id):
        '''each neuron excites itself and num_links neurons left and right with wraparound connections'''
        excite=[]
        for i in range(-self.num_links,self.num_links+1):
            excite.append((id + i) % N)
        return np.array(excite)

    def full_weights(self,radius):
        x=np.arange(-radius,radius+1)
        return 1/(np.std(x) * np.sqrt(2 * np.pi)) * np.exp( - (x - np.mean(x))**2 / (2 * np.std(x)**2))  

    def update_weights_dynamics(self,prev_weights,neurons):
        indexes,non_zero_weights,non_zero_weights_shifted, inhbit_val=np.arange(N),np.zeros(N),np.zeros(N),0

        '''copied and shifted activity'''
        non_zero_idxs=indexes[prev_weights>0] # indexes of non zero prev_weights
        
        non_zero_weights[non_zero_idxs]=prev_weights[non_zero_idxs] 
        
        non_zero_weights_shifted[neurons]=prev_weights[non_zero_idxs] #non zero weights shifted by delta
        
        '''inhibition'''
        for i in range(len(non_zero_weights_shifted)):
            inhbit_val+=non_zero_weights_shifted[i]*self.inhibit_scale
        
        '''excitation'''
        excite=np.zeros(N)
        for i in range(len(non_zero_idxs)):
            excite[self.excitations(non_zero_idxs[i])]+=self.full_weights(self.excite_radius)*prev_weights[non_zero_idxs[i]]

        prev_weights+=(non_zero_weights_shifted+excite-inhbit_val)
        return prev_weights/np.linalg.norm(prev_weights)

def data_processing():
    poses = pd.read_csv('./data/dataset/poses/00.txt', delimiter=' ', header=None)
    gt = np.zeros((len(poses), 3, 4))
    for i in range(len(poses)):
        gt[i] = np.array(poses.iloc[i]).reshape((3, 4))
    return gt

def visualise(sparse_gt):
    fig = plt.figure(figsize=(13, 4))
    ax0 = fig.add_subplot(1, 3, 1)
    ax1 = fig.add_subplot(1, 3, 2)
    ax2 = fig.add_subplot(1, 3, 3)

    '''Initalise network'''            
    current,prediction, velocity=[],[],[]
    delta1=sparse_gt[:, :, 3][0,2] #y_axis
    delta2=sparse_gt[:, :, 3][0,0] #x_axis
    # delta3=sparse_gt[:, :, 3][0,1] #x_axis
    net=attractorNetwork(int(delta1),int(delta2),N,num_links,int(excite), activity_mag,inhibit_scale)
    prev_weights_trans[net.activation(int(delta1))]=net.full_weights(num_links)
    prev_weights_angle[net.activation(int(delta2))]=net.full_weights(num_links)
    # prev_weights_z[net.activation(int(delta3))]=net.full_weights(num_links)

    
    def animate(i):
        ax0.set_title("Ground Truth Pose")
        ax0.scatter(sparse_gt[:, :, 3][i, 0],sparse_gt[:, :, 3][i, 2],s=15)
        ax0.set_xlim([-300,300])
        ax0.set_ylim([-100,500])
        # ax0.set_zlim([-50,50])
        ax0.invert_yaxis()
        # ax0.view_init(elev=39, azim=140)

        global prev_weights_trans,prev_weights_angle, num_links, excite, activity_mag,inhibit_scale, curr_x, curr_y, curr_z
        ax1.clear()
        if i>=1:
            '''distributed weights with excitations and inhibitions'''
            x1=sparse_gt[:, :, 3][i-1,0]
            x2=sparse_gt[:, :, 3][i,1]
            y1=sparse_gt[:, :, 3][i-1,2]
            y2=sparse_gt[:, :, 3][i,2]
            
            delta1=np.sqrt(((x2-x1)**2)+((y2-y2)**2)) #translation
            delta2=np.rad2deg(np.arctan2(y2-y1,x2-x1)) #angle
            # delta3=sparse_gt[:, :, 3][i,1]-sparse_gt[:, :, 3][i-1,1] #z_axis
            prev_angle=np.argmax(prev_weights_angle)
            prev_trans=np.argmax(prev_weights_trans)
            # prev_z=np.argmax(prev_weights_z)
            

            net=attractorNetwork(int(delta1),int(delta2),N,num_links,int(excite),activity_mag,inhibit_scale)
            Neurons1,Neurons2=net.neuron_update(prev_weights_trans,prev_weights_angle)
            prev_weights_trans= net.update_weights_dynamics(prev_weights_trans,Neurons1)
            prev_weights_angle= net.update_weights_dynamics(prev_weights_angle,Neurons2)
            # prev_weights_z= net.update_weights_dynamics(prev_weights_z,Neurons3)
            prev_weights_trans[prev_weights_trans<0]=0
            prev_weights_angle[prev_weights_angle<0]=0
            # prev_weights_z[prev_weights_z<0]=0

            ax1.set_title("2D Attractor Network")
            ax1.imshow(np.tile(prev_weights_trans,(N,1)).T*np.tile(prev_weights_angle,(N,1)))
            # ax1.pts = mlab.points3d(prev_weights_x, prev_weights_y, prev_weights_z, scale_mode='none', scale_factor=0.07)
            # ax1.scatter(prev_weights_x, prev_weights_y, prev_weights_z)
    
            # del_trans=np.argmax(prev_weights_trans)-prev_trans
            # del_angle=np.argmax(prev_weights_angle)-prev_angle
            # del_z=np.argmax(prev_weights_z)-prev_z
            trans=np.argmax(prev_weights_trans)
            angle=np.deg2rad(np.argmax(prev_weights_angle))
            curr_x=curr_x + (trans*np.cos(angle))
            curr_y=curr_y+(trans*np.sin(angle))
            # curr_z=curr_z+del_z

            # print(delta1, delta2, del_y,del_x)
            ax2.set_title("Decoded Pose")
            ax2.scatter(curr_x, curr_y,c='b',s=15)
            # ax2.set_xlim([0,N])
            # ax2.set_ylim([0,N])
            # ax2.set_zlim([0,N])
            # ax2.invert_yaxis()
            # ax2.view_init(elev=39, azim=140)

            print(str(delta1 )+"--"+str( delta2)+ "------"+str(trans )+"--"+str( np.rad2deg(angle)))
            # print(x2-x1, y2-y1)
            # print(len(signal.find_peaks(prev_weights_trans)[0]),len(signal.find_peaks(prev_weights_angle)[0]) )
            

    ani = FuncAnimation(fig, animate, interval=1,frames=len(sparse_gt),repeat=False)
    plt.show()


'''Test Area'''
sparse_gt=data_processing()[0::10]
# visualise(sparse_gt)


def testing_Conversion(sparse_gt):
    fig = plt.figure(figsize=(13, 4))
    ax0 = fig.add_subplot(1, 2, 1)
    ax1 = fig.add_subplot(1, 2, 2)
    def animate(i):
        global curr_x, curr_y, x, y
        x1=sparse_gt[:,:,3][i,0]
        y1=sparse_gt[:,:,3][i,2]

        x2=sparse_gt[:,:,3][i+1,0]
        y2=sparse_gt[:,:,3][i+1,2]

        delta1=np.sqrt(((x2-x1)**2)+((y2-y2)**2)) #translation
        delta2=(np.arctan2(y2-y1,x2-x1)) #angle

        curr_x=curr_x + (delta1*np.cos(delta2))
        curr_y=curr_y+(delta1*np.sin(delta2))

        x=x+(x2-x1)
        y=y+(y2-y1)

        ax1.set_title('Converted')
        ax1.scatter(curr_x, curr_y,c='b',s=15)
        ax1.set_xlim([-300,300])
        ax1.set_ylim([-100,500])

        ax0.set_title('Original')
        ax0.scatter(x, y,c='b',s=15)
        ax0.set_xlim([-300,300])
        ax0.set_ylim([-100,500])

    ani = FuncAnimation(fig, animate, interval=1,frames=len(sparse_gt),repeat=False)
    plt.show()


testing_Conversion(sparse_gt)