import matplotlib.pyplot as plt
import numpy as np
import math
import os
import pandas as pd
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D 
from scipy import signal

# from mayavi import mlab

'''Parameters'''
N=[1200,360] #number of neurons
neurons=[np.arange(0,N[0]), np.arange(0,N[1])]
curr_Neuron=[0,0]
prev_weights=[np.zeros(N[0]), np.zeros(N[1])]
# prev_weights_z=np.zeros(N)
num_links=[50,60]
excite=[30,47]
activity_mag=[1,1]
inhibit_scale=[0.005,0.005]
curr_parameter=[0,0]
curr_x,curr_y=0,0
x,y=0,0

# print(prev_weights[0][:])
def angdiff( th1, th2):
    d = th1 - th2
    d = np.mod(d+np.pi, 2*np.pi) - np.pi
    return d

class attractorNetwork:
    '''defines 1D attractor network with N neurons, angles associated with each neurons 
    along with inhitory and excitatory connections to update the weights'''
    def __init__(self, delta, N, num_links, excite_radius, activity_mag,inhibit_scale):
        self.delta=delta
        self.excite_radius=excite_radius
        self.N=N  
        self.num_links=num_links
        self.activity_mag=activity_mag
        self.inhibit_scale=inhibit_scale

    def neuron_update(self,prev_weights):
        indexes=np.arange(self.N)
        non_zero_idxs=indexes[prev_weights>0]
        return (non_zero_idxs+ int(self.delta)) % self.N
        
    def inhibitions(self,id):
        ''' each nueuron inhibits all other nueurons but itself'''
        return np.delete(np.arange(self.N),self.excitations(id))

    def excitations(self,id):
        '''each neuron excites itself and num_links neurons left and right with wraparound connections'''
        excite=[]
        for i in range(-self.excite_radius,self.excite_radius+1):
            excite.append((id + i) % self.N)
        return np.array(excite)

    def activation(self,id):
        '''each neuron excites itself and num_links neurons left and right with wraparound connections'''
        excite=[]
        for i in range(-self.num_links,self.num_links+1):
            excite.append((int(id) + i) % self.N)
        return np.array(excite)

    def full_weights(self,radius):
        x=np.arange(-radius,radius+1)
        return 1/(np.std(x) * np.sqrt(2 * np.pi)) * np.exp( - (x - np.mean(x))**2 / (2 * np.std(x)**2))  

    def fractional_weights(self,non_zero_prev_weights,delta):
        frac=delta%1
        if frac == 0:
            return non_zero_prev_weights
        else: 
            inv_frac=1-frac
            frac_weights=np.zeros((len(non_zero_prev_weights)))
            frac_weights[0]=non_zero_prev_weights[0]*inv_frac
            for i in range(1,len(non_zero_prev_weights)):
                frac_weights[i]=non_zero_prev_weights[i-1]*frac + non_zero_prev_weights[i]*inv_frac
            return frac_weights

    def update_weights_dynamics(self,prev_weights):
        indexes,non_zero_weights,non_zero_weights_shifted, inhbit_val=np.arange(self.N),np.zeros(self.N),np.zeros(self.N),0
        shifted_indexes=self.neuron_update(prev_weights)

        '''copied and shifted activity'''
        non_zero_idxs=indexes[prev_weights>0] # indexes of non zero prev_weights
        
        non_zero_weights[non_zero_idxs]=prev_weights[non_zero_idxs] 
        
        non_zero_weights_shifted[shifted_indexes]=self.fractional_weights(prev_weights[non_zero_idxs],self.delta) #non zero weights shifted by delta
        
        '''inhibition'''
        for i in range(len(non_zero_weights_shifted)):
            inhbit_val+=non_zero_weights_shifted[i]*self.inhibit_scale
        
        '''excitation'''
        excite=np.zeros(self.N)
        for i in range(len(non_zero_idxs)):
            excite[self.excitations(non_zero_idxs[i])]+=self.full_weights(self.excite_radius)*prev_weights[non_zero_idxs[i]]

        prev_weights+=(non_zero_weights_shifted+excite-inhbit_val)
        return prev_weights/np.linalg.norm(prev_weights)

class attractorNetworkSettling:
    '''defines 1D attractor network with N neurons, angles associated with each neurons 
    along with inhitory and excitatory connections to update the weights'''
    def __init__(self, N, num_links, excite_radius, activity_mag,inhibit_scale):
        self.excite_radius=excite_radius
        self.N=N  
        self.num_links=num_links
        self.activity_mag=activity_mag
        self.inhibit_scale=inhibit_scale
        
    def inhibitions(self,id):
        ''' each nueuron inhibits all other nueurons but itself'''
        return np.delete(np.arange(self.N),self.excitations(id))

    def excitations(self,id):
        '''each neuron excites itself and num_links neurons left and right with wraparound connections'''
        excite=[]
        for i in range(-self.excite_radius,self.excite_radius+1):
            excite.append((id + i) % self.N)
        return np.array(excite)

    def activation(self,id):
        '''each neuron excites itself and num_links neurons left and right with wraparound connections'''
        excite=[]
        for i in range(-self.num_links,self.num_links+1):
            excite.append((int(id) + i) % self.N)
        return np.array(excite)

    def full_weights(self,radius):
        x=np.arange(-radius,radius+1)
        return 1/(np.std(x) * np.sqrt(2 * np.pi)) * np.exp( - (x - np.mean(x))**2 / (2 * np.std(x)**2))  

    def fractional_weights(self,non_zero_prev_weights,activeNeuron):
        frac=activeNeuron%1
        if frac == 0:
            return non_zero_prev_weights
        else: 
            inv_frac=1-frac
            frac_weights=np.zeros((len(non_zero_prev_weights)))
            frac_weights[0]=non_zero_prev_weights[0]*inv_frac
            for i in range(1,len(non_zero_prev_weights)):
                frac_weights[i]=non_zero_prev_weights[i-1]*frac + non_zero_prev_weights[i]*inv_frac
            return frac_weights

    def update_weights_dynamics(self,prev_weights,activeNeuron):

        delta=(int(activeNeuron)-np.argmax(prev_weights))%self.N

        indexes,non_zero_weights,non_zero_weights_shifted, inhbit_val=np.arange(self.N),np.zeros(self.N),np.zeros(self.N),0
        # shifted_indexes=self.neuron_update(prev_weights)

        '''copied and shifted activity'''
        non_zero_idxs=indexes[prev_weights>0] # indexes of non zero prev_weights

        if len(prev_weights[non_zero_idxs])==0:
            prev_weights[self.activation(activeNeuron)]=self.full_weights(self.num_links)
            non_zero_idxs=indexes[prev_weights>0] # indexes of non zero prev_weights
        
        non_zero_weights[non_zero_idxs]=prev_weights[non_zero_idxs] 
        
        non_zero_weights_shifted[(non_zero_idxs+delta)%self.N]=self.fractional_weights(prev_weights[non_zero_idxs],activeNeuron) #non zero weights shifted by delta
        
        '''inhibition'''
        for i in range(len(non_zero_weights_shifted)):
            inhbit_val+=non_zero_weights_shifted[i]*self.inhibit_scale
        
        '''excitation'''
        excite=np.zeros(self.N)
        for i in range(len(non_zero_idxs)):
            excite[self.excitations(non_zero_idxs[i])]+=self.full_weights(self.excite_radius)*prev_weights[non_zero_idxs[i]]

        prev_weights+=(non_zero_weights_shifted+excite-inhbit_val)
        return prev_weights/np.linalg.norm(prev_weights)

def activityDecoding(prev_weights,radius,N,neurons):
    '''Isolating activity at a radius around the peak to decode position'''
    peak=np.argmax(prev_weights) 
    local_activity=np.zeros(N)
    local_activity_idx=[]
    for i in range(-radius,radius+1):
        local_activity_idx.append((peak + i) % N)
    local_activity[local_activity_idx]=prev_weights[local_activity_idx]

    x,y=local_activity*np.cos(np.deg2rad(neurons*360/N)), local_activity*np.sin(np.deg2rad(neurons*360/N))
    vect_sum=np.rad2deg(math.atan2(sum(y),sum(x))) % 360
    weighted_sum = N*(vect_sum/360)


    # weighted_sum=0
    # for i in range(len(local_activity_idx)):
    #     weighted_sum+=local_activity_idx[i]*prev_weights[local_activity_idx[i]]
    return weighted_sum

def activityDecodingAngle(prev_weights,radius,N,neurons):
    '''Isolating activity at a radius around the peak to decode position'''
    peak=np.argmax(prev_weights) 
    local_activity=np.zeros(N)
    local_activity_idx=[]
    for i in range(-radius,radius+1):
        local_activity_idx.append((peak + i) % N)
    local_activity[local_activity_idx]=prev_weights[local_activity_idx]

    x,y=local_activity*np.cos(np.deg2rad(neurons*360/N)), local_activity*np.sin(np.deg2rad(neurons*360/N))
    vect_sum=np.rad2deg(math.atan2(sum(y),sum(x))) % 360
    # changing range from [-179, 180] to [0,360]
    # if vect_sum<0:
    #     shifted_vec=vect_sum+360
    # else:
    #     shifted_vec=vect_sum
    # return shifted_vec*(N/360)
    return vect_sum

def data_processing():
    poses = pd.read_csv('./data/dataset/poses/00.txt', delimiter=' ', header=None)
    gt = np.zeros((len(poses), 3, 4))
    for i in range(len(poses)):
        gt[i] = np.array(poses.iloc[i]).reshape((3, 4))
    return gt

def testing_Conversion(sparse_gt):
    fig = plt.figure(figsize=(13, 4))
    ax0 = fig.add_subplot(1, 2, 1)
    ax1 = fig.add_subplot(1, 2, 2)
    def animate(i):
        global curr_x, curr_y, x, y
        if i>0:
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

            curr_x=curr_x + (delta1*np.cos(delta2))
            curr_y=curr_y + (delta1*np.sin(delta2))

            x=x+(x2-x1)
            y=y+(y2-y1)

            print(delta1, delta2)

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

def visualise(data_x,data_y):
    fig = plt.figure(figsize=(13, 4))
    ax0 = fig.add_subplot(1, 3, 1)
    ax1 = fig.add_subplot(1, 3, 2)
    ax2 = fig.add_subplot(1, 3, 3)

    '''Initalise network'''            
    delta=[0,0]
    for i in range(len(delta)):
        net=attractorNetwork(delta[i],N[i],num_links[i],excite[i], activity_mag[i],inhibit_scale[i])
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
            
            delta[0]=np.sqrt(((x2-x1)**2)+((y2-y1)**2)) #translation
            delta[1]=np.rad2deg(math.atan2(y2-y1,x2-x1)) % 360          #angle

           
            '''updating network'''
            prev_trans=activityDecoding(prev_weights[0][:],num_links[0],N[0],neurons[0][:])
            # prev_angle=activityDecoding(prev_weights[1][:],num_links[1],N[1],neurons[1][:])
            
            
            net=attractorNetwork(delta[0],N[0],num_links[0],excite[0], activity_mag[0],inhibit_scale[0])
            prev_weights[0][:]= net.update_weights_dynamics(prev_weights[0][:])
            prev_weights[0][prev_weights[0][:]<0]=0

            net=attractorNetworkSettling(N[1],num_links[1],excite[1], activity_mag[1],inhibit_scale[1])
            prev_weights[1][:]= net.update_weights_dynamics(prev_weights[1][:],delta[1])
            prev_weights[1][prev_weights[1][:]<0]=0
                
            ax1.set_title("2D Attractor Network")
            # im=np.tile(prev_weights[0][:],(N[0],1)).T*np.tile(prev_weights[1][:],(N[1],1))
            im=np.outer(prev_weights[0][:],prev_weights[1][:])
            ax1.imshow(im,interpolation='nearest', aspect='auto')
    
            '''decoding mangnitude and direction of movement'''
            trans=activityDecoding(prev_weights[0][:],num_links[0],N[0],neurons[0][:]) - prev_trans
            angle=np.deg2rad(activityDecodingAngle(prev_weights[1][:],num_links[1],N[1],neurons[1][:]))

            curr_parameter[0]=curr_parameter[0] + (trans*np.cos(angle))
            curr_parameter[1]=curr_parameter[1]+ (trans*np.sin(angle))
            # curr_z=curr_z+del_z

            # print(delta1, delta2, del_y,del_x)
            ax2.set_title("Decoded Pose")
            ax2.scatter(curr_parameter[0], curr_parameter[1],s=15)
            ax2.axis('equal')
            # ax2.set_xlim([-100,100])
            # ax2.set_ylim([-100,100])
            # ax2.set_zlim([0,N])


            print(str(delta[0])+"__"+str( delta[1])+ "------"+str(trans )+"__"+str(np.rad2deg(angle)))
            # print(x2-x1, y2-y1)
            # print(len(signal.find_peaks(prev_weights_trans)[0]),len(signal.find_peaks(prev_weights_angle)[0]) )
            

    ani = FuncAnimation(fig, animate, interval=1,frames=len(data_x),repeat=False)
    plt.show()

def encodingDecodingMotion(data_x,data_y):
    global prev_weights, num_links, excite, activity_mag,inhibit_scale, curr_parameter

    '''Initalise network'''            
    delta=[0,0]
    # for i in range(len(delta)):
    #     net=attractorNetwork(delta[i],N[i],num_links[i],excite[i], activity_mag[i],inhibit_scale[i])
    #     prev_weights[i][net.activation(delta[i])]=net.full_weights(num_links[i])
   
    # [1,:]ev_weights_angle[net.activation(delta2)]=net.full_weights(num_links)
    # prev_weights_z[net.activation(int(delta3))]=net.full_weights(num_links)
    
    curr_x,curr_y=np.zeros((len(data_x))), np.zeros((len(data_y)))
    tran,rot=np.zeros((len(data_x))), np.zeros((len(data_y)))
    tran_out,rot_out=np.zeros((len(data_x))), np.zeros((len(data_y)))
    for i in range(len(data_x)):
        if i>=1:
            '''encoding mangnitude and direction of movement'''
            x1=data_x[i-1]
            x2=data_x[i]
            y1=data_y[i-1]
            y2=data_y[i]
            
            delta[0]=np.sqrt(((x2-x1)**2)+((y2-y1)**2)) *100 #translation
            delta[1]=np.rad2deg(math.atan2(y2-y1,x2-x1)) % 360          #angle
            
            '''updating network'''
            # prev_trans=np.argmax(prev_weights[0][:])
            # prev_angle=np.argmax(prev_weights[1][:])

            prev_trans=activityDecoding(prev_weights[0][:],num_links[0],N[0],neurons[0][:])
            prev_angle=activityDecoding(prev_weights[1][:],num_links[1],N[1],neurons[1][:])
                
            net=attractorNetworkSettling(N[0],num_links[0],excite[0], activity_mag[0],inhibit_scale[0])
            prev_weights[0][:]= net.update_weights_dynamics(prev_weights[0][:],delta[0])
            prev_weights[0][prev_weights[0][:]<0]=0

            net=attractorNetworkSettling(N[1],num_links[1],excite[1], activity_mag[1],inhibit_scale[1])
            prev_weights[1][:]= net.update_weights_dynamics(prev_weights[1][:],delta[1])
            prev_weights[1][prev_weights[1][:]<0]=0

            '''decoding mangnitude and direction of movement'''
            # trans=np.argmax(prev_weights[0][:])-prev_trans
            # angle=np.argmax(prev_weights[1][:])-prev_angle
            trans=activityDecoding(prev_weights[0][:],num_links[0],N[0],neurons[0][:])/100#-prev_trans
            angle=np.deg2rad(activityDecodingAngle(prev_weights[1][:],num_links[1],N[1],neurons[1][:]))#-prev_angle

            curr_x[i]=curr_x[i-1]+ (trans*np.cos(angle))
            curr_y[i]=curr_y[i-1]+ (trans*np.sin(angle))
        

            tran[i]=delta[0]
            rot[i]= delta[1]

            tran_out[i]=trans
            rot_out[i]=np.rad2deg(angle)
            print(str(delta[0])+"  "+str( delta[1])+ "_______"+str(trans )+"  "+str(np.rad2deg(angle)))

    fig = plt.figure(figsize=(13, 10))
    plt.gcf().text(0.02,0.02,"N= " + str(N[0]) +",  num links= " + str(num_links[0]) + ",  excite links= " + str(excite[0]),  fontsize=14)
    ax0 = fig.add_subplot(3, 2, 1)
    ax1 = fig.add_subplot(3, 2, 2)
    ax2 = fig.add_subplot(3, 2, 3)
    ax3 = fig.add_subplot(3, 2, 4)
    ax4 = fig.add_subplot(3, 2, 5)
    ax5 = fig.add_subplot(3, 2, 6)


    ax1.set_title('Converted')
    ax1.scatter(curr_x, curr_y,c='b',s=15)
    ax1.axis('equal')
    # ax1.set_xlim([-300,300])
    # ax1.set_ylim([-100,500])

    ax0.set_title('Original')
    ax0.scatter(data_x, data_y,c='b',s=15)
    ax0.axis('equal')
    # ax0.set_xlim([-300,300])
    # ax0.set_ylim([-100,500])

    ax2.set_title('Traslation Input')
    ax2.scatter(np.arange(len(tran)),tran,s=5)
    # ax2.axis('equal')
    ax4.set_title('Traslation Output')
    ax4.scatter(np.arange(len(tran_out)),tran_out,s=5)
    # ax4.axis('equal')

    ax3.set_title('Rotation Input')
    ax3.scatter(np.arange(len(rot)),rot,s=5)
    ax3.set_ylim([0,360])
    # ax3.axis('equal')
    ax5.set_title('Rotation Output')
    ax5.scatter(np.arange(len(rot_out)),rot_out,s=5)
    ax5.set_ylim([0,360])

    # ax5.axis('equal')

    plt.show()





'''Test Area'''
sparse_gt=data_processing()[0::4]
data_x=sparse_gt[:, :, 3][:,0]#[:1000]
data_y=sparse_gt[:, :, 3][:,2]#[:1000]

# data_y=np.concatenate([np.zeros(100), np.arange(100), np.ones(100)*100, np.arange(100,5,-1)])
# data_x=np.concatenate([np.arange(100), np.ones(100)*100, np.arange(100,0,-1), np.zeros(95)])

# data_x=np.arange(200) *-1
# data_y=np.arange(200)*-1

# data_y=np.zeros(195)
# data_x=np.concatenate([np.arange(100), np.arange(100,5,-1)])

# visualise(data_x,data_y)
encodingDecodingMotion(data_x,data_y)

# print(np.rad2deg(math.atan2(0,-1)))

# testing_Conversion(sparse_gt)

# testing_Conversion(sparse_gt)


# print(np.shape(np.transpose(prev_weights[0][:])))

# print(np.shape(prev_weights[1][:]))

# print(np.outer(prev_weights[0][:],prev_weights[1][:]))