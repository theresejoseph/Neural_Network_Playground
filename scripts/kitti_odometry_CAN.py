import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D 
from scipy import signal

# from mayavi import mlab

'''Parameters'''
N=[200,360] #number of neurons
neurons=[np.arange(0,N[0]), np.arange(0,N[1])]
curr_Neuron=[0,0]
prev_weights=[np.zeros(N[0]), np.zeros(N[1])]
# prev_weights_z=np.zeros(N)
num_links=[10,36]
excite=[5,24]
activity_mag=[1,1]
inhibit_scale=[0.01,0.005]
curr_parameter=[0,0]

# print(prev_weights[0][:])
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

def activityDecoding(prev_weights,radius,N,neurons):
    '''Isolating activity at a radius around the peak to decode position'''
    peak=np.argmax(prev_weights) 
    local_activity=np.zeros(N)
    local_activity_idx=[]
    for i in range(-radius,radius+1):
        local_activity_idx.append((peak + i) % N)
    local_activity[local_activity_idx]=prev_weights[local_activity_idx]
    x,y=local_activity*np.cos(np.deg2rad(neurons*360/N)), local_activity*np.sin(np.deg2rad(neurons*360/N))
    vect_sum=np.rad2deg(np.arctan2(sum(y),sum(x)))
    # changing range from [-179, 180] to [0,360]
    if vect_sum<0:
        shifted_vec=vect_sum+360
    else:
        shifted_vec=vect_sum
    return shifted_vec*(N/360)
    # return vect_sum

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
                delta2=(np.arctan2(y2-y1,x2-x1)) #angle

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

    
    def animate(i):
        ax0.set_title("Ground Truth Pose")
        ax0.scatter(data_x[i],data_y[i],s=15)
        # ax0.set_xlim([-300,300])
        # ax0.set_ylim([-100,500])
        # ax0.set_zlim([-50,50])
        # ax0.invert_yaxis()
        # ax0.view_init(elev=39, azim=140)
        
        global prev_weights, num_links, excite, activity_mag,inhibit_scale, curr_parameter
        ax1.clear()
        if i>=1:
            '''encoding mangnitude and direction of movement'''
            x1=data_x[i-1]
            x2=data_x[i]
            y1=data_y[i-1]
            y2=data_y[i]
            
            delta[0]=np.sqrt(((x2-x1)**2)+((y2-y1)**2)) #translation
            delta[1]=np.rad2deg(np.arctan2(y2-y1,x2-x1)) #angle
            
            '''updating network'''
            prev_trans=activityDecoding(prev_weights[0][:],num_links[0],N[0],neurons[0][:])
                # prev_trans=np.argmax(prev_weights[0][:])
            prev_angle=activityDecoding(prev_weights[1][:],num_links[1],N[1],neurons[1][:])
            num=3
            for n in range(num):

                
                for j in range(len(delta)):
                    net=attractorNetwork(delta[j],N[j],num_links[j],excite[j], activity_mag[j],inhibit_scale[j])
                    prev_weights[j][:]= net.update_weights_dynamics(prev_weights[j][:])
                    
            ax1.set_title("2D Attractor Network")
            # im=np.tile(prev_weights[0][:],(N[0],1)).T*np.tile(prev_weights[1][:],(N[1],1))
            im=np.outer(prev_weights[0][:],prev_weights[1][:])
            ax1.imshow(im)
    
            '''decoding mangnitude and direction of movement'''
            trans=(activityDecoding(prev_weights[0][:],num_links[0],N[0],neurons[0][:])-prev_trans)/num
            # trans=np.argmax(prev_weights[0][:])-prev_trans
            angle=np.deg2rad((activityDecoding(prev_weights[1][:],num_links[1],N[1],neurons[1][:])-prev_angle)/num)

            curr_parameter[0]=curr_parameter[0] + (trans*np.cos(angle))
            curr_parameter[1]=curr_parameter[1]+ (trans*np.sin(angle))
            # curr_z=curr_z+del_z

            # print(delta1, delta2, del_y,del_x)
            ax2.set_title("Decoded Pose")
            ax2.scatter(curr_parameter[0], curr_parameter[1],c='b',s=15)
            ax2.set_xlim([-100,100])
            ax2.set_ylim([-100,100])
            # ax2.set_zlim([0,N])


            print(str(delta[0])+"__"+str( delta[1])+ "------"+str(trans )+"__"+str(np.rad2deg(angle)))
            # print(x2-x1, y2-y1)
            # print(len(signal.find_peaks(prev_weights_trans)[0]),len(signal.find_peaks(prev_weights_angle)[0]) )
            

    ani = FuncAnimation(fig, animate, interval=1,frames=len(data_x),repeat=False)
    plt.show()

def encodingDecodingMotion(data_x,data_y):
    global prev_weights_trans,prev_weights_angle, num_links, excite, activity_mag,inhibit_scale

    '''Initalise network'''            
    current,prediction, velocity=[],[],[]
    delta1=0 #y_axis
    delta2=0 #x_axis
    print(delta1,delta2)
    # [1,:]ev_weights_angle[net.activation(delta2)]=net.full_weights(num_links)
    # prev_weights_z[net.activation(int(delta3))]=net.full_weights(num_links)

    curr_x,curr_y=np.zeros((len(data_x))), np.zeros((len(data_y)))
    for i in range(len(data_x)):
        if i>=1:
            '''distributed weights with excitations and inhibitions'''
            x1=data_x[i-1]
            x2=data_x[i]
            y1=data_y[i-1]
            y2=data_y[i]    
            delta1=np.sqrt(((x2-x1)**2)+((y2-y1)**2)) #translation
            delta2=np.rad2deg(np.arctan2(y2-y1,x2-x1)) #angle

            #angle and translation before shift
            prev_angle=activityDecoding(prev_weights_angle,num_links)
            prev_trans=activityDecoding(prev_weights_trans,num_links)


            net=attractorNetwork(delta1,delta2,N,num_links,int(excite),activity_mag,inhibit_scale)
            Neurons1,Neurons2=net.neuron_update(prev_weights_trans,prev_weights_angle)
            prev_weights_trans= net.update_weights_dynamics(prev_weights_trans,Neurons1,delta1)
            prev_weights_angle= net.update_weights_dynamics(prev_weights_angle,Neurons2,delta2)
            prev_weights_trans[prev_weights_trans<0]=0
            prev_weights_angle[prev_weights_angle<0]=0

            #angle and trasnlation after shift 
            trans=activityDecoding(prev_weights_trans,num_links)-prev_trans
            angle=np.deg2rad(activityDecoding(prev_weights_angle,num_links)-prev_angle)
            # update decoded posotion based on shift in network 
            curr_x[i]=curr_x[i-1] + (trans*np.cos(angle))
            curr_y[i]=curr_y[i-1]+( trans*np.sin(angle))


            print(str(delta1 )+"  "+str( delta2)+ "____________"+str(trans )+"  "+str(np.rad2deg(angle)))

    fig = plt.figure(figsize=(13, 4))
    ax0 = fig.add_subplot(1, 2, 1)
    ax1 = fig.add_subplot(1, 2, 2)
    ax1.set_title('Converted')
    ax1.scatter(curr_x, curr_y,c='b',s=15)
    # ax1.set_xlim([-300,300])
    # ax1.set_ylim([-100,500])

    ax0.set_title('Original')
    ax0.scatter(data_x, data_y,c='b',s=15)
    # ax0.set_xlim([-300,300])
    # ax0.set_ylim([-100,500])
    plt.show()





'''Test Area'''
sparse_gt=data_processing()#[0::2]
data_x=sparse_gt[:, :, 3][:,0]
data_y=sparse_gt[:, :, 3][:,2]

data_y=np.arange(100)
data_x=np.zeros((100))

visualise(data_x,data_y)
# print(data_y[2])

# data_y=data_y[500:1000]
# data_x=data_x[500:1000]
# encodingDecodingMotion(data_x,data_y)



# testing_Conversion(sparse_gt)


# print(np.shape(np.transpose(prev_weights[0][:])))

# print(np.shape(prev_weights[1][:]))

# print(np.outer(prev_weights[0][:],prev_weights[1][:]))