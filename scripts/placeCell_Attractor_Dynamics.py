import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import math 
# import pandas as pd
# import seaborn as sns
# import plotly.express as px
# import plotly.graph_objects as go

def col_round(x):
  frac = x - math.floor(x)
  if frac < 0.5: return math.floor(x)
  return math.ceil(x)

######################--VARABLES--############################
#simulation
sim_speed=10
iters=400

#network
N=390 #number of neurons
N_inc=360/N # number of degrees per neuron
neurons=np.arange(0,N)
curr_theta=0
curr_Neuron=0
iteration=1
prev_weights_x=np.zeros(N)
prev_weights_y=np.zeros(N)
# prev_weights=np.zeros((N,N))
lndmrk_confidence=1
input_error=0
decodingRadius=4 #meter
steady_state_val=10

#landmarks
inc=40 # increment angle for landmarks 
ang_rate=1 #deg per iteration for the robot
landmark_dect_toler=2


################################################################
class attractorNetwork:
    '''defines 1D attractor network with N neurons, angles associated with each neurons 
    along with inhitory and excitatory connections to update the weights'''
    def __init__(self, delta1, delta2,N,num_links,activity_mag):
        self.delta1=delta1
        self.delta2=delta2
        self.N=N  
        self.num_links=num_links
        self.activity_mag=activity_mag

    def neuron_update(self,prev_weights_x,prev_weights_y):
        indexes=np.arange(N)
        non_zero_idxs_x=indexes[prev_weights_x>0]
        non_zero_idxs_y=indexes[prev_weights_y>0]
        return (non_zero_idxs_x+self.delta1) % N, (non_zero_idxs_y+self.delta2) % N

        
    def inhibitions(self,id):
        ''' each nueuron inhibits all other nueurons but itself'''
        return np.delete(np.arange(self.N),self.excitations(id))

    def excitations(self,id):
        '''each neuron excites itself and num_links neurons left and right with wraparound connections'''
        excite=[]
        for i in range(-self.num_links,self.num_links+1):
            excite.append((id + i) % N)
        return np.array(excite)
         
    def full_weights(self):
        x=np.arange(-self.num_links,self.num_links+1)
        y=np.arange(-self.num_links,self.num_links+1)
        return 1/(np.std(x) * np.sqrt(2 * np.pi)) * np.exp( - (x - np.mean(x))**2 / (2 * np.std(x)**2))  

    def update_weights_dynamics(self,prev_weights,neurons):
        indexes,non_zero_weights,non_zero_weights_shifted, inhbit_val=np.arange(N),np.zeros(N),np.zeros(N),0

        '''copied and shifted activity'''
        non_zero_idxs=indexes[prev_weights>0] # indexes of non zero prev_weights
        
        non_zero_weights[non_zero_idxs]=prev_weights[non_zero_idxs] 
        
        non_zero_weights_shifted[neurons]=prev_weights[non_zero_idxs] #non zero weights shifted by delta
        
        intermediate_activity=non_zero_weights_shifted+non_zero_weights

        '''inhibition'''
        for i in range(len(non_zero_weights_shifted)):
            inhbit_val+=non_zero_weights_shifted[i]*inhibit_scale
        
        '''excitation'''
        excitations_store=np.zeros((len(non_zero_idxs),N))
        excitation_array,excite=np.zeros(N),np.zeros(N)
        for i in range(len(non_zero_idxs)):
            excitation_array[self.excitations(non_zero_idxs[i])]=self.full_weights()*prev_weights[non_zero_idxs[i]]
            excitations_store[i,:]=excitation_array
            excite[self.excitations(non_zero_idxs[i])]+=self.full_weights()*prev_weights[non_zero_idxs[i]]

        prev_weights+=(non_zero_weights_shifted+excite-inhbit_val)
        # print(prev_weights)
        return prev_weights/np.linalg.norm(prev_weights), non_zero_weights, non_zero_weights_shifted, intermediate_activity,[inhbit_val]*N, excitations_store

def activityDecoding(prev_weights,radius):
    '''Isolating activity at a radius around the peak to decode position'''
    peak=np.argmax(prev_weights) 
    local_activity=np.zeros(N)
    local_activity_idx=[]
    for i in range(-radius,radius+1):
        local_activity_idx.append((peak + i) % N)
    local_activity[local_activity_idx]=prev_weights[local_activity_idx]
    x,y=local_activity*np.cos(np.deg2rad(neurons*360/N)), local_activity*np.sin(np.deg2rad(neurons*360/N))
    vect_sum=np.round(np.rad2deg(np.arctan2(sum(y),sum(x))), 10)
    # changing range from [-179, 180] to [0,360]
    if vect_sum<0:
        shifted_vec=vect_sum+360
    else:
        shifted_vec=vect_sum
    return shifted_vec*(N/360)

def plotting_CAN_dynamics(activity_mag,delta1,delta2):
    # fig = plt.figure(figsize=(8,8))
    # gs = fig.add_gridspec(6,1)
    # ax1 = fig.add_subplot(gs[0, :])
    # ax2 = fig.add_subplot(gs[1, :])
    # ax3 = fig.add_subplot(gs[2,:])
    # ax4 =  fig.add_subplot(gs[3,:])
    # ax5 =  fig.add_subplot(gs[4,:])
    # ax6 =  fig.add_subplot(gs[5,:])
    # fig.tight_layout()


    fig1 = plt.figure(figsize=(7, 7))
    gs = fig1.add_gridspec(12,12)
    ax0 = plt.subplot(gs[4:12, 0:8])
    axx = plt.subplot(gs[0:3, 0:9])
    axy = plt.subplot(gs[3:12, 9:12])
    fig1.tight_layout()

    current,prediction, velocity=[],[],[]
    net=attractorNetwork(delta1,delta2,N,num_links,activity_mag)
    prev_weights_x[net.excitations(delta1)]=net.full_weights()
    prev_weights_y[net.excitations(delta2)]=net.full_weights()
    def animate(i):
        global prev_weights_x,prev_weights_y,lndmrk_confidence, curr_Neuron,iteration
        # ax1.clear(), ax2.clear(), ax3.clear(), ax4.clear(), ax5.clear(), ax6.clear()
        if i>0:
            '''distributed weights with excitations and inhibitions'''
            net=attractorNetwork(delta1,delta2,N,num_links,activity_mag)
            Neurons1,Neurons2=net.neuron_update(prev_weights_x,prev_weights_y)
            prev_weights_x,activity, activity_shifted,intermediate_activity,inhbit_val, excitations_store= net.update_weights_dynamics(prev_weights_x,Neurons1)
            prev_weights_y,activity, activity_shifted,intermediate_activity,inhbit_val, excitations_store= net.update_weights_dynamics(prev_weights_y,Neurons2)
            
            '''ax1.set_title("Network Activity Shifting by " + str(delta) + " Neurons")
            # ax1.bar(neurons, activity,width=0.9,color='green')
            # ax1.set_ylim([-0.2,0.3])
                
            # ax2.bar(neurons, activity_shifted, width=0.9,color='blue')
            # ax2.set_ylim([0,0.3])
            ax2.set_title('Shifted Copy')

            # ax3.bar(neurons, activity,width=0.9,color='green')
            # ax3.bar(neurons, activity_shifted, width=0.9,color='blue')
            # ax3.bar(neurons, intermediate_activity, width=0.9)
            # ax3.set_ylim([0,0.3])
            ax3.set_title('Sum of Activity and Shifted Copy')

            # ax4.bar(neurons,inhbit_val,width=0.9,color='purple')
            # ax3.set_ylim([-0.5,0])
            ax4.set_title('Inhibition')

            for i in range(len(excitations_store)):
                # ax5.bar(np.arange(N),excitations_store[i])
                ax5.set_title('Excitation')'''

            prev_weights_x[prev_weights_x<0]=0
            prev_weights_y[prev_weights_y<0]=0
            ax0.clear(), axx.clear(), axy.clear()
            ax0.set_title("2D Activity")
            ax0.imshow(np.tile(prev_weights_x,(N,1)).T*np.tile(prev_weights_y,(N,1)))
            axx.bar(neurons,prev_weights_x,width=1)
            axy.barh(neurons,prev_weights_y,height=1)
           

        
       
    '''animation for driving in a circle'''
    ani = FuncAnimation(fig1, animate, frames=iters, interval= sim_speed, repeat=False)
    plt.show() 

'''Testing''' 
inhibit_scale=0.01
num_links=80
plotting_CAN_dynamics(1,10,10)