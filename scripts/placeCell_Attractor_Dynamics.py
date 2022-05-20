import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.widgets as wig
import mpl_toolkits.axes_grid1

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
pause=False

#network


#landmarks
inc=40 # increment angle for landmarks 
ang_rate=1 #deg per iteration for the robot
landmark_dect_toler=2


################################################################
class attractorNetwork:
    '''defines 1D attractor network with N neurons, angles associated with each neurons 
    along with inhitory and excitatory connections to update the weights'''
    def __init__(self, delta1, delta2,N,num_links, excite_radius, activity_mag,inhibit_scale):
        self.delta1=delta1
        self.delta2=delta2
        self.excite_radius=excite_radius
        self.N=N  
        self.num_links=num_links
        self.activity_mag=activity_mag
        self.inhibit_scale=inhibit_scale

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
        
        intermediate_activity=non_zero_weights_shifted+non_zero_weights

        '''inhibition'''
        for i in range(len(non_zero_weights_shifted)):
            inhbit_val+=non_zero_weights_shifted[i]*self.inhibit_scale
        
        '''excitation'''
        excitations_store=np.zeros((len(non_zero_idxs),N))
        excitation_array,excite=np.zeros(N),np.zeros(N)
        for i in range(len(non_zero_idxs)):
            excitation_array[self.excitations(non_zero_idxs[i])]=self.full_weights(self.excite_radius)*prev_weights[non_zero_idxs[i]]
            excitations_store[i,:]=excitation_array
            excite[self.excitations(non_zero_idxs[i])]+=self.full_weights(self.excite_radius)*prev_weights[non_zero_idxs[i]]

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
    fig1 = plt.figure(figsize=(7, 6))
   
    gs = fig1.add_gridspec(15,12)
    ax0 = plt.subplot(gs[4:12, 0:8])
    axx = plt.subplot(gs[0:3, 0:8])
    axy = plt.subplot(gs[4:12, 9:12])
    # plt.subplots_adjust(bottom=0.3)
    fig1.tight_layout()

    '''Slider for Parameters'''
    button_ax = plt.axes([.05, .05, .05, .04]) # x, y, width, height
    exciteax = plt.axes([0.25, 0.15, 0.65, 0.03])
    delta1ax = plt.axes([0.25, 0.1, 0.65, 0.03])
    delta2ax = plt.axes([0.25, 0.05, 0.65, 0.03])
    inhax = plt.axes([0.25, 0.0, 0.65, 0.03])
    # Create a slider from 0.0 to 20.0 in axes axfreq with 3 as initial value
    start_stop=wig.Button(button_ax,label='$\u25B6$')
    inhibit_scale=wig.Slider(inhax, 'Scale of Inhibition', 0, 0.05, 0.01)
    excite = wig.Slider(exciteax, 'Excitation Radius', 1, 40, 5, valstep=1)
    delta1 = wig.Slider(delta1ax, 'Delta 1', 1, N, 1, valstep=1)
    delta2 = wig.Slider(delta2ax, 'Delta 2', 1, N, 1, valstep=1)

    '''Initalise network'''                
    current,prediction, velocity=[],[],[]
    net=attractorNetwork(int(delta1.val),int(delta2.val),N,num_links,int(excite.val), activity_mag,inhibit_scale.val)
    prev_weights_x[net.activation(int(delta1.val))]=net.full_weights(num_links)
    prev_weights_y[net.activation(int(delta2.val))]=net.full_weights(num_links)

    def animate(i):
        global prev_weights_x,prev_weights_y, num_links
        # ax1.clear(), ax2.clear(), ax3.clear(), ax4.clear(), ax5.clear(), ax6.clear()
       
        if not pause:
            ax0.clear(), axx.clear(), axy.clear()
            '''distributed weights with excitations and inhibitions'''
            net=attractorNetwork(int(delta1.val),int(delta2.val),N,num_links,int(excite.val),activity_mag,inhibit_scale.val)
            Neurons1,Neurons2=net.neuron_update(prev_weights_x,prev_weights_y)
            prev_weights_x,activity, activity_shifted,intermediate_activity,inhbit_val, excitations_store= net.update_weights_dynamics(prev_weights_x,Neurons1)
            prev_weights_y,activity, activity_shifted,intermediate_activity,inhbit_val, excitations_store= net.update_weights_dynamics(prev_weights_y,Neurons2)
            prev_weights_x[prev_weights_x<0]=0
            prev_weights_y[prev_weights_y<0]=0

            
            ax0.set_title("2D Activity")
            axy.invert_yaxis()
            ax0.imshow(np.tile(prev_weights_x,(N,1)).T*np.tile(prev_weights_y,(N,1)))
            axx.bar(neurons,prev_weights_y,width=1)
            axy.barh(neurons,prev_weights_x,height=1)
   

    def update(val):
        global prev_weights_x,prev_weights_y, num_links
        net=attractorNetwork(int(delta1.val),int(delta2.val),N,num_links,int(excite.val),activity_mag,inhibit_scale.val)
        Neurons1,Neurons2=net.neuron_update(prev_weights_x,prev_weights_y)
        prev_weights_x,activity, activity_shifted,intermediate_activity,inhbit_val, excitations_store= net.update_weights_dynamics(prev_weights_x,Neurons1)
        prev_weights_y,activity, activity_shifted,intermediate_activity,inhbit_val, excitations_store= net.update_weights_dynamics(prev_weights_y,Neurons2)
        prev_weights_x[prev_weights_x<0]=0
        prev_weights_y[prev_weights_y<0]=0   

    def onClick(event):
        global pause
        (xm,ym),(xM,yM) = start_stop.label.clipbox.get_points()
        if xm < event.x < xM and ym < event.y < yM:
            pause ^= True
  

    '''animation for Place Cells'''
    excite.on_changed(update)
    delta1.on_changed(update)
    delta2.on_changed(update)
    inhibit_scale.on_changed(update)
    fig1.canvas.mpl_connect('button_press_event', onClick)
    ani = FuncAnimation(fig1, animate, frames=iters)
    plt.show() 

'''Testing''' 
N=60 #number of neurons
neurons=np.arange(0,N)
curr_Neuron=0
prev_weights_x=np.zeros(N)
prev_weights_y=np.zeros(N)

# inhibit_scale=0.05
excite_radius=4
num_links=10

plotting_CAN_dynamics(1,2,4)