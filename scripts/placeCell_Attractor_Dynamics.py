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
N=[50,50] #number of neurons
neurons=[np.arange(0,N[0]), np.arange(0,N[1])]
curr_Neuron=[0,0]
prev_weights=[np.zeros(N[0]), np.zeros(N[1])]
# prev_weights_z=np.zeros(N)
num_links=[10,10]
activity_mag=[1,1]
curr_parameter=[0,0]

neurons=np.arange(0,N[0])
curr_Neuron=0
curr_x,curr_y=[0,0]

inhibit_scale=0.05
excite_radius=4


#simulation
sim_speed=10
iters=400
pause=False
resetDone=True

#landmarks
inc=40 # increment angle for landmarks 
ang_rate=1 #deg per iteration for the robot
landmark_dect_toler=2


################################################################

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
        
        intermediate_activity=non_zero_weights_shifted+non_zero_weights

        '''inhibition'''
        for i in range(len(non_zero_weights_shifted)):
            inhbit_val+=non_zero_weights_shifted[i]*self.inhibit_scale
        
        '''excitation'''
        excitations_store=np.zeros((len(non_zero_idxs),self.N))
        excitation_array,excite=np.zeros(self.N),np.zeros(self.N)
        for i in range(len(non_zero_idxs)):
            excitation_array[self.excitations(non_zero_idxs[i])]=self.full_weights(self.excite_radius)*prev_weights[non_zero_idxs[i]]
            excitations_store[i,:]=excitation_array
            excite[self.excitations(non_zero_idxs[i])]+=self.full_weights(self.excite_radius)*prev_weights[non_zero_idxs[i]]

        prev_weights+=(non_zero_weights_shifted+excite-inhbit_val)
        # print(prev_weights)
        return prev_weights/np.linalg.norm(prev_weights), non_zero_weights, non_zero_weights_shifted, intermediate_activity,[inhbit_val]*self.N, excitations_store


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
    # # changing range from [-179, 180] to [0,360]
    # if vect_sum<0:
    #     shifted_vec=vect_sum+360
    # else:
    #     shifted_vec=vect_sum
    # return shifted_vec*(N/360)
    return vect_sum

def plotting_decomposed_CAN(ax1,ax2,ax3,delta, activity, activity_shifted, intermediate_activity, inhbit_val, excitations_store,N):
    ax1.set_title("Network Activity Shifting by " + str(delta) + " Neurons")
    ax1.bar(neurons, activity, color= '#C79FEF')
    ax1.axis('off')
    # ax1.set_ylim([-0.2,0.3])
        
    ax2.bar(neurons, activity_shifted, color= '#04D8B2')
    # ax2.set_ylim([0,0.3])
    ax2.set_title('Shifted Copy')
    ax2.axis('off')

    # ax3.bar(neurons, activity,width=0.9,color='green')
    # ax3.bar(neurons, activity_shifted, width=0.9,color='blue')
    # ax3.bar(neurons, intermediate_activity, width=0.9, color='red')
    # # ax3.set_ylim([0,0.3])
    # ax3.set_title('Sum of Activity and Shifted Copy')

    # ax4.bar(neurons,inhbit_val,width=0.9,color='purple')
    # # ax3.set_ylim([-0.5,0])
    # ax4.set_title('Inhibition')

    for i in range(len(excitations_store)):
        ax3.bar(np.arange(N),excitations_store[i])
        ax3.set_title('Excitation')
        ax3.axis('off')


def plotting_CAN_dynamics(delta1,delta2):
    fig1 = plt.figure(figsize=(11, 6))
    # ax0 =  plt.subplot2grid(shape=(3, 5), loc=(1, 2), rowspan=2,colspan=2)
    # axx =  plt.subplot2grid(shape=(3, 5), loc=(0, 2), colspan=2)
    # axy =  plt.subplot2grid(shape=(3, 5), loc=(1, 4), rowspan=2)

    # axy1 = plt.subplot2grid(shape=(3, 5), loc=(0, 0), colspan=2)
    # axy2 = plt.subplot2grid(shape=(3, 5), loc=(1, 0), colspan=2)
    # axy3 = plt.subplot2grid(shape=(3, 5), loc=(2, 0), colspan=2)
   
    gs = fig1.add_gridspec(32,32)
    #place cell
    ax0 = plt.subplot(gs[8:24, 12:20])
    axx = plt.subplot(gs[3:8, 12:20])
    axy = plt.subplot(gs[8:24, 20:23])
    
    #deconstructed CANN
    axy1 = plt.subplot(gs[0:7, 0:10])
    axy2 = plt.subplot(gs[10:17, 0:10])
    axy3 = plt.subplot(gs[20:27, 0:10])

    ax4 = plt.subplot(gs[9:23, 24:32])

    # axx1 = plt.subplot(gs[0:3, 26:36])
    # axx2 = plt.subplot(gs[6:9, 26:36])
    # axx3 = plt.subplot(gs[12:15, 26:36])
    # axx4 = plt.subplot(gs[18:21, 26:36])
    # axx5 = plt.subplot(gs[24:27, 26:36])

    plt.subplots_adjust(bottom=0.1)
    # fig1.tight_layout()

    '''Slider for Parameters'''
    button_ax = plt.axes([.05, .05, .05, .04]) # x, y, width, height
    button2_ax = plt.axes([.05, .12, .05, .04]) # x, y, width, height
    exciteax = plt.axes([0.25, 0.15, 0.65, 0.03])
    delta1ax = plt.axes([0.25, 0.1, 0.65, 0.03])
    delta2ax = plt.axes([0.25, 0.05, 0.65, 0.03])
    inhax = plt.axes([0.25, 0.0, 0.65, 0.03])
    # Create a slider from 0.0 to 20.0 in axes axfreq with 3 as initial value
    start_stop=wig.Button(button_ax,label='$\u25B6$')
    reset=wig.Button(button2_ax,'Reset')
    inhibit_scale=wig.Slider(inhax, 'Scale of Inhibition', 0, 0.05, 0.01)
    excite = wig.Slider(exciteax, 'Excitation Radius', 1, 40, 5, valstep=1)
    delta1 = wig.Slider(delta1ax, 'Delta 1', -10, 10, 0, valstep=1)
    delta2 = wig.Slider(delta2ax, 'Delta 2', -10, 10, 0, valstep=1)

    '''Initalise network'''            
    delta=[int(delta1.val),int(delta2.val)]
    for i in range(len(delta)):
        net=attractorNetwork(delta[i],N[i],num_links[i],int(excite.val), activity_mag[i],inhibit_scale.val)
        prev_weights[i][net.activation(delta[i])]=net.full_weights(num_links[i])
        prev_weights[i][prev_weights[i][:]<0]=0


    def animate(i):
        global prev_weights, num_links, activity_mag,N, curr_x, curr_y
        # ax1.clear(), ax2.clear(), ax3.clear(), ax4.clear(), ax5.clear(), ax6.clear()
        
        if not pause and resetDone:
            ax0.clear(),
            axx.clear(), axy.clear(),
            axy1.clear(), axy2.clear(), axy3.clear(), 
            #  axx1.clear(), axx2.clear(), axx3.clear(), axx4.clear(), axx5.clear()
            prev_y=np.argmax(prev_weights[1][:])
            prev_x=np.argmax(prev_weights[0][:])
            '''distributed weights with excitations and inhibitions'''
            delta=[int(delta1.val),int(delta2.val)]
            for j in range(len(delta)):
                net=attractorNetwork(delta[j],N[j],num_links[j],int(excite.val), activity_mag[j],inhibit_scale.val)
                prev_weights[j][:],activity, activity_shifted,intermediate_activity,inhbit_val, excitations_store= net.update_weights_dynamics(prev_weights[j][:])
                prev_weights[j][prev_weights[j][:]<0]=0
            
            # ax0.set_title("2D Attractor Network")
            ax0.imshow(np.tile(prev_weights[0][:],(N[0],1)).T*np.tile(prev_weights[1][:],(N[1],1)))
            axy.invert_yaxis()
            axx.bar(neurons,prev_weights[1][:],width=1)
            axx.axis('off')
            axy.barh(neurons,prev_weights[0][:],height=1)
            axy.axis('off')

            del_y=np.argmax(prev_weights[0][:])-prev_x
            del_x=np.argmax(prev_weights[1][:])-prev_y
            curr_x=curr_x+del_x
            curr_y=curr_y+del_y

            ax4.scatter(np.argmax(prev_weights[1][:]), np.argmax(prev_weights[0][:]),c='b',s=15)
            ax4.set_xlim([0,N[0]])
            ax4.set_ylim([0,N[1]])
            ax4.invert_yaxis()



            # plotting_decomposed_CAN(axx1,axx2,axx3,axx4,axx5,delta1.val, activity_x, activity_shifted_x, intermediate_activity_x, inhbit_val_x, excitations_store_x)
            # print(delta1.val,activityDecoding(prev_weights_x,num_links)-prev_x)
            plotting_decomposed_CAN(axy1,axy2,axy3,delta2.val, activity, activity_shifted, intermediate_activity, inhbit_val, excitations_store,N[-1])
   

    def update(val):
        global prev_weights, num_links, activity_mag
        '''distributed weights with excitations and inhibitions'''
        delta=[int(delta1.val),int(delta2.val)]
        for j in range(len(delta)):
            net=attractorNetwork(delta[j],N[j],num_links[j],int(excite.val), activity_mag[j],inhibit_scale.val)
            prev_weights[j][:],activity, activity_shifted,intermediate_activity,inhbit_val, excitations_store= net.update_weights_dynamics(prev_weights[j][:])
            prev_weights[j][prev_weights[j][:]<0]=0

    def onClick(event):
        global pause, prev_weights, resetDone 
        (xm,ym),(xM,yM) = start_stop.label.clipbox.get_points()
        if xm < event.x < xM and ym < event.y < yM:
            pause ^= True

        (xn,yn),(xN,yN) = reset.label.clipbox.get_points()
        if xn < event.x < xN and yn < event.y < yN:
            delta=[0,0]
            ax4.clear()
            for j in range(len(delta)):
                prev_weights[j][:]=np.zeros(N[j])
            
            for i in range(len(delta)):
                net=attractorNetwork(delta[i],N[i],num_links[i],int(excite.val), activity_mag[i],inhibit_scale.val)
                prev_weights[i][net.activation(delta[i])]=net.full_weights(num_links[i])
                prev_weights[i][prev_weights[i][:]<0]=0
            resetDone ^= False
            pause ^=True


        
  

    '''animation for Place Cells'''
    excite.on_changed(update)
    delta1.on_changed(update)
    delta2.on_changed(update)
    inhibit_scale.on_changed(update)
    fig1.canvas.mpl_connect('button_press_event', onClick)
    ani = FuncAnimation(fig1, animate, frames=iters)
    plt.show() 

'''Testing''' 

plotting_CAN_dynamics(2,4)