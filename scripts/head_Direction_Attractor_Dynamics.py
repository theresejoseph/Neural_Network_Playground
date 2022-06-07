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
N=60 #number of neurons
N_inc=360/N # number of degrees per neuron
neurons=np.arange(0,N)
curr_theta=0
curr_Neuron=0
iteration=1
prev_weights=np.zeros(N)
num_links=2
lndmrk_confidence=1
input_error=0
decodingRadius=4 #meter
inhibit_scale=0.05
steady_state_val=10

#landmarks
inc=40 # increment angle for landmarks 
ang_rate=1 #deg per iteration for the robot
landmark_dect_toler=2


################################################################
class attractorNetwork:
    '''defines 1D attractor network with N neurons, angles associated with each neurons 
    along with inhitory and excitatory connections to update the weights'''
    def __init__(self, delta, landmark,N,num_links,lndmrk_confidence,activity_mag):
        self.delta=delta 
        self.landmark=landmark
        self.N=N  
        self.num_links=num_links
        self.lndmrk_confidence=lndmrk_confidence
        self.activity_mag=activity_mag

    def theta_update(self,curr_theta):
        return (curr_theta+self.delta) % 360

    def neuron_update(self,curr_Neuron):
        return (curr_Neuron+self.delta) % N

    def theta_shift(self,shift):
        return shift 
        
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
        return 1/(np.std(x) * np.sqrt(2 * np.pi)) * np.exp( - (x - np.mean(x))**2 / (2 * np.std(x)**2))  

    def update_weights_dynamics(self,prev_weights,neuron):
        indexes,non_zero_weights,non_zero_weights_shifted, inhbit_val=np.arange(N),np.zeros(N),np.zeros(N),0

        '''copied and shifted activity'''
        non_zero_idxs=indexes[prev_weights>0] # indexes of non zero prev_weights
        
        non_zero_weights[non_zero_idxs]=prev_weights[non_zero_idxs] 
        
        non_zero_weights_shifted[(non_zero_idxs+self.delta)%N]=prev_weights[non_zero_idxs] #non zero weights shifted by delta
        
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
        
        # 
        # norm_prev_weights=(prev_weights - np.min(prev_weights)) / (np.max(prev_weights) - np.min(prev_weights))
        return prev_weights/np.linalg.norm(prev_weights), non_zero_weights, non_zero_weights_shifted, intermediate_activity,[inhbit_val]*N, excitations_store

def activity_center(prev_weights):
    '''Takes the previous weights as inputs and finds the activity center with pdf as polar coordinates'''
    indexes=np.arange(0,N)
    non_zero_idxs=indexes[prev_weights>0]
    non_zero_weights=np.zeros(N)
    if non_zero_idxs is not np.empty:
        non_zero_weights[non_zero_idxs]=prev_weights[non_zero_idxs]
        x,y=non_zero_weights*np.cos(np.deg2rad(neurons*360/N)), non_zero_weights*np.sin(np.deg2rad(neurons*360/N))
        vect_sum=np.round(np.rad2deg(np.arctan2(sum(y),sum(x))), 10)
        # changing range from [-179, 180] to [0,360]
        if vect_sum<0:
            shifted_vec=vect_sum+360
        else:
            shifted_vec=vect_sum
    return shifted_vec, x, y

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

def logLikelyhood(prev_weights, radius):
    peak=np.argmax(prev_weights) 
    local_activity=np.zeros(N)
    local_activity_idx=[]
    for i in range(-radius,radius+1):
        local_activity_idx.append((peak + i) % N)
    
    local_activity[local_activity_idx]=prev_weights[local_activity_idx]
    ll=np.zeros(N)
    sigma=1#np.std(local_activity_idx)
    mu=0#np.mean(local_activity_idx)
    for i in range(len(local_activity)):
        ll[i]=-np.log(sigma*np.sqrt(2*np.pi))-0.5*((local_activity[i]-mu)/sigma)**2
    
    return np.sum(ll)

def plotting_CAN_dynamics(activity_mag,delta):
    fig = plt.figure(figsize=(8,8))
    gs = fig.add_gridspec(6,1)
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, :])
    ax3 = fig.add_subplot(gs[2,:])
    ax4 =  fig.add_subplot(gs[3,:])
    ax5 =  fig.add_subplot(gs[4,:])
    ax6 =  fig.add_subplot(gs[5,:])
    fig.tight_layout()

    current,prediction, velocity=[],[],[]
    net=attractorNetwork(delta,None,N,num_links,lndmrk_confidence,activity_mag)
    prev_weights[net.excitations(delta)]=net.full_weights()
    def animate(i):
        global prev_weights,lndmrk_confidence, curr_Neuron,iteration
        ax1.clear(), ax2.clear(), ax3.clear(), ax4.clear(), ax5.clear(), ax6.clear()
        if i>0:
            '''distributed weights with excitations and inhibitions'''
            net=attractorNetwork(delta,None,N,num_links,lndmrk_confidence,activity_mag)
            curr_Neuron=net.neuron_update(curr_Neuron)
            prev_weights,activity, activity_shifted,intermediate_activity,inhbit_val, excitations_store= net.update_weights_dynamics(prev_weights,curr_Neuron)
            
            ax1.set_title("Network Activity Shifting by " + str(delta) + " Neurons")
            ax1.bar(neurons, activity,width=0.9,color='green')
            # ax1.set_ylim([-0.2,0.3])
                
            ax2.bar(neurons, activity_shifted, width=0.9,color='blue')
            # ax2.set_ylim([0,0.3])
            ax2.set_title('Shifted Copy')

            # ax3.bar(neurons, activity,width=0.9,color='green')
            # ax3.bar(neurons, activity_shifted, width=0.9,color='blue')
            ax3.bar(neurons, intermediate_activity, width=0.9)
            # ax3.set_ylim([0,0.3])
            ax3.set_title('Sum of Activity and Shifted Copy')

            ax4.bar(neurons,inhbit_val,width=0.9,color='purple')
            # ax3.set_ylim([-0.5,0])
            ax4.set_title('Inhibition')

            for i in range(len(excitations_store)):
                ax5.bar(np.arange(N),excitations_store[i])
                ax5.set_title('Excitation')

            ax6.set_title("Activity with Excitation and Inhibition")
            ax6.bar(neurons, prev_weights,width=0.9,color='pink')

            
            # neuron_pred=activityDecoding(prev_weights,decodingRadius)
            # current.append(abs(delta))
            # prediction.append(abs(neuron_pred))
            # ax5.set_title('Decoded Activity:'+ str(neuron_pred))
            # line1, =ax5.plot(np.arange(len(prediction)),np.array(prediction),'b.')
            # # line2, =ax3.plot(np.arange(len(current)),np.array(current),'r-')
            # ax5.legend([line1], ['prediction'])

            '''printing outputs'''
            # if len(prediction)>10 and (velocity[-steady_state_val:] == [velocity[-1]]*steady_state_val) and abs(velocity[-1]-delta)<1:
            #     print("done",velocity[-1], np.mean(velocity), len(velocity)) #[:-(steady_state_val-1)]
            # elif len(prediction)>1:
            #     if np.sign(np.round(prediction[-1]-prediction[-2],2)) != np.sign(delta):
            #         vel=np.round(prediction[-1]-prediction[-2]+N)
            #     else:
            #         vel=np.round(prediction[-1]-prediction[-2])
            #     velocity.append(vel)
            #     print(curr_Neuron, neuron_pred, velocity[-1])
       
    '''animation for driving in a circle'''
    ani = FuncAnimation(fig, animate, frames=iters, interval= sim_speed, repeat=False)
    plt.show() 

def CANdynamics(activity_mag,delta):
    global lndmrk_confidence
    prev_weights=np.zeros(N)
    prediction, velocity=[],[] 
    curr_Neuron=0
    print("start "+str(activity_mag),str(delta))
    while (1):
        net=attractorNetwork(delta,None,N,num_links,lndmrk_confidence,activity_mag)
        prev_weights,activity= net.update_weights_dynamics(prev_weights,curr_Neuron)
        curr_Neuron=net.neuron_update(curr_Neuron)
        neuron_pred=activityDecoding(prev_weights,decodingRadius)
            
        prediction.append(abs(neuron_pred))

        if len(prediction)>1:
            if np.sign(np.round(prediction[-1]-prediction[-2],2)) != np.sign(delta):
                vel=np.round(prediction[-1]-prediction[-2]+N)
            else:
                vel=np.round(prediction[-1]-prediction[-2])
            velocity.append(vel)

        if (len(prediction)>10) and (velocity[-steady_state_val:] == [velocity[-1]]*steady_state_val) and abs(velocity[-1]-delta)<1:
            print("done "+ str(velocity[-1]), str(np.mean(velocity))) #[:-(steady_state_val-1)])))
            break
    return velocity[-1] 
    # return np.mean(velocity) #[:-(steady_state_val-1)])
    # return (len(velocity)-steady_state_val)
 
# def heatmap(activity_mags,shifts,plotType):
#     dynamics=np.zeros((len(activity_mags),len(shifts)))
#     for i in range(len(activity_mags)):
#         for j in range(len(shifts)):
#             dynamics[i][j]=CANdynamics(activity_mags[i],shifts[j])
     
#     dynamics_df = pd.DataFrame(dynamics, 
#         index=[str(val) for val in activity_mags],
#         columns=[str(val) for val in shifts])

#     if plotType==0:
#         fig, ax = plt.subplots(figsize=(7, 7))
#         sns.heatmap(dynamics_df,linewidth=0.001,cbar_kws={'label': 'Steady State Velocity'})
#         ax.set_xlabel('Shifts [Neurons]')
#         ax.set_ylabel('Activity Magnitude')
#         plt.show()
    
#     elif plotType==1:
#         fig = go.Figure(data=[go.Surface(z=dynamics, x=shifts, y=activity_mags)])
#         fig.update_layout(title='Average Velocity of Activity', 
#         scene = dict(
#                     xaxis_title='Input Shift [Neurons]',
#                     yaxis_title='Magnitude of Activity',
#                     zaxis_title='Steady State Velocity'))
#         fig.show()



##### TEST AREA #####  
activity_mags=[0.05,0.1,0.25,0.5,1,2,3,5]
shifts = np.arange(0,60)
plot_Type=1
# heatmap(activity_mags,shifts,plot_Type)


plotting_CAN_dynamics(1,1)

# print(CANdynamics(1,2.5))

'''exictations'''
# indexes,non_zero_weights, inhbit_val=np.arange(0,N),np.zeros(N),0
# activity_mag=1
# delta=2
# net=attractorNetwork(None,N,inhbit_val,num_links,lndmrk_confidence,activity_mag)
# prev_weights[net.excitations(delta)]=net.full_weights()

# non_zero_idxs=indexes[prev_weights>0] # all non zero prev_weights
# non_zero_weights[(non_zero_idxs+delta)%N]=prev_weights[non_zero_idxs] #non zero weights shifted by delta
      

# excitations_store=np.zeros((len(non_zero_idxs),N))
# excitation_array=np.zeros(N)
# for i in range(len(non_zero_idxs)):
#     excitation_array[net.excitations(non_zero_idxs[i])]=net.full_weights()
#     excitations_store[i,:]=excitation_array
    
#     # excitations_store[i,curr_exite]=net.full_weights

# print(np.sum(excitations_store,axis=0))
# # excite=np.sum(excitations_store)
# for i in range(len(excitations_store)):
#     plt.bar(np.arange(N),excitations_store[i])
# plt.show()

'''all fractions at a fixed interval'''
# val=[]
# end=101
# frac =1/(end-1)
# for i in range(0,end):
#     delta=i*frac
#     curr_Neuron=0
#     activity_mag=1
#     net=attractorNetwork(delta,None,N,inhbit_val,num_links,lndmrk_confidence,activity_mag)
#     curr_Neuron=net.neuron_update(curr_Neuron)
#     prev_weights,activity= net.update_weights_dynamics(prev_weights,curr_Neuron)
#     # activity=net.frac_weights(curr_Neuron)
#     val.append(activityDecoding(prev_weights,num_links))
#     # print(np.mean(activity))

# # print(val)
# plt.plot(np.arange(0,end)*frac,val,'b.')
# plt.show()



'''one fraction at a time'''
# curr_Neuron=30
# delta=0.8
# activity_mag=1
# net=attractorNetwork(delta,None,N,inhbit_val,num_links,lndmrk_confidence,activity_mag)
# curr_Neuron=net.neuron_update(curr_Neuron)
# prev_weights,activity= net.update_weights_dynamics(prev_weights,curr_Neuron)


# print(activityDecoding(activity,num_links))
# print(logLikelyhood(prev_weights,num_links))
# plt.bar(np.arange(len(prev_weights)),prev_weights)
# plt.show()




###############code cemetary#######################
# def circularSubtraction(a,b):
#     x= np.cos(np.deg2rad(np.array([a,b])*360/N))
#     y= np.sin(np.deg2rad(np.array([a,b])*360/N))
#     vect_sub = np.round(np.rad2deg(np.arctan2((y[0]-y[1]),(x[0]-x[1]))), 10)
#     if vect_sub<0:
#             shifted_vec=(vect_sub+360)*(N/360)
#     else:
#         shifted_vec=vect_sub*(N/360)
#     return shifted_vec




    # def frac_weights(self,neuron):
    #     decimal=neuron %1
    #     if decimal!=0:
    #         if (1/decimal)%1 != 0:
    #             multiplier=int(np.round((1/decimal)*10)) #acuuracy to 1 decimal places
    #         else:
    #             multiplier=int((1/decimal))
            
    #         reso=(self.num_links*2+1)#*multiplier
    #         hiRes_Neurons=np.arange(-self.num_links,self.num_links+1)
            
    #         activity=1/(np.std(hiRes_Neurons) * np.sqrt(2 * np.pi)) * np.exp( - (hiRes_Neurons - decimal)**2 / (2 * np.std(hiRes_Neurons)**2))
    #         # activity = np.add.reduceat(hiRes_activity, np.arange(0, len(hiRes_activity), multiplier))
    #     else:
    #         activity=self.full_weights()
    #     return activity

'''fractional shfits - '''
    # def frac_weights(self,neuron):
    #     decimal=neuron %1
    #     if decimal!=0:
    #         if (1/decimal)%1 != 0:
    #             multiplier=int(np.round((1/decimal)*10)) #acuuracy to 1 decimal places
    #         else:
    #             multiplier=int((1/decimal))
            
    #         reso=((self.num_links*2)+1)*multiplier
    #         hiRes_Neurons=np.arange(-(reso/2),(reso/2))
            
    #         hiRes_activity=1/(np.std(hiRes_Neurons) * np.sqrt(2 * np.pi)) * np.exp( - (hiRes_Neurons - 1)**2 / (2 * np.std(hiRes_Neurons)**2))
    #         activity = np.add.reduceat(hiRes_activity, np.arange(0, len(hiRes_activity), multiplier))
    #     else:
    #         activity=self.full_weights()
    #     return activity
    
    # def frac_weights(self,neuron):
    #     decimal=neuron %1
    #     reso=np.arange((self.num_links*2)+1)
    #     Neurons=np.arange(-self.num_links,self.num_links+1)
    #     activity=1/(np.std(Neurons) * np.sqrt(2 * np.pi)) * np.exp( - (Neurons - 1)**2 / (2 * np.std(Neurons)**2))
    #     if decimal!=0:
    #         activity=activity*decimal
    #         # activity[self.num_links+1]=activity[self.num_links+1]*(1-decimal)
    #     # elif decimal!=0 and decimal < 0.5:
    #     #     # activity[self.num_links]=activity[self.num_links]*decimal+1
    #     #     activity[self.num_links+1]=activity[self.num_links+1]*((1-decimal))

    #     return activity