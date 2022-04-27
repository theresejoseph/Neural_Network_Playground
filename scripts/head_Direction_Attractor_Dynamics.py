import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import math 
import pandas as pd
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

def col_round(x):
  frac = x - math.floor(x)
  if frac < 0.5: return math.floor(x)
  return math.ceil(x)

######################--VARABLES--############################
#simulation
sim_speed=100
iters=400

#network
N=60 #number of neurons
N_inc=360/N # number of degrees per neuron
neurons=np.arange(0,N)
curr_theta=0
iteration=1
delta=1
prev_weights=np.zeros(N)
inhbit_val=0.01
num_links=1
lndmrk_confidence=1
input_error=0

#landmarks
inc=40 # increment angle for landmarks 
ang_rate=1 #deg per iteration for the robot
radius=1.8  #meter
landmark_dect_toler=2


################################################################
class attractorNetwork:
    '''defines 1D attractor network with N neurons, angles associated with each neurons 
    along with inhitory and excitatory connections to update the weights'''
    def __init__(self, delta, landmark,N,inhibit_val,num_links,lndmrk_confidence,activity_mag):
        self.delta=delta 
        self.landmark=landmark
        self.N=N  
        self.inhibit_val=inhibit_val
        self.num_links=num_links
        self.lndmrk_confidence=lndmrk_confidence
        self.activity_mag=activity_mag

    def theta_update(self,curr_theta):
        return (curr_theta+self.delta) % 360

    def theta_shift(self,shift):
        return shift 
        
    def inhibitions(self,id):
        ''' each nueuron inhibits all other nueurons but itself'''
        return np.delete(np.arange(self.N),id)

    def excitations(self,id):
        '''each neuron excites itself and num_links neurons left and right with wraparound connections'''
        excite=[]
        for i in range(-self.num_links,self.num_links+1):
            excite.append((id + i) % N)
        return np.array(excite)

    def frac_weights(self):
        " returns weights for the exitation connections with a normal disrtibution"
        if (360/N)-round(360/N) ==0:
            resolution=(self.num_links*2+1)*(360/N)
            iteration =curr_theta % (360/N)

            x_angles=np.arange(-resolution/2,(resolution/2))
            distr_angles=1/(np.std(x_angles) * np.sqrt(2 * np.pi)) * np.exp( - (x_angles - iteration)**2 / (2 * np.std(x_angles)**2))
            return np.add.reduceat(distr_angles, np.arange(0, len(distr_angles), round(360/N)))
        else: 
            return self.full_weights()

    def full_weights(self):
        x=np.arange(-self.num_links,self.num_links+1)
        return 1/(np.std(x) * np.sqrt(2 * np.pi)) * np.exp( - (x - np.mean(x))**2 / (2 * np.std(x)**2))  

    def update_weights_dynamics(self,prev_weights,theta):
        if col_round(theta*(N/360))==N:
            Neuron=0
        else:
            Neuron= col_round(theta*(N/360))
        prev_weights[self.excitations(Neuron)]+=self.full_weights()*self.activity_mag
        prev_weights[self.inhibitions(Neuron)]-=self.inhibit_val
        
        activity=np.zeros(N)
        activity[self.excitations(Neuron)]=self.full_weights()*self.activity_mag
        return (prev_weights/np.linalg.norm(prev_weights)), activity

def activity_center(prev_weights):
    '''Takes the previous weights as inputs and finds the activity center with pdf as polar coordinates'''
    indexes=np.arange(0,N)
    non_zero_idx=indexes[prev_weights>0]
    non_zero_weights=np.zeros(N)
    if non_zero_idx is not np.empty:
        non_zero_weights[non_zero_idx]=prev_weights[non_zero_idx]
        x,y=non_zero_weights*np.cos(np.deg2rad(neurons*360/N)), non_zero_weights*np.sin(np.deg2rad(neurons*360/N))
        vect_sum=np.round(np.rad2deg(np.arctan2(sum(y),sum(x))), 10)
        # changing range from [-179, 180] to [0,360]
        if vect_sum<0:
            shifted_vec=vect_sum+360
        else:
            shifted_vec=vect_sum
    return shifted_vec, x, y

def plotting_CAN_velocity():
    fig, (ax1) = plt.subplots()
   
    def animate(i):
        global prev_weights, inhbit_val,lndmrk_confidence, curr_theta
        '''distributed weights with excitations and inhibitions'''
        net=attractorNetwork(delta,None,N,inhbit_val,num_links,lndmrk_confidence)
        curr_theta=net.theta_update(curr_theta)
        prev_weights,activity=net.update_weights_dynamics(prev_weights,curr_theta)
        
        ax1.clear()
        ax1.bar(neurons*(360/N), prev_weights,width=2)
        ax1.set_ylim([-0.2,0.6])

        '''printing outputs'''
        theta_pred,x,y=activity_center(prev_weights)
        print(np.round(theta_pred), curr_theta)

    '''animation for driving in a circle'''
    ani = FuncAnimation(fig, animate, frames=iters, interval= 1000, repeat=False)
    plt.show() 

def plotting_CAN_dynamics(activity_mag,shift):
    fig = plt.figure(figsize=(7,5))
    gs = fig.add_gridspec(3,1)
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, :])
    ax3 = fig.add_subplot(gs[2,:])
    fig.tight_layout()
    current,prediction, velocity=[],[],[]
    def animate(i):
        global prev_weights, inhbit_val,lndmrk_confidence , curr_theta,iteration
        ax1.clear(), ax2.clear()
        if i>0:
            '''distributed weights with excitations and inhibitions'''
            net=attractorNetwork(delta,None,N,inhbit_val,num_links,lndmrk_confidence,activity_mag)
            prev_weights,activity= net.update_weights_dynamics(prev_weights,curr_theta)
            
            if i%iteration ==0:
                curr_theta=net.theta_shift(shift)

            theta_pred,x,y=activity_center(prev_weights)
            current.append(abs(curr_theta))
            
            ax1.set_title("Shifting Forwards to " + str(curr_theta))
            ax1.bar(neurons*(360/N), prev_weights,width=5)
            #ax1.set_ylim([-0.2,0.4])
            ax2.bar(neurons*(360/N), activity, width=5)
            ax3.set_title('Input and Prediction Angle')
            line1, =ax3.plot(np.arange(len(prediction)),np.array(prediction),'b-')
            line2, =ax3.plot(np.arange(len(current)),np.array(current),'r-')
            ax3.legend([line1, line2], ['prediction', 'input'])

            '''printing outputs'''
            if (len(prediction)>10) and (prediction[-10:] == [prediction[-1]]*10):
                # for i in range(1,len(prediction)-9):
                #     velocity.append(prediction[i]-prediction[i-1])
                print("done",np.mean(prediction[:len(prediction)-9]), np.mean(velocity[:-8]))
                print(velocity[:-8])
            else:
                if len(prediction)>1:
                    velocity.append(prediction[-1]-prediction[-2])
                print(curr_theta, theta_pred)
                prediction.append(abs(theta_pred))
                
    '''animation for driving in a circle'''
    ani = FuncAnimation(fig, animate, frames=iters, interval= 1000, repeat=False)
    plt.show() 

def CANdynamics(activity_mag,shift,curr_theta):
    global inhbit_val,lndmrk_confidence
    prev_weights=np.zeros(N)
    prediction, velocity=[],[] 

    while (1):
        net=attractorNetwork(delta,None,N,inhbit_val,num_links,lndmrk_confidence,activity_mag)
        prev_weights,activity= net.update_weights_dynamics(prev_weights,curr_theta)
        curr_theta=net.theta_shift(shift)
        theta_pred,x,y=activity_center(prev_weights)
        prediction.append(abs(theta_pred))

        if len(prediction)>1:
            velocity.append(prediction[-1]-prediction[-2])

        if((len(prediction)>10) and (prediction[-10:] == [prediction[-1]]*10)):
            break

    return np.mean(velocity[:-9])
 
def heatmap(activity_mags,shifts,curr_theta,plotType,save):
    vals_3d=[]
    dynamics=np.zeros((len(activity_mags),len(shifts)))

    for i in range(len(activity_mags)):
        for j in range(len(shifts)):
            dynamics[i][j]=CANdynamics(activity_mags[i],shifts[j],curr_theta)
            vals_3d.append([i,j,dynamics[i][j]])
            
    dynamics_df = pd.DataFrame(dynamics, 
        index=[str(val) for val in activity_mags],
        columns=[str(val) for val in shifts])

    if plotType==0:
        fig, ax = plt.subplots(figsize=(7, 7))
        sns.heatmap(dynamics_df,linewidth=0.3)
        ax.set_xlabel('Shifts [degrees]')
        ax.set_ylabel('Activity Magnitude')
        plt.show()
    
    elif plotType==1:
        fig = go.Figure(data=[go.Surface(z=dynamics, x=shifts, y=activity_mags)])
        fig.update_layout(title='Average Velocity of Activity', 
        scene = dict(
                    xaxis_title='Input Shift (degrees)',
                    yaxis_title='Magnitude of Activity',
                    zaxis_title='Velocity of Activity (deg/iters)'))
        # if save==True:
        #      fig.write_html("path/to/file.html")
        fig.show()



##### TEST AREA #####  
activity_mags=[0.001,0.01,0.05,0.1,0.25,0.5,1,2]
# shifts = np.arange(3,9)
# shifts=np.arange(((num_links*2)+1)*N_inc)  
# shifts=np.arange(0,360)
# shifts=np.concatenate((np.arange(0,176), np.arange(183,359)))
shifts=np.arange(0,360,N_inc)
curr_theta,plot_Type=0,1
save=True
heatmap(activity_mags,shifts,curr_theta,plot_Type,save)


# plotting_CAN_dynamics(1,357)
#print(CANdynamics(0.5,90))
  


#plotting_CAN_velocity()
