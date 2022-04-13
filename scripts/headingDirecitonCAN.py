from turtle import width
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pyrsistent import inc
from scipy import spatial
import math 

def col_round(x):
  frac = x - math.floor(x)
  if frac < 0.5: return math.floor(x)
  return math.ceil(x)
  
######################--VARABLES--############################
#simulation
sim_speed=10
iters=400

#network
N=40 #number of neurons
N_inc=360/N # number of degrees per neuron
neurons=np.arange(0,N)
curr_theta=0
delta=-1
angles = np.arange(0,360,abs(delta))
prev_weights=np.zeros(N)
inhbit_val=0.05
num_links=4
lndmrk_confidence=1

#landmarks
inc=60 # increment angle for landmarks 
ang_rate=1 #deg per iteration for the robot
radius=1.8  #meter
prediction,current, true=[],[],[]

#robot parameters
whl_wdth=0.8 # meter
tic_rate=0.1 # meter per wheel tick 
dt=0.2 #sampling rate
left_speed,right_speed=0,10 #ticks/sec
start_x,start_y=0,-1.2
################################################################

class Robot:
    def __init__(self, wheels_width, wheels_scale):
        # State is a vector of [x,y,theta]'
        self.state = np.zeros((3,1))
        # Wheel parameters
        self.wheels_width = wheels_width  # The distance between the left and right wheels
        self.wheels_scale = wheels_scale  # The scaling factor converting ticks/s to m/s
            
    def drive(self, left_speed, right_speed, dt):
        # left_speed and right_speed are the speeds in ticks/s of the left and right wheels.
        # dt is the length of time to drive for

        # Compute the linear and angular velocity from wheel speeds
        linear_velocity, angular_velocity = self.convert_wheel_speeds(left_speed, right_speed)
        #This is the current state of the robot
        x_k = self.state[0]
        y_k = self.state[1]
        theta_k = self.state[2]
        # Apply the velocities
    
        if angular_velocity == 0:
            #-----------------------------FILL OUT DRIVE STRAIGHT CODE--------------
            x_kp1 = x_k + np.cos(theta_k)*linear_velocity*dt
            y_kp1 = y_k + np.sin(theta_k)*linear_velocity*dt
            theta_kp1 = theta_k
            #-----------------------------------------------------------------------
        else:
            #-----------------------------FILL OUT DRIVE CODE-----------------------
            x_kp1 = x_k + linear_velocity / angular_velocity * (np.sin(theta_k+dt*angular_velocity) - np.sin(theta_k))
            y_kp1 = y_k + linear_velocity / angular_velocity * (-np.cos(theta_k+dt*angular_velocity) + np.cos(theta_k))
            theta_kp1 = theta_k + angular_velocity*dt
            #----------------------------------------------------------------------- 
        #Save our state 
        self.state[0] = x_kp1
        self.state[1] = y_kp1
        self.state[2] = theta_kp1
            
                
    def convert_wheel_speeds(self, left_speed, right_speed):
        # Convert to m/s
        left_speed_m = left_speed * self.wheels_scale
        right_speed_m = right_speed * self.wheels_scale
        # Compute the linear and angular velocity
        linear_velocity = (left_speed_m + right_speed_m) / 2.0
        angular_velocity = (right_speed_m - left_speed_m) / self.wheels_width
        
        return linear_velocity, angular_velocity

def robotState(whl_wdth, tic_rate, left_speed, right_speed, start_x, start_y, dt,iters):
    #Creater a new robot object with wheel width of 15cm and 1cm per wheel tick
    bot = Robot(whl_wdth, tic_rate)
    #Place the robot at -0,-1,2 which is bottom middle of our arena
    bot.state = np.array([[start_x],[start_y],[0]])
    # Number of iterations 
    state  = np.zeros((iters,3))
    vels = np. zeros((iters,2))
    for c in range(iters):
        state[c,:] = bot.state[:,0]
        vels[c,:] = bot.convert_wheel_speeds(left_speed,right_speed)
        bot.drive(left_speed, right_speed, dt)
    x,y,theta=np.array(state)[:,0], np.array(state)[:,1], np.array(state)[:,2]
    lin_vel,ang_vel=np.array(vels)[:,0], np.array(vels)[:,1]
    return x,y,theta

class attractorNetwork:
    '''defines 1D attractor network with N neurons, angles associated with each neurons 
    along with inhitory and excitatory connections to update the weights'''
    def __init__(self, delta, landmark,N,inhibit_val,num_links,lndmrk_confidence):
        self.delta=delta 
        self.landmark=landmark
        self.N=N  
        self.inhibit_val=inhibit_val
        self.num_links=num_links
        self.lndmrk_confidence=lndmrk_confidence

    def theta_update(self,curr_theta):
        # angles=np.arange(360)
        x= curr_theta+self.delta 
        return x % 360
        
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
        frac=(curr_theta*(N/360))-np.floor(curr_theta*(N/360))
        iteration=round(frac*(360/N))#*np.sign(delta)
        resolution=(self.num_links*2+1)*(360/N)

        x_angles=np.arange(-resolution/2,(resolution/2))
        distr_angles=1/(np.std(x_angles) * np.sqrt(2 * np.pi)) * np.exp( - (x_angles - iteration)**2 / (2 * np.std(x_angles)**2))
        distr_reduced=np.add.reduceat(distr_angles, np.arange(0, len(distr_angles), round(360/N)))

        return distr_reduced

    def full_weights(self):
        x=np.arange(-self.num_links,self.num_links+1)
        return 1/(np.std(x) * np.sqrt(2 * np.pi)) * np.exp( - (x - np.mean(x))**2 / (2 * np.std(x)**2))  

    def update_weights(self,prev_weights,theta):
        '''excites (num_links x 2)+1  neurons with distributed weights and inhibits all but current '''
        #exite and inhibit robot angle 
        prev_weights=np.zeros(self.N)
        
        prev_weights[self.excitations(int(theta*(N/360)))]=self.frac_weights()
        prev_weights[self.inhibitions(int(theta*(N/360)))]-=self.inhibit_val
        
        landmark_weights=np.zeros(self.N)
        if self.landmark is not None:
            landmark_weights[self.excitations(int(self.landmark*(N/360)))]=self.full_weights()*self.lndmrk_confidence 
            #landmark_weights = landmark_weights/np.linalg.norm(landmark_weights)
          
        return prev_weights/np.linalg.norm(prev_weights), landmark_weights

def activity_center(prev_weights):
    '''Takes the previous weights as inputs and finds the activity center with pdf as polar coordinates'''
    indexes=np.arange(0,N)
    non_zero_idx=indexes[prev_weights>0]
    non_zero_weights=np.zeros(N)
    if non_zero_idx is not np.empty:
        non_zero_weights[non_zero_idx]=prev_weights[non_zero_idx]
        x,y=non_zero_weights*np.cos(np.deg2rad(neurons*360/N)), non_zero_weights*np.sin(np.deg2rad(neurons*360/N))
        vect_sum=round(np.rad2deg(np.arctan2(sum(y),sum(x))))
        # changing range from [-179, 180] to [0,360]
        if vect_sum<0:
            shifted_vec=vect_sum+360
        else:
            shifted_vec=vect_sum
    return shifted_vec, x, y

def selfMotionLandmark_integration(radius,inc,iters):
    # Landmarks position and angle -> every inc degrees with radius of 1.8m
    mark_x=[]
    mark_y=[]
    lndmrk_angles=[]
    for phi in range (0,360,inc):
        mark_x.append(radius * math.cos(np.deg2rad(phi)) )
        mark_y.append(radius * math.sin(np.deg2rad(phi)))
        lndmrk_angles.append(phi)
    lndmrks=np.stack((np.array(mark_x),np.array(mark_y)),axis=1)

    # create the figure and axes objects
    fig = plt.figure(figsize=(7,8))
    gs = fig.add_gridspec(4,2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])
    ax4= fig.add_subplot(gs[2, :])
    ax5= fig.add_subplot(gs[3, :])
    def animate(i):
        if i >= 1:
            global prev_weights, inhbit_val, lndmrk_confidence, curr_theta, prediction, current, landmark_weights
            ax1.clear()
            ax2.clear()
            ax3.clear()
            ax4.clear()
        #robot arena
            ax1.grid(True)
            ax1. set_aspect('equal')
            ax1.set_xlim([-2,2])
            ax1.set_ylim([-2,2])
        #landmarks
            ax1.scatter(mark_x, mark_y, marker="*", c='b')
        #robot angle 
            ax1.arrow(0,0,0.5*np.cos(np.deg2rad(curr_theta)),0.5*np.sin(np.deg2rad(curr_theta)),width=0.04)
        #detecting landmark
            if delta in (np.array(lndmrk_angles)-curr_theta):
                lndmrk_id=np.argmin(abs(np.array(lndmrk_angles)-curr_theta))
                lndmrk_neuron=int(lndmrk_angles[lndmrk_id])
                ax1.scatter(lndmrks[lndmrk_id,0], lndmrks[lndmrk_id,1], marker="*", c='r')
                print("landmark:" + str(lndmrk_neuron))
            else: lndmrk_neuron=None 
        # shifting network with delta
            net=attractorNetwork(delta,lndmrk_neuron,N,inhbit_val,num_links,lndmrk_confidence)
            curr_theta=net.theta_update(curr_theta)
            prev_weights, landmark_weights=net.update_weights(prev_weights,curr_theta)
        #predicting theta
            theta_pred,x,y=activity_center(prev_weights)
        #plotting arrows of bumps
            ax2.set_xlim([-0.2*5,0.2*5])
            ax2.set_ylim([-0.2*5,0.2*5])
            ax2. set_aspect('equal')
            for j in range(0,N):
                ax2.arrow(0,0,x[j]*15,y[j]*15)
        #plotting activity for self motion 
            ax3.bar(neurons, prev_weights,width=0.8)
            ax3.set_ylim([-0.2,0.3])

        #plotting activity for landmark 
            if landmark_weights is not None:
                ax4.bar(neurons, landmark_weights,width=0.8, color='green')
                ax4.set_ylim([-0.2,0.3])

        # printing parameters and plotting error 
            print(i)
            print("activity center: "+ str(theta_pred) + "---model input: " + str(int(curr_theta)) + "---true angle: "+ str(i*delta % 360))
            prediction.append(abs((i*delta % 360) - theta_pred))
            current.append(abs((i*delta % 360) -curr_theta))

            line1, =ax5.plot(np.arange(len(prediction)),np.array(prediction),'b-')
            line2, =ax5.plot(np.arange(len(current)),np.array(current),'r-')
            ax5.legend([line1, line2], ['prediction error', 'input error'])


    '''animation for driving in a circle'''
    ani = FuncAnimation(fig, animate, frames=iters, interval= sim_speed, repeat=False)
    plt.show()

def plotting_CAN_dynamics(landmark):
    fig, (ax1) = plt.subplots()
    def animate(i):
        global prev_weights, inhbit_val,lndmrk_weight , curr_theta
        ax1.clear()
        '''distributed weights with excitations and inhibitions'''
        net=attractorNetwork(delta,None,N,inhbit_val,num_links,lndmrk_weight)
        net1=attractorNetwork(delta,landmark,N,inhbit_val,num_links,lndmrk_weight)

        stored_activity=np.zeros((40,N))
        stored_activity[i,:]=prev_weights
    
        if i==0:
            curr_theta=30
        if i < 3:
            curr_theta=net1.theta_update(curr_theta)
            prev_weights= net1.update_weights(prev_weights,curr_theta)
            plt.title("Landmark at 90, 3 iters")
        elif (i>10)&(i<20):
            curr_theta=net1.theta_update(curr_theta)
            prev_weights= net1.update_weights(prev_weights,curr_theta)
            plt.title("Landmark at 90, 10 iters")
        else:
            curr_theta=net.theta_update(curr_theta)
            prev_weights= net.update_weights(prev_weights,curr_theta)
            plt.title("Shifting Forwards")
        
        
        ax1.bar(neurons*(360/N), prev_weights,width=(360/N)-1)
        ax1.set_ylim([-0.2,0.4])

        '''printing outputs'''
        print(np.round(activity_center(prev_weights)*360/N))

    '''animation for driving in a circle'''
    ani = FuncAnimation(fig, animate, frames=30, interval= 10, repeat=False)
    plt.show() 

def plotting_CAN_velocity():
    fig, (ax1) = plt.subplots()
   
    def animate(i):
        global prev_weights, inhbit_val,lndmrk_weight, curr_theta
        '''distributed weights with excitations and inhibitions'''
        net=attractorNetwork(delta,None,N,inhbit_val,num_links,lndmrk_weight)
        curr_theta=net.theta_update(curr_theta)
        prev_weights=net.update_weights(prev_weights,curr_theta)
        
        ax1.clear()
        ax1.bar(neurons*(360/N), prev_weights,width=2)
        ax1.set_ylim([-0.2,0.6])

        '''printing outputs'''
        center=activity_center(prev_weights)*360/N
        print(np.round(center), curr_theta)

    '''animation for driving in a circle'''
    ani = FuncAnimation(fig, animate, frames=iters, interval= 1000, repeat=False)
    plt.show() 

##### TEST AREA #####  
selfMotionLandmark_integration(radius,inc,iters)  #online learning 

#plotting_CAN_dynamics(90)

#plotting_CAN_velocity()


# def mean(neurons, distr_reduced):
#     x,y=distr_reduced*np.cos(np.deg2rad(neurons)), distr_reduced*np.sin(np.deg2rad(neurons))
#     vect_sum=np.rad2deg(np.arctan2(sum(y),sum(x)))
#     print(vect_sum)

# ang=np.arange(0,360)
# #x=np.arange(-24,25)
# x=np.arange(-num_links,num_links+1)
# angs=np.arange(0,7)
# distr= 1/(np.std(x) * np.sqrt(2 * np.pi)) * np.exp( - (x - 1.25)**2 / (2 * np.std(x)**2))  

# resolution=int((num_links*2+1)*(360/N))
# x_angles=np.arange(-resolution/2,(resolution/2))
# for i in range(6):
# #print(np.mean(x_angles))
#     distr_angles=1/(np.std(x_angles) * np.sqrt(2 * np.pi)) * np.exp( - (x_angles - i)**2 / (2 * np.std(x_angles)**2))
#     #mean(x_angles,distr_angles)


#     distr_reduced=np.add.reduceat(distr_angles, np.arange(0, len(distr_angles), round(360/N)))
#     neurons=np.arange(-len(distr_reduced)/2,len(distr_reduced)/2)
#     mean(neurons,distr_reduced)

# fig,(ax1,ax2)=plt.subplots(2,1)
# ax2.bar(neurons,distr_reduced,width=1)
# ax1.bar(x_angles,distr_angles)

# plt.show()

'''
x,y,theta=robotState(whl_wdth, tic_rate, left_speed, right_speed, start_x, start_y, dt,iters)
fig,ax=plt.subplots()
ax.plot(x,y,"*")
plt.show()
'''