import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pyrsistent import inc
from scipy import spatial
import math 

##VARABLES##
N=360 #number of neurons in the network
angles = np.arange(0,N)
inc=15 # degrees 
radius=1.8  #meter
whl_wdth=0.4 # meter
tic_rate=0.1 # meter per wheel tick 
dt=0.2 #sampling rate
left_speed,right_speed=14,10
start_x,start_y=0,1.2
weights = np.random.rand(N)/np.linalg.norm(np.random.rand(N))
############

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

class attractorNetwork:
    '''defines 1D attractor network with N neurons, angles associated with each neurons 
    along with inhitory and excitatory connections to update the weights'''
    def __init__(self, lndmark_angle, robo_angle,N, angles):
        self.lndmark_angle=lndmark_angle
        self.robo_angle=robo_angle
        self.N=N  
        self.angles=angles
        

    def inhibitions(self,id):
        ''' each nueuron inhibits all other nueurons but itself'''
        return np.delete(np.arange(self.N),id)

    def excitations(self,id):
        '''each neuron excites itself and 2 neurons left and right with wraparound connections'''
        if (id>=2) & (id<=self.N-3):
            return np.array([id-2,id-1,id,id+1,id+2]) 
        #lower boundary connections
        if id==1:
            return np.array([self.N-1,id-1,id,id+1,id+2]) 
        if id==0:
            return np.array([self.N-2,self.N-1,id,id+1,id+2]) 
        #upper boundary connections 
        if id==self.N-2:
            return np.array([id-2,id-1,id,id+1,0]) 
        if id==self.N-1:
            return np.array([id-2,id-1,id,0,1]) 

    def update_weights(self,prev_weights):
        '''excites 5 neurons with distributed weights of 0.5, 0.75, 1, 0.75, 0.5 and inhibits all but current '''
        #landmark angle prediction 
        prev_weights[self.excitations(int(self.lndmark_angle))]+=np.array([0.5, 0.75, 1, 0.75, 0.5])
        prev_weights[self.inhibitions(int(self.lndmark_angle))]-=0.1

        #robot angle position 
        prev_weights[self.excitations(int(self.robo_angle))]+=np.array([0.5, 0.75, 1, 0.75, 0.5])
        prev_weights[self.inhibitions(int(self.robo_angle))]-=0.1
        return prev_weights/np.linalg.norm(prev_weights)

def driveInCircle(radius, whl_wdth, tic_rate, left_speed, right_speed, start_x, start_y, dt):
    #Creater a new robot object with wheel width of 15cm and 1cm per wheel tick
    bot = Robot(whl_wdth, tic_rate)
    #Place the robot at -0,-1,2 which is bottom middle of our arena
    bot.state = np.array([[start_x],[start_y],[0]])
    # Number of iterations 
    iters = 200
    state  = np.zeros((iters,3))
    for c in range(iters):
        state[c,:] = bot.state[:,0]
        bot.drive(left_speed, right_speed, dt)
    x,y,theta=np.array(state)[:,0], np.array(state)[:,1], np.array(state)[:,2]


    # Landmarks every 24 degrees with radius of 1.8m
    mark_x=[]
    mark_y=[]
    for angle in range (0,360,inc):
        mark_x.append(radius * np.sin(np.pi * 2 * angle / 360))
        mark_y.append(radius * np.cos(np.pi * 2 * angle / 360))
    lndmrks=np.stack((np.array(mark_x),np.array(mark_y)),axis=1)


    # create the figure and axes objects
    fig, (ax1,ax2) = plt.subplots(2,1)
    def animate(i):
        '''robot arena'''
        ax1.clear()
        ax1.grid(True)
        ax1.set_xlim([-2,2])
        ax1.set_ylim([-2,2])
        #landmarks
        ax1.scatter(mark_x, mark_y, marker="*", c='b')
        #closest landmark 
        closest_id=spatial.KDTree(lndmrks).query([x[i],y[i]])[1]
        closest=lndmrks[closest_id]
        lndmrk_angle=int(np.rad2deg(math.atan2(closest[1],closest[0]))) #range -pi to pi
        if lndmrk_angle<0: lndmrk_angle+=360 # add 360 to negative angles 
        ax1.scatter(closest[0], closest[1], marker="*", c='r')
        #robot position
        ax1.scatter(x[i], y[i], marker='h')
        robo_angle=int(np.rad2deg(math.atan2(y[i],x[i]))) #range -pi to pi
        if robo_angle<0: robo_angle+=360 # add 360 to negative angles 
    
        '''distributed weights with excitations and inhibitions'''
        global weights
        weights=attractorNetwork(lndmrk_angle,robo_angle,N,angles).update_weights(weights)
        ax2.clear()
        ax2.bar(angles, weights,width = 1)
        ax2.set_ylim([0,0.4])
        ax2.set_xlim([0,360])

        print("robo_ang:"+str(robo_angle) + "-----lndm_ang:" + str(lndmrk_angle), "-----pred_ang:" + str(np.argmax(weights)))

    '''animation for driving in a circle'''
    ani = FuncAnimation(fig, animate, frames=iters, interval= 600, repeat=False)
    plt.show()

         
##### TEST AREA #####  
driveInCircle(radius,whl_wdth,tic_rate,left_speed, right_speed, start_x, start_y, dt)  #online learning 











