import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import math 

def col_round(x):
  frac = x - math.floor(x)
  if frac < 0.5: return math.floor(x)
  return math.ceil(x)
  
######################--VARABLES--############################
#simulation
sim_speed=500
iters=360
prediction,current, true=[],[],[]

#network
N=60 #number of neurons
N_inc=360/N # number of degrees per neuron
neurons=np.arange(0,N)
curr_theta=0
iteration=1
delta=1
prev_weights=np.zeros(N)
inhbit_val=0.05
num_links=3
lndmrk_confidence=1
input_error=0

#landmarks
inc=5 # increment angle for landmarks 
ang_rate=1 #deg per iteration for the robot
radius=1.8  #meter
landmark_dect_toler=6

'''#robot parameters
whl_wdth=0.8 # meter
tic_rate=0.1 # meter per wheel tick 
dt=0.2 #sampling rate
left_speed,right_speed=0,10 #ticks/sec
start_x,start_y=0,-1.2'''
################################################################
'''class Robot:
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
'''

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
        return (curr_theta+self.delta) % 360
        
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

    def update_weights(self,prev_weights,theta):
        '''excites (num_links x 2)+1  neurons with distributed weights and inhibits all but current '''
        #exite and inhibit robot angle 
        #prev_weights=np.zeros(self.N)
        selfmotion_weights=np.zeros(self.N)
        selfmotion_weights[self.excitations(int(theta*(N/360)))]=self.full_weights()

        prev_weights[self.excitations(int(theta*(N/360)))]+=self.full_weights()
        prev_weights[self.inhibitions(int(theta*(N/360)))]-=self.inhibit_val
        
        landmark_weights=np.zeros(self.N)
        if self.landmark is not None:
            landmark_weights[self.excitations(int(self.landmark*(N/360)))]=self.full_weights()*self.lndmrk_confidence 
            #landmark_weights = landmark_weights/np.linalg.norm(landmark_weights)
          
        return prev_weights/np.linalg.norm(prev_weights), landmark_weights, selfmotion_weights

    def update_weights_dynamics(self,prev_weights,theta):
        prev_weights[self.excitations(int(theta*(N/360)))]+=self.full_weights()
        prev_weights[self.inhibitions(int(theta*(N/360)))]-=self.inhibit_val
        activity=np.zeros(N)
        activity[self.excitations(int(theta*(N/360)))]=self.full_weights()
        return prev_weights/np.linalg.norm(prev_weights), activity

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

def selfMotion_Landmark(radius,inc,iters):
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
    plt.style.use(['science', 'no-latex'])
    fig = plt.figure(figsize=(7,7))
    gs = fig.add_gridspec(2,2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])
    fig.tight_layout(pad=5.0)
    # ax4= fig.add_subplot(gs[2, :])
    # ax5= fig.add_subplot(gs[3, :])
    # fig.tight_layout()
    def animate(i):
        if i >= 1:
            global prev_weights, inhbit_val, lndmrk_confidence, curr_theta, prediction, current, landmark_weights, input_error
            ax1.clear(), ax2.clear(), ax3.clear(),
            #  ax4.clear(), ax5.clear()
            ax1.scatter(mark_x, mark_y, marker="*", c='b') #landmarks 
            '''calculating values'''
            #detecting landmark
            if any(i < landmark_dect_toler for i in abs(np.array(lndmrk_angles)-curr_theta)):
                lndmrk_id=np.argmin(abs(np.array(lndmrk_angles)-curr_theta))
                lndmrk_neuron=int(lndmrk_angles[lndmrk_id])
                ax1.scatter(lndmrks[lndmrk_id,0], lndmrks[lndmrk_id,1], marker="*", c='r')
                print("landmark:" + str(lndmrk_neuron))
            else: lndmrk_neuron=None 
            # shifting network with delta
            net=attractorNetwork(delta+input_error,lndmrk_neuron,N,inhbit_val,num_links,lndmrk_confidence)
            curr_theta=net.theta_update(curr_theta)
            prev_weights, landmark_weights, selfmotion_weights=net.update_weights(prev_weights,curr_theta)
            prev_weights=prev_weights>0
            #predicting theta
            theta_pred,x,y=activity_center(prev_weights)

            '''plotting and printing results'''
            #plotting robot arena
            ax1.set_title('Robot Arena')
            ax1.grid(True)
            ax1. set_aspect('equal')
            ax1.set_xlim([-2,2])
            ax1.set_ylim([-2,2])
            ax1.set_xlabel('x axis [m]')
            ax1.set_ylabel('y axis [m]')
            ax1.arrow(0,0,0.5*np.cos(np.deg2rad(curr_theta)),0.5*np.sin(np.deg2rad(curr_theta)),width=0.04) #robot angle

            #plotting arrows of bumps
            ax2.set_title('Head Direction Activity')
            ax2.set_xlim([-0.2*12,0.2*12])
            ax2.set_ylim([-0.2*12,0.2*12])
            ax2. set_aspect('equal')
            ax2.set_xlabel('x axis [m]')
            ax2.set_ylabel('y axis [m]')
            for j in range(0,N):
                ax2.arrow(0,0,x[j]*10,y[j]*10)

            #plotting activity for self motion 
            ax3.set_title('Attractor Network Dynamics')
            ax3.bar(neurons, prev_weights,width=0.8,label='Selfmotion Activity')
            ax3.set_ylim([-0.2,0.5])
            ax3.set_xlabel('Neurons')
            ax3.set_ylabel('Activity Weight')

            # ax4.set_title('Self Motion and Landmark Activity')
            # ax4.bar(neurons, selfmotion_weights,width=0.8, color='purple')
            # ax4.set_ylim([-0.2,0.5])

            #plotting activity for landmark 
            if landmark_weights is not None:
                # ax5.set_title('Landmark Activity')
                ax3.bar(neurons, landmark_weights,width=0.8, label='Landmark Activity',color='red')
                # ax5.set_ylim([-0.2,0.5])
            ax3.legend()
                


            #plotting error and printing parameters
            #print("activity center: "+ str(theta_pred) + "---model input: " + str(int(curr_theta)) + "---true angle: "+ str(i*delta % 360))
            #print(theta_pred)
            # prediction.append(abs((i*delta % 360) - theta_pred))
            # current.append(abs((i*delta % 360) -curr_theta))

            # line1, =ax5.plot(np.arange(len(prediction)),np.array(prediction),'b-')
            # line2, =ax5.plot(np.arange(len(current)),np.array(current),'r-')
            # ax5.set_title('Error from true angle')
            # ax5.legend([line1, line2], ['prediction error', 'input error'])

    '''animation for driving in a circle'''
    ani = anim.FuncAnimation(fig, animate, frames=iters, interval= sim_speed, repeat=False)
    plt.show()

    # f =f"./results/GIFs/HDanimation.gif" 
    # writergif = anim.PillowWriter(fps=10) 
    # ani.save(f, writer=writergif)

def landmark_learning(radius,inc,iters):
    # Landmarks position and angle -> every inc degrees with radius of 1.8m
    mark_x=[]
    mark_y=[]
    lndmrk_angles=[]
    for phi in range (0,360,inc):
        mark_x.append(radius * math.cos(np.deg2rad(phi)) )
        mark_y.append(radius * math.sin(np.deg2rad(phi)))
        lndmrk_angles.append(phi)
    lndmrks=np.stack((np.array(mark_x),np.array(mark_y)),axis=1)
    stored_activity=np.zeros((len(lndmrks),N))
        
    # create the figure and axes objects
    fig = plt.figure(figsize=(7,7.5))
    gs = fig.add_gridspec(3,2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])
    ax4= fig.add_subplot(gs[2, :])
    fig.tight_layout()
    def animate(i):
        global prev_weights, inhbit_val, lndmrk_confidence, curr_theta, prediction, current, landmark_weights, input_error, landmark_dect_toler
        ax1.clear(), ax2.clear(), ax3.clear(), ax4.clear()
        ax1.scatter(mark_x, mark_y, marker="*", c='b') #landmarks 
        if (i >= 1) and  (i <= (360/delta)):
            '''calculating values'''
            #detecting landmark
            if any(i < landmark_dect_toler for i in abs(np.array(lndmrk_angles)-curr_theta)):
                lndmrk_id=np.argmin(abs(np.array(lndmrk_angles)-curr_theta))
                lndmrk_neuron=int(lndmrk_angles[lndmrk_id])
                ax1.scatter(lndmrks[lndmrk_id,0], lndmrks[lndmrk_id,1], marker="*", c='r')
                print("landmark:" + str(lndmrk_neuron))
                stored_activity[lndmrk_id,:]=prev_weights
            else: lndmrk_neuron=None 

            # shifting network with delta
            net=attractorNetwork(delta+input_error,lndmrk_neuron,N,inhbit_val,num_links,lndmrk_confidence)
            curr_theta=net.theta_update(curr_theta)
            prev_weights, landmark_weights=net.update_weights(prev_weights,curr_theta)

            #plotting activity for self motion 
            ax3.set_title('Learning self motion at landmark')
            ax3.bar(neurons, prev_weights,width=0.8)
            ax3.set_ylim([-0.2,0.3])

        elif (i > (360/delta)):
            net=attractorNetwork(delta+input_error,None,N,inhbit_val,num_links,lndmrk_confidence)
            curr_theta=net.theta_update(curr_theta)
            if any(i < landmark_dect_toler for i in abs(np.array(lndmrk_angles)-curr_theta)):
                lndmrk_id=np.argmin(abs(np.array(lndmrk_angles)-curr_theta))
                lndmrk_neuron=int(lndmrk_angles[lndmrk_id])
                ax1.scatter(lndmrks[lndmrk_id,0], lndmrks[lndmrk_id,1], marker="*", c='r')
                print("landmark:" + str(lndmrk_neuron))
                prev_weights=stored_activity[lndmrk_id,:]
                #plotting selfmotion activity at landmark 
                ax3.set_title('Replaying self motion at landmark')
                ax3.bar(neurons, prev_weights,width=0.8, color='green')
                ax3.set_ylim([-0.2,0.3])
            else: 
                lndmrk_neuron=None 
                prev_weights, landmark_weights=net.update_weights(prev_weights,curr_theta)
                #plotting activity for self motion 
                ax3.set_title('Self motion at landmark')
                ax3.bar(neurons, prev_weights,width=0.8)
                ax3.set_ylim([-0.2,0.3])
            
        if (i >= 0):
            '''plotting and printing results'''
            #plotting robot arena
            ax1.set_title('Robot Arena')
            ax1.grid(True)
            ax1. set_aspect('equal')
            ax1.set_xlim([-2,2])
            ax1.set_ylim([-2,2])
            ax1.set_xlabel('x axis [m]')
            ax1.set_xlabel('y axis [m]')
            ax1.arrow(0,0,0.5*np.cos(np.deg2rad(curr_theta)),0.5*np.sin(np.deg2rad(curr_theta)),width=0.04) #robot angle

            #plotting arrows of bumps
            #predicting theta
            theta_pred,x,y=activity_center(prev_weights)
            ax2.set_title('Head Direction Activity')
            ax2.set_xlim([-0.2*10,0.2*10])
            ax2.set_ylim([-0.2*10,0.2*10])
            ax2. set_aspect('equal')
            ax2.set_xlabel('x axis [m]')
            ax2.set_xlabel('y axis [m]')
            for j in range(0,N):
                ax2.arrow(0,0,x[j]*10,y[j]*10)

            #plotting error and printing parameters
            print("activity center: "+ str(theta_pred) + "---model input: " + str(int(curr_theta)) + "---true angle: "+ str(i*delta % 360))
            prediction.append(abs((i*delta % 360) - theta_pred))
            current.append(abs((i*delta % 360) -curr_theta))

            line1, =ax4.plot(np.arange(len(prediction)),np.array(prediction),'b-')
            line2, =ax4.plot(np.arange(len(current)),np.array(current),'r-')
            ax4.set_title('Error from true angle')
            ax4.legend([line1, line2], ['prediction error', 'input error'])
        

    '''animation for driving in a circle'''
    ani = FuncAnimation(fig, animate, frames=iters, interval= 1, repeat=False, blit=True)
    plt.show()

def selfMotion_HD_Network(radius,inc,iters):
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
    plt.style.use(['science', 'ieee'])
    fig = plt.figure(figsize=(6,4))
    gs = fig.add_gridspec(2,2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])
    # fig.tight_layout(pad=5.0)
    # ax4= fig.add_subplot(gs[2, :])
    # ax5= fig.add_subplot(gs[3, :])
    # fig.tight_layout()
    def animate(i):
        if i >= 1:
            global prev_weights, inhbit_val, lndmrk_confidence, curr_theta, prediction, current, landmark_weights, input_error
            ax1.clear(), ax2.clear(), ax3.clear()
            #  ax4.clear(), ax5.clear()
            ax1.scatter(mark_x, mark_y, marker=".", c='k') #landmarks 
            '''calculating values'''
            #detecting landmark
            # if any(i < landmark_dect_toler for i in abs(np.array(lndmrk_angles)-curr_theta)):
            #     lndmrk_id=np.argmin(abs(np.array(lndmrk_angles)-curr_theta))
            #     lndmrk_neuron=int(lndmrk_angles[lndmrk_id])
            #     ax1.scatter(lndmrks[lndmrk_id,0], lndmrks[lndmrk_id,1], marker="*", c='r')
            #     print("landmark:" + str(lndmrk_neuron))
            # else: 
            lndmrk_neuron=None 
            # shifting network with delta
            net=attractorNetwork(delta+input_error,lndmrk_neuron,N,inhbit_val,num_links,lndmrk_confidence)
            curr_theta=net.theta_update(curr_theta)
            prev_weights, landmark_weights, selfmotion_weights=net.update_weights(prev_weights,curr_theta)
            #predicting theta
            theta_pred,x,y=activity_center(prev_weights)

            '''plotting and printing results'''
            #plotting robot arena
            ax1.set_title('Robot Arena')
            ax1.grid(True)
            ax1. set_aspect('equal')
            ax1.set_xlim([-2,2])
            ax1.set_ylim([-2,2])
            # ax1.set_xlabel('x axis [m]')
            # ax1.set_ylabel('y axis [m]')
            ax1.arrow(0,0,0.5*np.cos(np.deg2rad(curr_theta)),0.5*np.sin(np.deg2rad(curr_theta)),width=0.04, color='green') #robot angle

            #plotting arrows of bumps
            ax2.set_title('Head Direction Activity')
            ax2.set_xlim([-0.2*12,0.2*12])
            ax2.set_ylim([-0.2*12,0.2*12])
            ax2. set_aspect('equal')
            # ax2.set_xlabel('x axis [m]')
            # ax2.set_ylabel('y axis [m]')
            for j in range(0,N):
                ax2.arrow(0,0,x[j]*10,y[j]*10, color='royalblue')

            # plotting activity for self motion 
            ax3.set_title('Attractor Network Dynamics')
            ax3.bar(neurons, prev_weights,color='royalblue',width=0.8,label='Selfmotion Activity')
            ax3.set_ylim([-0.2,0.5])
            ax3.set_xlabel('Neurons')
            ax3.set_ylabel('Activity Weight')

            # ax4.set_title('Self Motion and Landmark Activity')
            # ax4.bar(neurons, selfmotion_weights,width=0.8, color='purple')
            # ax4.set_ylim([-0.2,0.5])

            #plotting activity for landmark 
            # if landmark_weights is not None:
            #     # ax5.set_title('Landmark Activity')
            #     ax3.bar(neurons, landmark_weights,width=0.8, label='Landmark Activity',color='red')
            #     # ax5.set_ylim([-0.2,0.5])
            # ax3.legend()
                


            #plotting error and printing parameters
            #print("activity center: "+ str(theta_pred) + "---model input: " + str(int(curr_theta)) + "---true angle: "+ str(i*delta % 360))
            #print(theta_pred)
            # prediction.append(abs((i*delta % 360) - theta_pred))
            # current.append(abs((i*delta % 360) -curr_theta))

            # line1, =ax5.plot(np.arange(len(prediction)),np.array(prediction),'b-')
            # line2, =ax5.plot(np.arange(len(current)),np.array(current),'r-')
            # ax5.set_title('Error from true angle')
            # ax5.legend([line1, line2], ['prediction error', 'input error'])

    '''animation for driving in a circle'''
    ani = anim.FuncAnimation(fig, animate, frames=iters, interval= sim_speed, repeat=False)
    # plt.show()

    f =f"./results/GIFs/HDanimation.gif" 
    writergif = anim.PillowWriter(fps=10) 
    ani.save(f, writer=writergif)

##### TEST AREA #####  

# selfMotion_Landmark(radius,inc,iters)  #online learning 
selfMotion_HD_Network(radius,inc,iters)
# delta=+10
# landmark_dect_toler=5
# input_error=0.005
# landmark_learning(radius,inc,iters)



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