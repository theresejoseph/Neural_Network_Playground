

import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.animation import FuncAnimation
from scipy.stats import norm 
np.random.seed(5)


# % Storing calculated values in these vectors for plotting
XX = np.zeros((100))
tt = np.zeros((100))
xx_predicted = np.zeros((100))
xx = np.zeros((100))
PP = np.zeros((100))
yy = np.zeros((100))

# %Define System
X = 0
dt = 1
u = 10        #%speed/control = 10m/s
n = np.random.randn()
v = np.random.randn()

# %Kalman Filter variables
x = 50      #%state vector
A = 1       #%state transition matrix
B = 1       #%control input matrix 
P = 100     #%std_dev*std_dev = 10*10 
Q = 100     # %process noise covariance matrix 
R = 9       # %measurement noise covariance matrix
H = 1

fig, axs = plt.subplots(1, 1, figsize=(9, 5))
def animate(t):
    global X, dt, u, n, v, x, A, B, P, Q, R, H
    axs.clear()
    xaxis=np.arange(-20,1100,1)

#    %simulating the System
    n = np.sqrt(Q) * np.random.randn() #%random noise
    X = X + u*dt + n                   #% New Current State
    v = np.sqrt(R) * np.random.randn() 
    y = H*X + v                        #% Measurements 
    
#    %Prediction Step
    x_predicted = A*x + B*u            #predicting the new state (mean)
    P = A * P * np.transpose(A) + Q    #predicting the new uncertainity (covariance)


#    %Correction Step
    e = H*x_predicted          #expectation: predicted measurement from the o/p
    E = H*P*np.transpose(H)    # Covariance of ^ expectation
    z = y - e                  #innovation: diff between the expectation and real sensor measurement
    Z = R + E                  # Covariance of ^ - sum of uncertainities of expectation and real measurement
    K = P*np.transpose(H) * (Z.astype(float)**-1)
    
    x = x_predicted + K*z  #%final corrected state
    P = P - (K* H* P)      #%uncertainity in corrected state
    
#    %Saving the outputs
    xx_predicted[t] = x_predicted
    xx[t]  = x
    PP[t]  = P
    XX[t]  = X
    tt[t]  = t
    yy[t]  = y

    # axs.plot(tt,XX,tt,xx,tt,xx_predicted,tt,yy)
    # print(x,P, y,R)
    # print(x_predicted)
    axs.plot(xaxis,norm.pdf(xaxis,x_predicted,np.sqrt(Q))) #Predicted State
    axs.plot(xaxis,norm.pdf(xaxis,X,2)) # ground truth
    axs.plot(xaxis,norm.pdf(xaxis,x,P)) # corrected state 
    axs.plot(xaxis,norm.pdf(xaxis,y,np.sqrt(R))) # measurement 
    axs.set_title(' 1D Kalman Filter')
    axs.legend(['Predicted State','Ground Truth','Corrected State', 'Sensor Measurement'])
    axs.set_xlabel('Position')
    axs.set_ylabel('Probabilty Density')

# ani = FuncAnimation(fig, animate, interval=300,frames=len(tt),repeat=False)
# plt.show()

# tt,xx_predicted,tt,yy   ,'Predicted State','Sensor Measurements'

'''Kalman Filter Implimentation: https://github.com/motokimura/kalman_filter_witi_kitti/blob/master/demo.ipynb'''


import numpy as np
import matplotlib.pyplot as plt

# Create idealized system to match filter assumptions
# state is (position, velocity)'
# We are assuming Q, R, and P0 to be diagonal

T = 0.01

A = np.array([[1, T],
              [0, 1]])

B = np.array([[0],
              [T]])

C = np.array([1, 0]).reshape(1, 2)

# process variance
Q = np.array([[1e-6, 0],
              [0, 1e-5]])

# sensor noise variance
R = np.array([[1e-5]])

# initial state estimate variance
P0 = np.array([[1e-4, 0],
               [0, 1e-4]])

# Create some data

state = np.array([[np.sqrt(P0[0, 0]) * np.random.randn()],[ np.sqrt(P0[1, 1]) * np.random.randn()]])

posd = np.random.randn(1000)  # assume we have 1000 position setpoints
postrue = np.zeros_like(posd)
veltrue = np.zeros_like(posd)
measurement = np.zeros_like(posd)
torque = np.random.randn(1000)  # assume we have 1000 torque inputs

for i in range(len(posd)):
    postrue[i] = state[0]
    veltrue[i] = state[1]

    # simulate noisy measurement
    measurement[i] = C @ state + np.sqrt(R[0, 0]) * np.random.randn()

    process_noise = np.array([[np.sqrt(Q[0, 0]) * np.random.randn()],[np.sqrt(Q[1, 1]) * np.random.randn()]])

    state = A @ state + B * torque[i] + process_noise

# Design filter
# Note that we can design filter in advance of seeing the data.
Pm = P0
for i in range(1000):
    # measurement step
    S = C @ Pm @ C.T + R
    K = Pm @ C.T @ np.linalg.inv(S)
    Pp = Pm - K @ C @ Pm

    # prediction step
    Pm = A @ Pp @ A.T + Q

# Run the filter to create example output
sem = np.zeros((2, 1))
pose = np.zeros_like(posd)
vele = np.zeros_like(veltrue)
for i in range(len(posd)):
    # measurement step
    sep = sem + K @ (measurement[i] - C @ sem)
    pose[i] = sep[0]
    vele[i] = sep[1]

    # prediction step
    sem = A @ sep + B * torque[i]

# Let's plot the Kalman filter output
ii = np.arange(len(pose))
# plt.plot(ii, pose, 'b', ii, postrue, 'r')
# plt.plot(ii, measurement, 'b', ii, postrue, 'r')
plt.plot(ii, vele, 'b', ii, veltrue, 'r')
plt.legend(['KF velocity', 'true velocity'])
plt.xlabel('sample (100Hz)')
plt.show()