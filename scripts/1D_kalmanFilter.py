

import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.animation import FuncAnimation
from scipy.stats import norm 



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
    print(x_predicted)
    axs.plot(xaxis,norm.pdf(xaxis,x_predicted,np.sqrt(Q))) #Predicted State
    axs.plot(xaxis,norm.pdf(xaxis,X,2)) # ground truth
    axs.plot(xaxis,norm.pdf(xaxis,x,P)) # corrected state 
    axs.plot(xaxis,norm.pdf(xaxis,y,np.sqrt(R))) # measurement 
    axs.set_title(' 1D Kalman Filter')
    axs.legend(['Predicted State','Ground Truth','Corrected State', 'Sensor Measurement'])
    axs.set_xlabel('Position')
    axs.set_ylabel('Probabilty Density')

ani = FuncAnimation(fig, animate, interval=300,frames=len(tt),repeat=False)
plt.show()

# tt,xx_predicted,tt,yy   ,'Predicted State','Sensor Measurements'

