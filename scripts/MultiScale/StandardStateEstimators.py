import matplotlib.pyplot as plt
import numpy as np
import random  
import math
from numpy.random import randn
from SelectiveMultiScalewithWraparound2D import headDirectionAndPlaceNoWrapNet


'''===========================================================Extended Kalman Filter============================================================='''
from filterpy.kalman import ExtendedKalmanFilter as EKF
from numpy import array, sqrt
from math import sqrt, tan, cos, sin, atan2
import sympy
from sympy.abc import alpha, x, y, v, w, R, theta
from sympy import symbols, Matrix

def Hx(x, landmark_pos):
    """ takes a state variable and returns the measurement
    that would correspond to that state.
    """
    px = landmark_pos[0]
    py = landmark_pos[1]
    dist = sqrt((px - x[0, 0])**2 + (py - x[1, 0])**2)

    Hx = array([[dist],
                [atan2(py - x[1, 0], px - x[0, 0]) - x[2, 0]]])
    return Hx

def H_of(x, landmark_pos):
    """ compute Jacobian of H matrix where h(x) computes 
    the range and bearing to a landmark for state x """

    px = landmark_pos[0]
    py = landmark_pos[1]
    hyp = (px - x[0, 0])**2 + (py - x[1, 0])**2
    dist = sqrt(hyp)

    H = array(
        [[-(px - x[0, 0]) / dist, -(py - x[1, 0]) / dist, 0],
         [ (py - x[1, 0]) / hyp,  -(px - x[0, 0]) / hyp, -1]])
    return H

class RobotEKF(EKF):
    def __init__(self, dt, wheelbase, std_vel, std_steer):
        EKF.__init__(self, 3, 2, 2)
        self.dt = dt
        self.wheelbase = wheelbase
        self.std_vel = std_vel
        self.std_steer = std_steer

        a, x, y, v, w, theta, time = symbols(
            'a, x, y, v, w, theta, t')
        d = v*time
        beta = (d/w)*sympy.tan(a)
        r = w/sympy.tan(a)
    
        self.fxu = Matrix(
            [[x-r*sympy.sin(theta)+r*sympy.sin(theta+beta)],
             [y+r*sympy.cos(theta)-r*sympy.cos(theta+beta)],
             [theta+beta]])

        self.F_j = self.fxu.jacobian(Matrix([x, y, theta]))
        self.V_j = self.fxu.jacobian(Matrix([v, a]))

        # save dictionary and it's variables for later use
        self.subs = {x: 0, y: 0, v:0, a:0, 
                     time:dt, w:wheelbase, theta:0}
        self.x_x, self.x_y, = x, y 
        self.v, self.a, self.theta = v, a, theta

    def predict(self, u):
        self.x = self.move(self.x, u, self.dt)
        self.subs[self.x_x] = self.x[0, 0]
        self.subs[self.x_y] = self.x[1, 0]

        self.subs[self.theta] = self.x[2, 0]
        self.subs[self.v] = u[0]
        self.subs[self.a] = u[1]

        F = array(self.F_j.evalf(subs=self.subs)).astype(float)
        V = array(self.V_j.evalf(subs=self.subs)).astype(float)

        # covariance of motion noise in control space
        M = array([[self.std_vel**2, 0], 
                   [0, self.std_steer**2]])

        self.P = F @ self.P @ F.T + V @ M @ V.T

    def move(self, x, u, dt):
        hdg = x[2, 0]
        vel = u[0]
        steering_angle = u[1]
        dist = vel * dt

        if abs(steering_angle) > 0.001: # is robot turning?
            beta = (dist / self.wheelbase) * tan(steering_angle)
            r = self.wheelbase / tan(steering_angle) # radius

            dx = np.array([[-r*sin(hdg) + r*sin(hdg + beta)], 
                           [r*cos(hdg) - r*cos(hdg + beta)], 
                           [beta]])
        else: # moving in straight line
            dx = np.array([[dist*cos(hdg)], 
                           [dist*sin(hdg)], 
                           [0]])
        return x + dx

def residual(a, b):
    """ compute residual (a-b) between measurements containing 
    [range, bearing]. Bearing is normalized to [-pi, pi)"""
    y = a - b
    y[1] = y[1] % (2 * np.pi)    # force in range [0, 2 pi)
    if y[1] > np.pi:             # move to [-pi, pi)
        y[1] -= 2 * np.pi
    return y


def z_landmark(lmark, sim_pos, std_rng, std_brg):
    x, y = sim_pos[0, 0], sim_pos[1, 0]
    d = np.sqrt((lmark[0] - x)**2 + (lmark[1] - y)**2)  
    a = atan2(lmark[1] - y, lmark[0] - x) - sim_pos[2, 0]
    z = np.array([[d + randn()*std_rng],
                  [a + randn()*std_brg]])
    return z

def ekf_update(ekf, z, landmark):
    ekf.update(z, HJacobian=H_of, Hx=Hx, 
               residual=residual,
               args=(landmark), hx_args=(landmark))
                    
def run_localization(landmarks, std_vel, std_steer, 
                     std_range, std_bearing,
                     step=10, ellipse_step=20, ylim=None, plot=True):
    ekf = RobotEKF(dt, wheelbase=0.5, std_vel=std_vel, 
                   std_steer=std_steer)
    ekf.x = array([[0, 0, 0]]).T # x, y, steer angle
    ekf.P = np.diag([.1, .1, .1])
    ekf.R = np.diag([std_range**2, std_bearing**2])

    sim_pos = ekf.x.copy() # simulated position
    # steering command (vel, steering angle radians)
    # u = array([1.1, .01]) 

    # plt.figure()
    # plt.scatter(landmarks[:, 0], landmarks[:, 1],
                # marker='s', s=60)
    
    track = []
    ekfPos=[]
    for i in range(test_length):
        u = array([vel[i], angVel[i]]) 
        sim_pos = ekf.move(sim_pos, u, dt) # simulate robot
        track.append(sim_pos)

        x, y = sim_pos[0, 0], sim_pos[1, 0]
        ekfPos.append((x,y))

        if i % step == 0:
            ekf.predict(u=u)

            if i % ellipse_step == 0:
                pass
                # plot_covariance_ellipse(
                #     (ekf.x[0,0], ekf.x[1,0]), ekf.P[0:2, 0:2], 
                #      std=6, facecolor='k', alpha=0.3)

            x, y = sim_pos[0, 0], sim_pos[1, 0]
            # ekfPos.append((x,y))
            # for lmark in landmarks:
            #     z = z_landmark(lmark, sim_pos,
            #                    std_range, std_bearing)
            #     ekf_update(ekf, z, lmark)

            if i % ellipse_step == 0:
                pass
                # plot_covariance_ellipse(
                #     (ekf.x[0,0], ekf.x[1,0]), ekf.P[0:2, 0:2],
                #     std=6, facecolor='g', alpha=0.8)
    
    track = np.array(track)
    if plot==True:
        plt.plot(track[:, 0], track[:,1], '--',color='k', lw=2)
        plt.plot(x_integ, y_integ, 'g--')
        plt.plot(x_integ_err, y_integ_err, 'r')
        plt.plot(x_grid, y_grid, 'm--')
        plt.axis('equal')
        plt.title("EKF Robot localization")
        plt.legend(['EKF', 'Ground Truth','Naive Interation','Multiscale CAN'])
        if ylim is not None: plt.ylim(*ylim)
        plt.show()

   
    return ekf, np.array(ekfPos).T


'''========================================================================================================================'''
def pathIntegration(speed, angVel):
    q=[0,0,0]
    x_integ,y_integ=[],[]
    for i in range(len(speed)):
        q[0],q[1]=q[0]+speed[i]*np.cos(q[2]), q[1]+speed[i]*np.sin(q[2])
        q[2]+=angVel[i]
        x_integ.append(round(q[0],4))
        y_integ.append(round(q[1],4))

    return x_integ, y_integ

'''Testing 18 paths'''
# errors=[]
# for index in range(18):
#     outfile=f'./results/TestEnvironmentFiles/TraverseInfo/BerlineEnvPath{index}.npz'
#     traverseInfo=np.load(outfile, allow_pickle=True)
#     vel,angVel=traverseInfo['speeds'], traverseInfo['angVel']
#     if len(vel)<500:
#         test_length=len(vel)
#     else:
#         test_length=500
#     x_integ, y_integ=pathIntegration(vel[:test_length], angVel[:test_length])

#     noise=np.random.uniform(0,1,len(vel))
#     vel+=noise

#     x_integ_err, y_integ_err=pathIntegration(vel[:test_length], angVel[:test_length])
   
#     x_grid, y_grid=headDirectionAndPlaceNoWrapNet(test_length, vel, angVel, savePath=None, plot=False, printing=False)

#     dt = 1.0
#     landmarks = array([])
#     ekf,ekfPos = run_localization(
#         landmarks, std_vel=0.1, std_steer=np.radians(1),
#         std_range=0.3, std_bearing=0.1, plot=False)
    
#     integPos=np.array((x_integ, y_integ))
#     integErrPos=np.array((x_integ_err, y_integ_err))
#     canPos=np.array((x_grid,y_grid))

#     outfile=f'./results/TestEnvironmentFiles/ID_{index}_EKFcomparisonUniformErr_0_1.npz'
#     np.savez(outfile,integPos=integPos, canPos=canPos, integErrPos= integErrPos, ekfPos=ekfPos)
    

#     naiiveIntegMSE = ((integPos-integErrPos)**2).mean(axis=1)
#     ekfMSE = ((integPos-ekfPos)**2).mean(axis=1)
#     canMSE = ((integPos-canPos)**2).mean(axis=1)
#     print(f'MSE: naive integ {naiiveIntegMSE}, ekf {ekfMSE}, can {canMSE}')

#     errors.append([np.sum(naiiveIntegMSE), np.sum(ekfMSE), np.sum(canMSE)])
# np.save(f'./results/TestEnvironmentFiles/Errors_EKFcomparisonUniformErr_0_1.npy',np.array(errors))

'''Viewing test results'''
# errors=np.load(f'./results/TestEnvironmentFiles/Errors_EKFcomparisonUniformErr_0_1.npy')
# plt.plot(errors[:,0],'y.-')
# plt.plot(errors[:,1],'k.-')
# plt.plot(errors[:,2],'m.-')
# plt.legend(['Naive Interation', 'EKF', 'Multiscale CAN'])
# plt.title('MSE in position over Multiple Paths')
# plt.show()

# index=3
# outfile=f'./results/TestEnvironmentFiles/ID_{index}_EKFcomparisonUniformErr_0_1.npz'
# ComparisonInfo=np.load(outfile, allow_pickle=True)
# integPos, canPos,integErrPos, ekfPos=ComparisonInfo['integPos'], ComparisonInfo['canPos'], ComparisonInfo['integErrPos'], ComparisonInfo['ekfPos']

# plt.plot(ekfPos[0, :], ekfPos[1,:], '--',color='k', lw=2)
# plt.plot(integPos[0, :], integPos[1,:], 'g--')
# plt.plot(integErrPos[0, :], integErrPos[1,:], 'r')
# plt.plot(canPos[0, :], canPos[1,:], 'm--')
# plt.legend(['EKF', 'Ground Truth','Naive Interation','Multiscale CAN'])
# plt.title('Position estimation Comparison [m]')
# plt.axis('equal')
# plt.show()


'''Testing single path'''
index=0
outfile=f'./results/TestEnvironmentFiles/TraverseInfo/BerlineEnvPath{index}.npz'
traverseInfo=np.load(outfile, allow_pickle=True)
vel,angVel=traverseInfo['speeds'], traverseInfo['angVel']
if len(vel)<500:
    test_length=len(vel)
else:
    test_length=500
x_integ, y_integ=pathIntegration(vel[:test_length], angVel[:test_length])

# noise=np.random.uniform(-1,1,len(vel))
# vel+=noise

x_integ_err, y_integ_err=pathIntegration(vel[:test_length], angVel[:test_length])

x_grid, y_grid=headDirectionAndPlaceNoWrapNet(test_length, vel, angVel, savePath=None, plot=False, printing=True)

dt = 1.0
landmarks = array([])
ekf,ekfPos = run_localization(
landmarks, std_vel=0.1, std_steer=np.radians(1),
std_range=0.3, std_bearing=0.1, plot=True)

integPos=np.array((x_integ, y_integ))
integErrPos=np.array((x_integ_err, y_integ_err))
canPos=np.array((x_grid,y_grid))

outfile=f'./results/TestEnvironmentFiles/Single_ID_{index}_EKFcomparisonUniformErr_-0.5_0.5.npz'
np.savez(outfile,integPos=integPos, canPos=canPos, integErrPos= integErrPos, ekfPos=ekfPos)


naiiveIntegMSE = ((integPos-integErrPos)**2).mean(axis=1)
ekfMSE = ((integPos-ekfPos)**2).mean(axis=1)
canMSE = ((integPos-canPos)**2).mean(axis=1)
print(f'MSE: naive integ {naiiveIntegMSE}, ekf {ekfMSE}, can {canMSE}')





'''============================================================== Particle Filter ============================================================'''
from filterpy.monte_carlo import systematic_resample
from numpy.linalg import norm
from numpy.random import randn
import scipy.stats
from numpy.random import uniform

def create_uniform_particles(x_range, y_range, hdg_range, N):
    particles = np.empty((N, 3))
    particles[:, 0] = uniform(x_range[0], x_range[1], size=N)
    particles[:, 1] = uniform(y_range[0], y_range[1], size=N)
    particles[:, 2] = uniform(hdg_range[0], hdg_range[1], size=N)
    particles[:, 2] %= 2 * np.pi
    return particles

def create_gaussian_particles(mean, std, N):
    particles = np.empty((N, 3))
    particles[:, 0] = mean[0] + (randn(N) * std[0])
    particles[:, 1] = mean[1] + (randn(N) * std[1])
    particles[:, 2] = mean[2] + (randn(N) * std[2])
    particles[:, 2] %= 2 * np.pi
    return particles

def predict(particles, u, std, dt=1.):
    """ move according to control input u (heading change, velocity)
    with noise Q (std heading change, std velocity)`"""

    N = len(particles)
    # update heading
    particles[:, 2] += u[0] + (randn(N) * std[0])
    particles[:, 2] %= 2 * np.pi

    # move in the (noisy) commanded direction
    dist = (u[1] * dt) + (randn(N) * std[1])
    particles[:, 0] += np.cos(particles[:, 2]) * dist
    particles[:, 1] += np.sin(particles[:, 2]) * dist

def update(particles, weights, z, R, landmarks):
    for i, landmark in enumerate(landmarks):
        distance = np.linalg.norm(particles[:, 0:2] - landmark, axis=1)
        weights *= scipy.stats.norm(distance, R).pdf(z[i])

    weights += 1.e-300      # avoid round-off to zero
    weights /= sum(weights) # normalize

def estimate(particles, weights):
    """returns mean and variance of the weighted particles"""

    pos = particles[:, 0:2]
    mean = np.average(pos, weights=weights, axis=0)
    var  = np.average((pos - mean)**2, weights=weights, axis=0)
    return mean, var

def neff(weights):
    return 1. / np.sum(np.square(weights))

def resample_from_index(particles, weights, indexes):
    particles[:] = particles[indexes]
    weights.resize(len(particles))
    weights.fill (1.0 / len(weights))

def run_pf1(N, iters=18, sensor_std_err=.1, 
            do_plot=True, plot_particles=False,
            xlim=(0, 20), ylim=(0, 20),
            initial_x=None):
    landmarks = np.array([[-1, 2], [5, 10], [12,14], [18,21]])
    NL = len(landmarks)
    
    plt.figure()
   
    # create particles and weights
    if initial_x is not None:
        particles = create_gaussian_particles(
            mean=initial_x, std=(5, 5, np.pi/4), N=N)
    else:
        particles = create_uniform_particles((0,20), (0,20), (0, 6.28), N)
    weights = np.ones(N) / N

    if plot_particles:
        alpha = .20
        if N > 5000:
            alpha *= np.sqrt(5000)/np.sqrt(N)           
        plt.scatter(particles[:, 0], particles[:, 1], 
                    alpha=alpha, color='g')
    
    xs = []
    robot_pos = np.array([0., 0.])
    for x in range(iters):
        robot_pos += (1, 1)

        # distance from robot to each landmark
        zs = (norm(landmarks - robot_pos, axis=1) + 
              (randn(NL) * sensor_std_err))

        # move diagonally forward to (x+1, x+1)
        predict(particles, u=(0.00, 1.414), std=(.2, .05))
        
        # incorporate measurements
        update(particles, weights, z=zs, R=sensor_std_err, 
               landmarks=landmarks)
        
        # resample if too few effective particles
        if neff(weights) < N/2:
            indexes = systematic_resample(weights)
            resample_from_index(particles, weights, indexes)
            assert np.allclose(weights, 1/N)
        mu, var = estimate(particles, weights)
        xs.append(mu)

        if plot_particles:
            plt.scatter(particles[:, 0], particles[:, 1], 
                        color='k', marker=',', s=1)
        p1 = plt.scatter(robot_pos[0], robot_pos[1], marker='+',
                         color='k', s=180, lw=3)
        p2 = plt.scatter(mu[0], mu[1], marker='s', color='r')
    
    xs = np.array(xs)
    #plt.plot(xs[:, 0], xs[:, 1])
    plt.legend([p1, p2], ['Actual', 'PF'], loc=4, numpoints=1)
    plt.xlim(*xlim)
    plt.ylim(*ylim)
    print('final position error, variance:\n\t', mu - np.array([iters, iters]), var)
    plt.show()

# from numpy.random import seed
# seed(2) 
# run_pf1(N=5000, plot_particles=False)

'''=====================================================    Kalman Filter  ======================================================================='''

from filterpy.kalman import KalmanFilter
from filterpy.stats import plot_covariance_ellipse
from scipy.linalg import block_diag
from filterpy.common import Q_discrete_white_noise

# R_std = 0.35
# Q_std = 0.04

def tracker1():
    tracker = KalmanFilter(dim_x=4, dim_z=2)
    dt = 1.0   # time step

    tracker.F = np.array([[1, dt, 0,  0],
                          [0,  1, 0,  0],
                          [0,  0, 1, dt],
                          [0,  0, 0,  1]])
    tracker.u = 0.
    tracker.H = np.array([[1/0.3048, 0, 0, 0],
                          [0, 0, 1/0.3048, 0]])

    tracker.R = np.eye(2) * R_std**2
    q = Q_discrete_white_noise(dim=2, dt=dt, var=Q_std**2)
    tracker.Q = block_diag(q, q)
    tracker.x = np.array([[0, 0, 0, 0]]).T
    tracker.P = np.eye(4) * 500.
    return tracker

def plot_measurements(xs, ys=None, dt=None, color='k', lw=1, label='Measurements',
                      lines=False, **kwargs):
    """ Helper function to give a consistent way to display
    measurements in the book.
    """
    if ys is None and dt is not None:
        ys = xs
        xs = np.arange(0, len(ys)*dt, dt)

    plt.autoscale(tight=False)
    if lines:
        if ys is not None:
            return plt.plot(xs, ys, color=color, lw=lw, ls='--', label=label, **kwargs)
        else:
            return plt.plot(xs, color=color, lw=lw, ls='--', label=label, **kwargs)
    else:
        if ys is not None:
            return plt.scatter(xs, ys, edgecolor=color, facecolor='none',
                        lw=2, label=label, **kwargs),
        else:
            return plt.scatter(range(len(xs)), xs, edgecolor=color, facecolor='none',
                        lw=2, label=label, **kwargs),

def plot_filter(xs, ys=None, dt=None, c='C0', label='Filter', var=None, **kwargs):
    """ plot result of KF with color `c`, optionally displaying the variance
    of `xs`. Returns the list of lines generated by plt.plot()"""

    if ys is None and dt is not None:
        ys = xs
        xs = np.arange(0, len(ys) * dt, dt)
    if ys is None:
        ys = xs
        xs = range(len(ys))

    lines = plt.plot(xs, ys, color=c, label=label, **kwargs)
    if var is None:
        return lines

    var = np.asarray(var)
    std = np.sqrt(var)
    std_top = ys+std
    std_btm = ys-std

    plt.plot(xs, ys+std, linestyle=':', color='k', lw=2)
    plt.plot(xs, ys-std, linestyle=':', color='k', lw=2)
    plt.fill_between(xs, std_btm, std_top,
                     facecolor='yellow', alpha=0.2)

    return lines

class PosSensor(object):
    def __init__(self, pos=(0, 0), vel=(0, 0), noise_std=1.):
        self.vel = vel
        self.noise_std = noise_std
        self.pos = [pos[0], pos[1]]
        self.theta=0
        
    def read(self, vel, angVel):
        
        self.pos[0] += (vel*np.cos(self.theta))
        self.pos[1] += (vel*np.sin(self.theta))
        self.theta += angVel

        
        return [self.pos[0] ,
                self.pos[1]]
# simulate robot movement
# N = 300
# sensor = PosSensor((0, 0), (1,np.deg2rad(5)), noise_std=R_std)
# zs = np.array([sensor.read(vel[i], angVel[i]) for i in range(N)])

# # run filter
# robot_tracker = tracker1()
# mu, cov, _, _ = robot_tracker.batch_filter(zs)

# for x, P in zip(mu, cov):
#     # covariance of x and y
#     # cov = np.array([[P[0, 0], P[2, 0]], 
#     #                 [P[0, 2], P[2, 2]]])
#     cov=np.diag(np.array([0.1,0.1]))
#     mean = (x[0, 0], x[2, 0])
    # plot_covariance_ellipse(mean, cov=cov, fc='g', std=3, alpha=0.5)
    
#plot results
# zs *= .3048 # convert to meters
# plot_filter(mu[:, 0], mu[:, 2])
# plot_measurements(zs[:, 0], zs[:, 1])
# plt.legend(loc=2)
# # plt.xlim(0, 20)
# plt.show()
