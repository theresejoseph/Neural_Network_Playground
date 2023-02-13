import matplotlib.pyplot as plt
import numpy as np
import random  
import math



'''========================================================================================================================'''

class ExtendedKalmanFilter:
    def __init__(self, state_dim, measurement_dim, process_noise_cov, measurement_noise_cov):
        self.state_dim = state_dim
        self.measurement_dim = measurement_dim
        self.process_noise_cov = process_noise_cov
        self.measurement_noise_cov = measurement_noise_cov
        self.state_estimate = np.zeros((state_dim, 1))
        self.covariance_estimate = np.eye(state_dim)

    def predict(self, state_transition_matrix, control_input):
        self.state_estimate = np.dot(state_transition_matrix, self.state_estimate) + control_input
        self.covariance_estimate = np.dot(np.dot(state_transition_matrix, self.covariance_estimate), state_transition_matrix.T) + self.process_noise_cov

    def update(self, measurement, measurement_model):
        innovation = measurement - np.dot(measurement_model, self.state_estimate)
        innovation_cov = np.dot(np.dot(measurement_model, self.covariance_estimate), measurement_model.T) + self.measurement_noise_cov
        kalman_gain = np.dot(np.dot(self.covariance_estimate, measurement_model.T), np.linalg.inv(innovation_cov))
        self.state_estimate = self.state_estimate + np.dot(kalman_gain, innovation)
        self.covariance_estimate = self.covariance_estimate - np.dot(np.dot(kalman_gain, measurement_model), self.covariance_estimate)

def EKF_main(speed, angVel):
    state_dim = 4
    measurement_dim = 2
    process_noise_cov = np.eye(state_dim)
    measurement_noise_cov = np.eye(measurement_dim)

    ekf = ExtendedKalmanFilter(state_dim, measurement_dim, process_noise_cov, measurement_noise_cov)

    state_transition_matrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    # control_input = np.zeros((state_dim, 1))
    # control_input = np.array([[speed], [speed], [0], [angVel]])
    measurement_model = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
    # measurement = np.array([[0], [0]])

    ekf_pos=[]
    error=1.1
    q=[0,0,0]
    for i in range(len(speed)):
        speed[i]*=error
        q[0],q[1]=q[0]+speed[i]*np.cos(q[2]), q[1]+speed[i]*np.sin(q[2])
        q[2]+=angVel[i]
        control_input = np.array([[speed[i]], [0], [0], [angVel[i]]])
        measurement = np.array([[q[0]], [q[1]]])

        ekf.predict(state_transition_matrix, control_input)
        ekf.update(measurement, measurement_model)
        ekf_pos.append((ekf.state_estimate[0][0], ekf.state_estimate[1][0]))

    return ekf_pos

def particle_filter(speeds,angVel, num_particles=1000):
    particle_pos=[]
    # Initialize particles randomly in the space
    particles = np.random.rand(num_particles, 2)

    # Loop over each time step
    for i in range(len(angVel)):
        # Predict the new position of each particle based on the current velocity
        angle = np.random.normal(angVel[i], 0.1, num_particles)
        speed = np.random.normal(speeds[i], 0.1, num_particles)
        particles[:, 0] += np.cos(angle) * speed
        particles[:, 1] += np.sin(angle) * speed

        # Weight the particles based on some measurement model
        # In this example, we assume that the measurement is the true position
        weights = np.ones(num_particles)

        # Resample the particles based on the weights
        new_particles = np.zeros((num_particles, 2))
        new_indices = np.random.choice(num_particles, num_particles, p=weights/sum(weights), replace=True)
        for j, index in enumerate(new_indices):
            new_particles[j,:] = particles[index,:]
        
        particles = new_particles

        # Estimate the position based on the weighted average of the particles
        estimated_position = np.mean(particles, axis=0)
        particle_pos.append(estimated_position)
    
    return particle_pos

def pathIntegration(speed, angVel):
    q=[0,0,0]
    x_integ,y_integ=[],[]
    for i in range(len(speed)):
        q[0],q[1]=q[0]+speed[i]*np.cos(q[2]), q[1]+speed[i]*np.sin(q[2])
        q[2]+=angVel[i]
        x_integ.append(round(q[0],4))
        y_integ.append(round(q[1],4))

    return x_integ, y_integ

# test_length=1000
# particle_pos=particle_filter( vel[1:test_length], angVel[1:test_length])
# particle_x,particle_y=zip(*particle_pos)
# x_integ, y_integ=pathIntegration(vel[1:test_length],angVel[1:test_length])
# ekf_pos=EKF_main(vel[1:test_length], angVel[1:test_length])
# ekf_x,ekf_y=zip(*ekf_pos)


# plt.plot(x_integ, y_integ,'g.-')
# plt.plot(ekf_x,ekf_y, 'm-.')
# # plt.plot(particle_x,particle_y,'b.-')
# plt.axis('equal')
# plt.legend(('Path Integration', 'EKF', 'Particle Filter'))
# plt.show()

'''========================================================================================================================'''
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

from numpy.random import seed
seed(2) 
run_pf1(N=5000, plot_particles=False)