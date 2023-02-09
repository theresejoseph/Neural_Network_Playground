import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np 
from PIL import Image, ImageFilter
import cv2
import random
import roboticstoolbox as rtb
from roboticstoolbox import DistanceTransformPlanner
import math
from spatialmath.base import *
from math import sin, cos, atan2, radians
from random import uniform
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
from scipy.interpolate import CubicSpline
from numpy.typing import ArrayLike



def processMap(map_path, scale_percent):
    img=np.array(Image.open(map_path).convert("L"))
    imgColor=np.array(Image.open(map_path))

    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    img=cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    imgColor=cv2.resize(imgColor, dim, interpolation = cv2.INTER_AREA)

    imgSharp=cv2.filter2D(img,-1,np.array([[0, -1, 0],[-1, 5,-1],[0, -1, 0]])) 
    imgdia=np.zeros((np.shape(img)))
    imgdia[img==255]=0
    imgdia[img<255]=1
    imgdia=cv2.dilate(imgdia,np.ones((1,1),np.uint8))

    binMap=np.zeros((np.shape(img)))
    binMap[imgSharp < 255] = 0
    binMap[imgSharp==255] = 1
    
    return img, imgColor, imgdia, binMap


def findPathsthroughRandomPoints(img):

    free_spaces=[]
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j]==0:
                free_spaces.append((j,i))

    num_locations=5
    locations=random.choices(free_spaces, k=num_locations)
    print(locations)

    path=[]
    for i in range(len(locations)-1):
        dx = DistanceTransformPlanner(img, goal=locations[i+1], distance="euclidean")
        dx.plan()
        path.extend(dx.query(start=locations[i]))
        print(f"done {i+1} paths")

        outfile='/Users/theresejoseph/Documents/Neural_Network_Playground/results/testEnvMultiplePaths3_5kmrad_100pdi_0.2line.npy'
        np.save(outfile,path)



def rescalePath(path, img, scale, pxlPerMeter):
    #convert path to image
    path_x, path_y = zip(*path)
    pathImg=np.zeros((np.shape(img)))
    pathImg[(path_y, path_x)]=1

    # scale down path image by given percentage 
    width = int(img.shape[1] * scale)
    height = int(img.shape[0] * scale)
    dim = (width, height)
    newImg= cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    pathImgRescaled=cv2.resize(pathImg, dim, interpolation = cv2.INTER_AREA)
    
    #identify new path from rescaled image 
    # pathImgRescaled[pathImgRescaled>0]=1
    # newYpath, newXpath=np.where(pathImgRescaled==1)
    return [np.round(x*scale) for x in path_x], [np.round(y*scale) for y in path_y], newImg, pxlPerMeter*scale 


def remove_consecutive_duplicates(coords):
    # Initialize a new list to store the filtered coordinates
    filtered = []
    # Add the first element to the filtered list
    filtered.append(coords[0])
    # Loop through the remaining elements
    for i in range(1, len(coords)):
        # Check if the current element is different from the previous element
        if coords[i] != coords[i-1]:
        # If it is, add it to the filtered list
            filtered.append(coords[i])
        # if math.tan((coords[i][1]-coords[i-1][1])/((coords[i][0]-coords[i-1][0])))>= 2*np.pi :
            
    # Return the filtered list
    return filtered


normalise_angle = lambda angle: atan2(sin(angle), cos(angle))


class Vehicle:
    def __init__(self, path_x, path_y, throttle, dt,\
        control_gain=5, softening_gain=0.25, yaw_rate_gain=0.35, steering_damp_gain=0.0, max_steer=np.deg2rad(45), \
        c_r: float=0.1, c_a: float=2.0, wheelbase=2.96, \
        overall_length=4.97, overall_width=1.964, rear_overhang=0.0, tyre_diameter=0.4826, \
        tyre_width=0.265, axle_track=1.7):
      
        self.k = control_gain
        self.k_soft = softening_gain
        self.k_yaw_rate = yaw_rate_gain
        self.k_damp_steer = steering_damp_gain
        self.max_steer = max_steer
        self.wheelbase = wheelbase

        self.px = path_x
        self.py = path_y
        self.pyaw=self.calculate_spline_yaw(self.px,self.py)
        self.start_heading=self.pyaw[0]

        self.x= path_x[0]
        self.y= path_y[0]
        self.yaw= self.pyaw[0]
        self.crosstrack_error = None
        self.target_id = 0

        self.v = 0.0
        self.delta = 0.0
        self.omega = 0.0
        self.throttle = throttle

        self.dt = dt
        self.c_r = c_r
        self.c_a = c_a

        self.rear_overhang=0.5 * (overall_length - self.wheelbase)
        rear_axle_to_front_bumper  = overall_length - rear_overhang
        centreline_to_wheel_centre = 0.5 * axle_track
        centreline_to_side         = 0.5 * overall_width
        vehicle_vertices = np.array([
            (-rear_overhang,              centreline_to_side),
            ( rear_axle_to_front_bumper,  centreline_to_side),
            ( rear_axle_to_front_bumper, -centreline_to_side),
            (-rear_overhang,             -centreline_to_side)
        ])
        half_tyre_width            = 0.5 * tyre_width
        centreline_to_inwards_rim  = centreline_to_wheel_centre - half_tyre_width
        centreline_to_outwards_rim = centreline_to_wheel_centre + half_tyre_width
        # Rear right wheel vertices
        wheel_vertices = np.array([
            (-tyre_diameter, -centreline_to_inwards_rim),
            ( tyre_diameter, -centreline_to_inwards_rim),
            ( tyre_diameter, -centreline_to_outwards_rim),
            (-tyre_diameter, -centreline_to_outwards_rim)
        ])
        self.outlines         = np.concatenate([vehicle_vertices, [vehicle_vertices[0]]])
        self.rear_right_wheel = np.concatenate([wheel_vertices,   [wheel_vertices[0]]])
        # Reflect the wheel vertices about the x-axis
        self.rear_left_wheel  = self.rear_right_wheel.copy()
        self.rear_left_wheel[:, 1] *= -1
        # Translate the wheel vertices to the front axle
        front_left_wheel  = self.rear_left_wheel.copy()
        front_right_wheel = self.rear_right_wheel.copy()
        front_left_wheel[:, 0]  += wheelbase 
        front_right_wheel[:, 0] += wheelbase
        get_face_centre = lambda vertices: np.array([
            0.5*(vertices[0][0] + vertices[2][0]),
            0.5*(vertices[0][1] + vertices[2][1])
        ])
        # Translate front wheels to origin
        self.fr_wheel_centre = get_face_centre(front_right_wheel)
        self.fl_wheel_centre = get_face_centre(front_left_wheel)
        self.fr_wheel_origin = front_right_wheel - self.fr_wheel_centre
        self.fl_wheel_origin = front_left_wheel - self.fl_wheel_centre   
    
    def initialise_cubic_spline(self, x: ArrayLike, y: ArrayLike, ds: float, bc_type: str):

        distance = np.concatenate((np.zeros(1), np.cumsum(np.hypot(np.ediff1d(x), np.ediff1d(y)))))
        points = np.array([x, y]).T
        s = np.arange(0, distance[-1], ds)

        try:
            cs = CubicSpline(distance, points, bc_type=bc_type, axis=0, extrapolate=False)
            
        except ValueError as e:
            raise ValueError(f"{e} If you are getting a sequence error, do check if your input dataset contains consecutive duplicate(s).")

        return cs, s

    def calculate_spline_yaw(self, x: ArrayLike, y: ArrayLike, ds: float=0.05, bc_type: str='natural'):
        
        cs, s = self.initialise_cubic_spline(x, y, ds, bc_type)
        dx, dy = cs.derivative(1)(s).T
        return np.arctan2(dy, dx)
   
    def find_target_path_id(self, x, y, yaw):  
        # Calculate position of the front axle
        fx = x + self.wheelbase * cos(yaw)
        fy = y + self.wheelbase * sin(yaw)

        dx = fx - self.px    # Find the x-axis of the front axle relative to the path
        dy = fy - self.py    # Find the y-axis of the front axle relative to the path

        d = np.hypot(dx, dy) # Find the distance from the front axle to the path
        target_index = np.argmin(d) # Find the shortest distance in the array

        return target_index, dx[target_index], dy[target_index], d[target_index]

    def calculate_yaw_term(self, target_index, yaw):
        yaw_error = normalise_angle(self.pyaw[target_index] - yaw)

        return yaw_error

    def calculate_crosstrack_term(self, target_velocity, yaw, dx, dy, absolute_error):
        front_axle_vector = np.array([sin(yaw), -cos(yaw)])
        nearest_path_vector = np.array([dx, dy])
        crosstrack_error = np.sign(nearest_path_vector@front_axle_vector) * absolute_error

        crosstrack_steering_error = atan2((self.k * crosstrack_error), (self.k_soft + target_velocity))

        return crosstrack_steering_error, crosstrack_error

    def calculate_yaw_rate_term(self, target_velocity, steering_angle):
        yaw_rate_error = self.k_yaw_rate*(-target_velocity*sin(steering_angle))/self.wheelbase

        return yaw_rate_error

    def calculate_steering_delay_term(self, computed_steering_angle, previous_steering_angle):
        steering_delay_error = self.k_damp_steer*(computed_steering_angle - previous_steering_angle)

        return steering_delay_error

    def stanley_control(self, x, y, yaw, target_velocity, steering_angle):
        target_index, dx, dy, absolute_error = self.find_target_path_id(x, y, yaw)
        yaw_error = self.calculate_yaw_term(target_index, yaw)
        crosstrack_steering_error, crosstrack_error = self.calculate_crosstrack_term(target_velocity, yaw, dx, dy, absolute_error)
        yaw_rate_damping = self.calculate_yaw_rate_term(target_velocity, steering_angle)
        
        desired_steering_angle = yaw_error + crosstrack_steering_error + yaw_rate_damping

        # Constrains steering angle to the vehicle limits
        desired_steering_angle += self.calculate_steering_delay_term(desired_steering_angle, steering_angle)
        limited_steering_angle = np.clip(desired_steering_angle, -self.max_steer, self.max_steer)

        return limited_steering_angle, target_index, crosstrack_error
        
    def kinematic_model(self, x: float, y: float, yaw: float, velocity: float, throttle: float, steering_angle: float):
        # Compute the local velocity in the x-axis
        friction     = velocity * (self.c_r + self.c_a*velocity)
        new_velocity = velocity + self.dt*(throttle - friction)

        # Limit steering angle to physical vehicle limits
        steering_angle = -self.max_steer if steering_angle < -self.max_steer else self.max_steer if steering_angle > self.max_steer else steering_angle

        # Compute the angular velocity
        angular_velocity = velocity*np.tan(steering_angle) / self.wheelbase

        # Compute the final state using the discrete time model
        new_x   = x + velocity*np.cos(yaw)*self.dt
        new_y   = y + velocity*np.sin(yaw)*self.dt
        new_yaw = normalise_angle(yaw + angular_velocity*self.dt)
        
        return new_x, new_y, new_yaw, new_velocity, steering_angle, angular_velocity

    def get_rotation_matrix(_, angle: float) -> np.ndarray:
        cos_angle = cos(angle)
        sin_angle = sin(angle)

        return np.array([
            ( cos_angle, sin_angle),
            (-sin_angle, cos_angle)
        ])

    def transform(self, point: np.ndarray) -> np.ndarray:
        # Vector rotation
        point = point.dot(self.yaw_vector).T

        # Vector translation
        point[0, :] += self.x
        point[1, :] += self.y
        
        return point

    def plot_car(self, x: float, y: float, yaw: float, steer: float):
        self.x = x
        self.y = y
        
        # Rotation matrices
        self.yaw_vector = self.get_rotation_matrix(yaw)
        steer_vector    = self.get_rotation_matrix(steer)

        # Rotate the wheels about its position
        front_right_wheel  = self.fr_wheel_origin.copy()
        front_left_wheel   = self.fl_wheel_origin.copy()
        front_right_wheel  = front_right_wheel@steer_vector
        front_left_wheel   = front_left_wheel@steer_vector
        front_right_wheel += self.fr_wheel_centre
        front_left_wheel  += self.fl_wheel_centre

        outlines          = self.transform(self.outlines)
        rear_right_wheel  = self.transform(self.rear_right_wheel)
        rear_left_wheel   = self.transform(self.rear_left_wheel)
        front_right_wheel = self.transform(front_right_wheel)
        front_left_wheel  = self.transform(front_left_wheel)

        return outlines, front_right_wheel, rear_right_wheel, front_left_wheel, rear_left_wheel
    
    def drive(self):
        # throttle = 300 #uniform(50, 200)
        self.delta, self.target_id, self.crosstrack_error = self.stanley_control(self.x, self.y, self.yaw, self.v, self.delta)
        self.x, self.y, self.yaw, self.v, _, ang_vel = self.kinematic_model(self.x, self.y, self.yaw, self.v, self.throttle, self.delta)

        # print(f"Cross-track term: {self.crosstrack_error}{' '*10}", end="\r")
        
        velocity.append(self.v*self.dt)
        angVel.append(ang_vel*self.dt)
        trueCarPos.append(tuple((self.x,  self.y)))


# '''drive without visualisation'''
def noVisualisationDrive():
    global velocity, angVel, trueCarPos
    # Storage Variables
    velocity=[]
    angVel=[]
    trueCarPos=[]

    # car object
    dt=0.005
    frames=20000
    car  = Vehicle(path_x, path_y,100, dt, control_gain=5, softening_gain=0.25, yaw_rate_gain=0.4, 
    steering_damp_gain=0.0, max_steer=np.deg2rad(60), c_r=0.1, c_a=2.0)
    
    for i in range(frames):
        # Drive and draw car
        if (car.px[car.target_id], car.py[car.target_id]) !=(car.px[-1], car.py[-1]):
            car.drive()
            print(car.x, car.y)
        else:
            car.v=0
    
    outfile='./results/TestEnvironmentFiles/TraverseInfo/EnvPath.npz'
    np.savez(outfile,speeds=velocity, angVel=angVel, truePos= trueCarPos, startPose=np.array([path_x[0], path_y[0],car.start_heading]))
    
        

def runSimulation(path_x, path_y, path_img):
    global velocity, angVel, trueCarPos
    def animate(frame):
        global velocity, angVel, trueCarPos
        # Camera tracks car
        ax.set_xlim(car.x - map_size_x, car.x + map_size_x)
        ax.set_ylim(car.y - map_size_y, car.y + map_size_y)

        # Drive and draw car
        if (car.px[car.target_id], car.py[car.target_id]) !=(car.px[-1], car.py[-1]):
            car.drive()
            outline_plot, fr_plot, rr_plot, fl_plot, rl_plot = car.plot_car(car.x, car.y, car.yaw, car.delta)
            car_outline.set_data(*outline_plot)
            front_right_wheel.set_data(*fr_plot)
            rear_right_wheel.set_data(*rr_plot)
            front_left_wheel.set_data(*fl_plot)
            rear_left_wheel.set_data(*rl_plot)
            rear_axle.set_data(car.x, car.y)
        else:
            car.v=0
        
        # Show car's target
        target.set_data(car.px[car.target_id], car.py[car.target_id])

        # Annotate car's coordinate above car
        annotation.set_text(f'{car.x:.1f}, {car.y:.1f}')
        annotation.set_position((car.x, car.y + 5))

        plt.title(f'{car.dt*frame:.2f}s', loc='right')
        plt.xlabel(f'Speed: {car.v:.2f} m/s', loc='left')
        # plt.savefig(f'image/visualisation_{frame:03}.png', dpi=300)

        return car_outline, front_right_wheel, rear_right_wheel, front_left_wheel, rear_left_wheel, rear_axle, target,

    # Storage Variables
    velocity=[]
    angVel=[]
    trueCarPos=[]

    # Simulation Parameters
    fps = 20
    dt = 1/fps
    map_size_x = 70
    map_size_y = 40
    frames = 5000
    loop = False
    car  = Vehicle(path_x, path_y,100, dt)
    interval = car.dt * 10**3

    fig = plt.figure()
    ax = plt.axes()
    ax.set_aspect('equal')

    ax.plot(path_x, path_y, '--', color='gold')
    ax.imshow(path_img, cmap='gray')

    empty              = ([], [])
    target,            = ax.plot(*empty, '+r')
    car_outline,       = ax.plot(*empty, color='blue')
    front_right_wheel, = ax.plot(*empty, color='blue')
    rear_right_wheel,  = ax.plot(*empty, color='blue')
    front_left_wheel,  = ax.plot(*empty, color='blue')
    rear_left_wheel,   = ax.plot(*empty, color='blue')
    rear_axle,         = ax.plot(car.x, car.y, '+', color='black', markersize=2)
    annotation         = ax.annotate(f'{car.x:.1f}, {car.y:.1f}', xy=(car.x, car.y + 5), color='black', annotation_clip=False)


    ani = FuncAnimation(fig, animate, frames=frames, init_func=lambda: None, interval=interval, repeat=loop)
        # anim.save('animation.gif', writer='imagemagick', fps=50)
    plt.grid()
    plt.show()


    # outfile1='./results/TestEnvironmentFiles/TraverseInfo/EnvPathSpeed.npy'
    # np.save(outfile1,velocity)
    # outfile2='./results/TestEnvironmentFiles/TraverseInfo/EnvPathAngvel.npy'
    # np.save(outfile2,angVel)
    outfile='./results/TestEnvironmentFiles/TraverseInfo/EnvPath.npz'
    np.savez(outfile,speeds=velocity, angVel=angVel, truePos= trueCarPos, startPose=np.array([path_x[0], path_y[0],car.start_heading]))
    

def pathIntegration(speed, angVel, startPose):
    q=startPose
    x_integ,y_integ=[],[]
    for i in range(len(speed)):
        q[0],q[1]=q[0]+speed[i]*np.cos(q[2]), q[1]+speed[i]*np.sin(q[2])
        q[2]+=angVel[i]
        x_integ.append(round(q[0],4))
        y_integ.append(round(q[1],4))

    return x_integ, y_integ


'''Initialising Image'''
map_path = './results/TestingMaps/berlin_5kmrad_0.2Line_100pdi.png'
img=np.array(Image.open(map_path).convert("L"))

meterWidth=5000
pxlPerMeter= img.shape[0]/meterWidth

img[img<255]= 0 
img[img==255]=1

'''Generate Paths'''
# findPathsthroughRandomPoints(img)


'''Original'''
pathfile='results/TestEnvironmentFiles/Paths/testEnvMultiplePaths3_5kmrad_100pdi_0.2line.npy'


'''Scaled'''
scale=1
path_x, path_y, path_img, currentPxlPerMeter= rescalePath(np.load(pathfile), img, scale, pxlPerMeter)
print(f"scaled width{np.shape(path_img)[0], np.shape(path_img)[1]}, pxlPerMeter{np.shape(path_img)[0]/meterWidth, np.shape(path_img)[1]/meterWidth}")
# plt.imshow(path_img, cmap='gray')
# plt.plot(path_x, path_y,'r.')
# plt.show()
path= remove_consecutive_duplicates(list(zip(path_x, path_y)))
path_x, path_y = zip(*path)


'''Run Simulation'''
# runSimulation(path_x, path_y, path_img)

# noVisualisationDrive()

'''Test Stored Traverse'''

# outfile='./results/TestEnvironmentFiles/TraverseInfo/EnvPath.npz'
# traverseInfo=np.load(outfile)
# speeds,angVel,truePos, startPose=traverseInfo['speeds'], traverseInfo['angVel'], traverseInfo['truePos'], traverseInfo['startPose']

# x_integ,y_integ=pathIntegration(speeds, angVel, startPose)
# x,y=zip(*truePos)

# plt.plot(x_integ, y_integ, 'm.')
# plt.plot(x, y, 'g.-')
# plt.plot(path_x, path_y, 'y--')
# plt.show()


'''GPT kinematics model'''
def initialise_cubic_spline( x: ArrayLike, y: ArrayLike, ds: float, bc_type: str):

    distance = np.concatenate((np.zeros(1), np.cumsum(np.hypot(np.ediff1d(x), np.ediff1d(y)))))
    points = np.array([x, y]).T
    s = np.arange(0, distance[-1], ds)

    try:
        cs = CubicSpline(distance, points, bc_type=bc_type, axis=0, extrapolate=False)
        
    except ValueError as e:
        raise ValueError(f"{e} If you are getting a sequence error, do check if your input dataset contains consecutive duplicate(s).")

    return cs, s

def calculate_spline_yaw( x: ArrayLike, y: ArrayLike, ds: float, bc_type: str='natural'):
    
    cs, s = initialise_cubic_spline(x, y, ds, bc_type)
    dx, dy = cs.derivative(1)(s).T
    return np.arctan2(dy, dx)

def update_state(state, inputs, dt):
    x, y, theta, v = state
    delta, a = inputs
    x = x + v * np.cos(theta) * dt
    y = y + v * np.sin(theta) * dt
    theta = theta + v / Lf * np.tan(delta) * dt
    v = v + a * dt
    return x, y, theta, v

def cost_function(state, inputs, desired_state):
    x, y, theta, v = update_state(state, inputs, dt)
    x_d, y_d, theta_d, v_d = desired_state
    return np.sum((x - x_d)**2 + (y - y_d)**2 + (theta - theta_d)**2 + (v - v_d)**2)

def optimal_controller(state, desired_state, dt):
    inputs = np.random.randn(2) # Initialize inputs randomly
    for i in range(1000): # Perform optimization for 1000 iterations
        gradient = np.gradient(cost_function(state, inputs, desired_state))
        print(gradient)
        inputs = inputs - gradient
    return inputs

dt = 0.1 # Time step
Lf = 2.67 # Length of front wheel to center of gravity
yaw=calculate_spline_yaw(path_x,path_y,ds=dt)
state = np.array([path_x[0], path_y[0], yaw[0], 0]) # Initial state
pos=[]
for i in range(100):
    desired_state = np.array([path_x[i], path_y[i], yaw[i], 1]) # Desired state

    inputs = optimal_controller(state, desired_state, dt)

    state=update_state(state, inputs)
    pos.append(tuple((state[0], state[1])))

x,y=zip(*pos)
plt.plot(x,y, 'g.-')
plt.plot(path_x, path_y, 'y--')
plt.show()


