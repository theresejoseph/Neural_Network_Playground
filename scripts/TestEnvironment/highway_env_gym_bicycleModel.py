import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np 
import random
from typing import Tuple, Dict, Callable, List, Optional, Union, Sequence
import numpy as np
from numpy.linalg import LinAlgError
from collections import deque
import control

Vector = Union[np.ndarray, Sequence[float]]
Matrix = Union[np.ndarray, Sequence[Sequence[float]]]
Interval = Union[np.ndarray,
                 Tuple[Vector, Vector],
                 Tuple[Matrix, Matrix],
                 Tuple[float, float],
                 List[Vector],
                 List[Matrix],
                 List[float]]

def intervals_scaling(a: Interval, b: Interval) -> np.ndarray:
    """
    Scale an intervals
    :param a: matrix a
    :param b: interval [b_min, b_max]
    :return: the interval of their product ab
    """
    p = lambda x: np.maximum(x, 0)
    n = lambda x: np.maximum(-x, 0)
    return np.array(
        [np.dot(p(a), b[0]) - np.dot(n(a), b[1]),
         np.dot(p(a), b[1]) - np.dot(n(a), b[0])])

def is_metzler(matrix: np.ndarray, eps: float = 1e-9) -> bool:
    return (matrix - np.diag(np.diag(matrix)) >= -eps).all()

class LPV(object):
    def __init__(self,
                 x0: Vector,
                 a0: Matrix,
                 da: List[Vector],
                 b: Matrix = None,
                 d: Matrix = None,
                 omega_i: Matrix = None,
                 u: Vector = None,
                 k: Matrix = None,
                 center: Vector = None,
                 x_i: Matrix = None) -> None:
        """
        A Linear Parameter-Varying system:
        dx = (a0 + sum(da))(x - center) + bd + c
        :param x0: initial state
        :param a0: nominal dynamics
        :param da: list of dynamics deviations
        :param b: control matrix
        :param d: perturbation matrix
        :param omega_i: perturbation bounds
        :param u: constant known control
        :param k: linear feedback: a0 x + bu -> (a0+bk)x + b(u-kx), where a0+bk is stable
        :param center: asymptotic state
        :param x_i: initial state interval
        """
        self.x0 = np.array(x0, dtype=float)
        self.a0 = np.array(a0, dtype=float)
        self.da = [np.array(da_i) for da_i in da]
        self.b = np.array(b) if b is not None else np.zeros((*self.x0.shape, 1))
        self.d = np.array(d) if d is not None else np.zeros((*self.x0.shape, 1))
        self.omega_i = np.array(omega_i) if omega_i is not None else np.zeros((2, 1))
        self.u = np.array(u) if u is not None else np.zeros((1,))
        self.k = np.array(k) if k is not None else np.zeros((self.b.shape[1], self.b.shape[0]))
        self.center = np.array(center) if center is not None else np.zeros(self.x0.shape)

        # Closed-loop dynamics
        self.a0 += self.b @ self.k

        self.coordinates = None

        self.x_t = self.x0
        self.x_i = np.array(x_i) if x_i is not None else np.array([self.x0, self.x0])
        self.x_i_t = None

        self.update_coordinates_frame(self.a0)

    def update_coordinates_frame(self, a0: np.ndarray) -> None:
        """
        Ensure that the dynamics matrix A0 is Metzler.
        If not, design a coordinate transformation and apply it to the model and state interval.
        :param a0: the dynamics matrix A0
        """
        self.coordinates = None
        # Rotation
        if not is_metzler(a0):
            eig_v, transformation = np.linalg.eig(a0)
            if np.isreal(eig_v).all():
                try:
                    self.coordinates = (transformation, np.linalg.inv(transformation))
                except LinAlgError:
                    pass
            if not self.coordinates:
                print("Non Metzler A0 with eigenvalues: ", eig_v)
        else:
            self.coordinates = (np.eye(a0.shape[0]), np.eye(a0.shape[0]))

        # Forward coordinates change of states and models
        self.a0 = self.change_coordinates(self.a0, matrix=True)
        self.da = self.change_coordinates(self.da, matrix=True)
        self.b = self.change_coordinates(self.b, offset=False)
        self.x_i_t = np.array(self.change_coordinates([x for x in self.x_i]))

    def set_control(self, control: np.ndarray, state: np.ndarray = None) -> None:
        if state is not None:
            control = control - self.k @ state  # the Kx part of the control is already present in A0.
        self.u = control

    def change_coordinates(self, value: Union[np.ndarray, List[np.ndarray]], matrix: bool = False, back: bool = False,
                           interval: bool = False, offset: bool = True) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Perform a change of coordinate: rotation and centering.
        :param value: the object to transform
        :param matrix: is it a matrix or a vector?
        :param back: if True, transform back to the original coordinates
        :param interval: when transforming an interval, lossy interval arithmetic must be used to preserve the inclusion
                         property.
        :param offset: should we apply the centering or not
        :return: the transformed object
        """
        if self.coordinates is None:
            return value
        transformation, transformation_inv = self.coordinates
        if interval:
            if back:
                value = intervals_scaling(transformation, value[:, :, np.newaxis]).squeeze() + \
                        offset * np.array([self.center, self.center])
                return value
            else:
                value = value - offset * np.array([self.center, self.center])
                value = intervals_scaling(transformation_inv, value[:, :, np.newaxis]).squeeze()
                return value
        elif matrix:  # Matrix
            if back:
                return transformation @ value @ transformation_inv
            else:
                return transformation_inv @ value @ transformation
        elif isinstance(value, list):  # List
            return [self.change_coordinates(v, back) for v in value]
        else:
            if back:
                value = transformation @ value
                if offset:
                    value += self.center
                return value
            else:
                if offset:
                    value -= self.center
                return transformation_inv @ value

    def step(self, dt: float) -> None:
        if is_metzler(self.a0):
            self.x_i_t = self.step_interval_predictor(self.x_i_t, dt)
        else:
            self.x_i_t = self.step_naive_predictor(self.x_i_t, dt)
        dx = self.a0 @ self.x_t + self.b @ self.u.squeeze(-1)
        self.x_t = self.x_t + dx * dt

    def step_naive_predictor(self, x_i: Interval, dt: float) -> np.ndarray:
        """
        Step an interval predictor with box uncertainty.
        :param x_i: state interval at time t
        :param dt: time step
        :return: state interval at time t+dt
        """
        a0, da, d, omega_i, b, u = self.a0, self.da, self.d, self.omega_i, self.b, self.u
        a_i = a0 + sum(intervals_product([0, 1], [da_i, da_i]) for da_i in da)
        bu = (b @ u).squeeze(-1)
        dx_i = intervals_product(a_i, x_i) + intervals_product([d, d], omega_i) + np.array([bu, bu])
        return x_i + dx_i*dt

    def step_interval_predictor(self, x_i: Interval, dt: float) -> np.ndarray:
        """
        Step an interval predictor with polytopic uncertainty.
        :param x_i: state interval at time t
        :param dt: time step
        :return: state interval at time t+dt
        """
        a0, da, d, omega_i, b, u = self.a0, self.da, self.d, self.omega_i, self.b, self.u
        p = lambda x: np.maximum(x, 0)
        n = lambda x: np.maximum(-x, 0)
        da_p = sum(p(da_i) for da_i in da)
        da_n = sum(n(da_i) for da_i in da)
        x_m, x_M = x_i[0, :, np.newaxis], x_i[1, :, np.newaxis]
        o_m, o_M = omega_i[0, :, np.newaxis], omega_i[1, :, np.newaxis]
        dx_m = a0 @ x_m - da_p @ n(x_m) - da_n @ p(x_M) + p(d) @ o_m - n(d) @ o_M + b @ u
        dx_M = a0 @ x_M + da_p @ p(x_M) + da_n @ n(x_m) + p(d) @ o_M - n(d) @ o_m + b @ u
        dx_i = np.array([dx_m.squeeze(axis=-1), dx_M.squeeze(axis=-1)])
        return x_i + dx_i * dt

class BicycleVehicle():
    """
    A dynamical bicycle model, with tire friction and slipping.
    
    See Chapter 2 of Lateral Vehicle Dynamics. Vehicle Dynamics and Control. Rajamani, R. (2011)
    """
    LENGTH = 5.0
    WIDTH = 2.0
    DEFAULT_INITIAL_SPEEDS = [23, 25]
    MASS: float = 1  # [kg]
    LENGTH_A: float = LENGTH / 2  # [m]
    LENGTH_B: float = LENGTH / 2  # [m]
    INERTIA_Z: float = 1/12 * MASS * (LENGTH ** 2 + WIDTH ** 2)  # [kg.m2]
    FRICTION_FRONT: float = 15.0 * MASS  # [N]
    FRICTION_REAR: float = 15.0 * MASS  # [N]

    MAX_ANGULAR_SPEED: float = 2 * np.pi  # [rad/s]
    MAX_SPEED: float = 15  # [m/s]
    HISTORY_SIZE = 30
    DEFAULT_INITIAL_SPEEDS = [23, 25]
    MIN_SPEED = -40.
    def __init__(self, position, heading: float = 0, speed: float = 0) -> None:
        self.position=position 
        self.speed=speed
        self.heading=heading 
        self.lateral_speed = 0
        self.yaw_rate = 0
        self.theta = None
        self.A_lat, self.B_lat = self.lateral_lpv_dynamics()

       
        self.action = {'steering': 0, 'acceleration': 0}
        self.crashed = False
        self.impact = None
        self.log = []
        self.history = deque(maxlen=self.HISTORY_SIZE)

    @property
    def state(self) -> np.ndarray:
        return np.array([[self.position[0]],
                         [self.position[1]],
                         [self.heading],
                         [self.speed],
                         [self.lateral_speed],
                         [self.yaw_rate]])

    @property
    def derivative(self) -> np.ndarray:
        """
        See Chapter 2 of Lateral Vehicle Dynamics. Vehicle Dynamics and Control. Rajamani, R. (2011)
        :return: the state derivative
        """
        delta_f = self.action["steering"]
        delta_r = 0
        theta_vf = np.arctan2(self.lateral_speed + self.LENGTH_A * self.yaw_rate, self.speed)  # (2.27)
        theta_vr = np.arctan2(self.lateral_speed - self.LENGTH_B * self.yaw_rate, self.speed)  # (2.28)
        f_yf = 2*self.FRICTION_FRONT * (delta_f - theta_vf)  # (2.25)
        f_yr = 2*self.FRICTION_REAR * (delta_r - theta_vr)  # (2.26)
        if abs(self.speed) < 1:  # Low speed dynamics: damping of lateral speed and yaw rate
            f_yf = - self.MASS * self.lateral_speed - self.INERTIA_Z/self.LENGTH_A * self.yaw_rate
            f_yr = - self.MASS * self.lateral_speed + self.INERTIA_Z/self.LENGTH_A * self.yaw_rate
        d_lateral_speed = 1/self.MASS * (f_yf + f_yr) - self.yaw_rate * self.speed  # (2.21)
        d_yaw_rate = 1/self.INERTIA_Z * (self.LENGTH_A * f_yf - self.LENGTH_B * f_yr)  # (2.22)
        c, s = np.cos(self.heading), np.sin(self.heading)
        R = np.array(((c, -s), (s, c)))
        speed = R @ np.array([self.speed, self.lateral_speed])
        return np.array([[speed[0]],
                         [speed[1]],
                         [self.yaw_rate],
                         [self.action['acceleration']],
                         [d_lateral_speed],
                         [d_yaw_rate]])

    @property
    def derivative_linear(self) -> np.ndarray:
        """
        Linearized lateral dynamics.
            
        This model is based on the following assumptions:
        - the vehicle is moving with a constant longitudinal speed
        - the steering input to front tires and the corresponding slip angles are small
        
        See https://pdfs.semanticscholar.org/bb9c/d2892e9327ec1ee647c30c320f2089b290c1.pdf, Chapter 3.
        """
        x = np.array([[self.lateral_speed], [self.yaw_rate]])
        u = np.array([[self.action['steering']]])
        self.A_lat, self.B_lat = self.lateral_lpv_dynamics()
        dx = self.A_lat @ x + self.B_lat @ u
        c, s = np.cos(self.heading), np.sin(self.heading)
        R = np.array(((c, -s), (s, c)))
        speed = R @ np.array([self.speed, self.lateral_speed])
        return np.array([[speed[0]], [speed[1]], [self.yaw_rate], [self.action['acceleration']], dx[0], dx[1]])

    def step(self, dt: float) -> None:
        self.clip_actions()
        derivative = self.derivative
        self.position += derivative[0:2, 0] * dt
        self.heading += self.yaw_rate * dt
        self.speed += self.action['acceleration'] * dt
        self.lateral_speed += derivative[4, 0] * dt
        self.yaw_rate += derivative[5, 0] * dt

        self.on_state_update()

    def clip_actions(self) -> None:
        if self.crashed:
            self.action['steering'] = 0
            self.action['acceleration'] = -1.0*self.speed
        self.action['steering'] = float(self.action['steering'])
        self.action['acceleration'] = float(self.action['acceleration'])
        if self.speed > self.MAX_SPEED:
            self.action['acceleration'] = min(self.action['acceleration'], 1.0 * (self.MAX_SPEED - self.speed))
        elif self.speed < self.MIN_SPEED:
            self.action['acceleration'] = max(self.action['acceleration'], 1.0 * (self.MIN_SPEED - self.speed))

        # Required because of the linearisation
        self.action["steering"] = np.clip(self.action["steering"], -np.pi/2, np.pi/2)
        self.yaw_rate = np.clip(self.yaw_rate, -self.MAX_ANGULAR_SPEED, self.MAX_ANGULAR_SPEED)

    def lateral_lpv_structure(self):
        """
        State: [lateral speed v, yaw rate r]
        :return: lateral dynamics A0, phi, B such that dx = (A0 + theta^T phi)x + B u
        """
        B = np.array([
            [2*self.FRICTION_FRONT / self.MASS],
            [self.FRICTION_FRONT * self.LENGTH_A / self.INERTIA_Z]
        ])

        speed_body_x = self.speed
        A0 = np.array([
            [0, -speed_body_x],
            [0, 0]
        ])

        if abs(speed_body_x) < 1:
            return A0, np.zeros((2, 2, 2)), B*0

        phi = np.array([
            [
                [-2 / (self.MASS*speed_body_x), -2*self.LENGTH_A / (self.MASS*speed_body_x)],
                [-2*self.LENGTH_A / (self.INERTIA_Z*speed_body_x), -2*self.LENGTH_A**2 / (self.INERTIA_Z*speed_body_x)]
            ], [
                [-2 / (self.MASS*speed_body_x), 2*self.LENGTH_B / (self.MASS*speed_body_x)],
                [2*self.LENGTH_B / (self.INERTIA_Z*speed_body_x), -2*self.LENGTH_B**2 / (self.INERTIA_Z*speed_body_x)]
            ],
        ])
        return A0, phi, B
    
    def act(self, action: Union[dict, str] = None) -> None:
        """
        Store an action to be repeated.
        :param action: the input action
        """
        if action:
            self.action = action

    def step(self, dt: float) -> None:
        """
        Propagate the vehicle state given its actions.
        Integrate a modified bicycle model with a 1st-order response on the steering wheel dynamics.
        If the vehicle is crashed, the actions are overridden with erratic steering and braking until complete stop.
        The vehicle's current lane is updated.
        :param dt: timestep of integration of the model [s]
        """
        self.clip_actions()
        delta_f = self.action['steering']
        beta = np.arctan(1 / 2 * np.tan(delta_f))
        v = self.speed * np.array([np.cos(self.heading + beta),
                                    np.sin(self.heading + beta)])
        self.position += v * dt
        if self.impact is not None:
            self.position += self.impact
            self.crashed = True
            self.impact = None
        self.heading += self.speed * np.sin(beta) / (self.LENGTH / 2) * dt
        self.speed += self.action['acceleration'] * dt
        self.on_state_update()


    def lateral_lpv_dynamics(self):
        """
        State: [lateral speed v, yaw rate r]
        :return: lateral dynamics A, B
        """
        A0, phi, B = self.lateral_lpv_structure()
        self.theta = np.array([self.FRICTION_FRONT, self.FRICTION_REAR])
        A = A0 + np.tensordot(self.theta, phi, axes=[0, 0])
        return A, B

    def full_lateral_lpv_structure(self):
        """
        State: [position y, yaw psi, lateral speed v, yaw rate r]
        The system is linearized around psi = 0
        :return: lateral dynamics A, phi, B
        """
        A_lat, phi_lat, B_lat = self.lateral_lpv_structure()

        speed_body_x = self.speed
        A_top = np.array([
            [0, speed_body_x, 1, 0],
            [0, 0, 0, 1]
        ])
        A0 = np.concatenate((A_top, np.concatenate((np.zeros((2, 2)), A_lat), axis=1)))
        phi = np.array([np.concatenate((np.zeros((2, 4)), np.concatenate((np.zeros((2, 2)), phi_i), axis=1)))
                        for phi_i in phi_lat])
        B = np.concatenate((np.zeros((2, 1)), B_lat))
        return A0, phi, B

    def full_lateral_lpv_dynamics(self):
        """
        State: [position y, yaw psi, lateral speed v, yaw rate r]
        The system is linearized around psi = 0
        :return: lateral dynamics A, B
        """
        A0, phi, B = self.full_lateral_lpv_structure()
        self.theta = [self.FRICTION_FRONT, self.FRICTION_REAR]
        A = A0 + np.tensordot(self.theta, phi, axes=[0, 0])
        return A, B
    
    def on_state_update(self) -> None:
        return
        if self.road:
            self.lane_index = self.road.network.get_closest_lane_index(self.position, self.heading)
            self.lane = self.road.network.get_lane(self.lane_index)
            if self.road.record_history:
                self.history.appendleft(self.create_from(self))

    def predict_trajectory_constant_speed(self, times: np.ndarray) -> Tuple[List[np.ndarray], List[float]]:
        if self.prediction_type == 'zero_steering':
            action = {'acceleration': 0.0, 'steering': 0.0}
        elif self.prediction_type == 'constant_steering':
            action = {'acceleration': 0.0, 'steering': self.action['steering']}
        else:
            raise ValueError("Unknown predition type")

        dt = np.diff(np.concatenate(([0.0], times)))

        positions = []
        headings = []
        v = copy.deepcopy(self)
        v.act(action)
        for t in dt:
            v.step(t)
            positions.append(v.position.copy())
            headings.append(v.heading)
        return (positions, headings)

    @property
    def velocity(self) -> np.ndarray:
        return self.speed * self.direction  # TODO: slip angle beta should be used here

    @property
    def destination(self) -> np.ndarray:
        if getattr(self, "route", None):
            last_lane_index = self.route[-1]
            last_lane_index = last_lane_index if last_lane_index[-1] is not None else (*last_lane_index[:-1], 0)
            last_lane = self.road.network.get_lane(last_lane_index)
            return last_lane.position(last_lane.length, 0)
        else:
            return self.position

    @property
    def destination_direction(self) -> np.ndarray:
        if (self.destination != self.position).any():
            return (self.destination - self.position) / np.linalg.norm(self.destination - self.position)
        else:
            return np.zeros((2,))

    @property
    def lane_offset(self) -> np.ndarray:
        if self.lane is not None:
            long, lat = self.lane.local_coordinates(self.position)
            ang = self.lane.local_angle(self.heading, long)
            return np.array([long, lat, ang])
        else:
            return np.zeros((3,))

    def to_dict(self, origin_vehicle: "Vehicle" = None, observe_intentions: bool = True) -> dict:
        d = {
            'presence': 1,
            'x': self.position[0],
            'y': self.position[1],
            'vx': self.velocity[0],
            'vy': self.velocity[1],
            'heading': self.heading,
            'cos_h': self.direction[0],
            'sin_h': self.direction[1],
            'cos_d': self.destination_direction[0],
            'sin_d': self.destination_direction[1],
            'long_off': self.lane_offset[0],
            'lat_off': self.lane_offset[1],
            'ang_off': self.lane_offset[2],
        }
        if not observe_intentions:
            d["cos_d"] = d["sin_d"] = 0
        if origin_vehicle:
            origin_dict = origin_vehicle.to_dict()
            for key in ['x', 'y', 'vx', 'vy']:
                d[key] -= origin_dict[key]
        return d


def simulate(dt: float = 0.1) -> None:
    time = np.arange(0, 20, dt)
    vehicle = BicycleVehicle( position=[0, 5], speed=8.3)
    xx, uu = [], []
    A, B = vehicle.full_lateral_lpv_dynamics()
    K = -np.asarray(control.place(A, B, -np.arange(1, 5)))
    lpv = LPV(x0=vehicle.state[[1, 2, 4, 5]].squeeze(), a0=A, da=[np.zeros(A.shape)], b=B,
              d=[[0], [0], [0], [1]], omega_i=[[0], [0]], u=None, k=K, center=None, x_i=None)

    for t in time:
        # Act
        u = K @ vehicle.state[[1, 2, 4, 5]]
        omega = 2*np.pi/20
        u_p = 0*np.array([[-20*omega*np.sin(omega*t) * dt]])
        u += u_p
        # Record
        xx.append(np.array([vehicle.position[0], vehicle.position[1], vehicle.heading])[:, np.newaxis])
        uu.append(u)
        # Interval
        lpv.set_control(u, state=vehicle.state[[1, 2, 4, 5]])
        lpv.step(dt)
        # x_i_t = lpv.change_coordinates(lpv.x_i_t, back=True, interval=True)
        # Step
        vehicle.act({"acceleration": 0, "steering": u})
        vehicle.step(dt)

    xx, uu = np.array(xx), np.array(uu)
    plot(time, xx, uu)


def plot(time: np.ndarray, xx: np.ndarray, uu: np.ndarray) -> None:
    pos_x, pos_y = xx[:, 0, 0], xx[:, 1, 0]
    psi_x, psi_y = np.cos(xx[:, 2, 0]), np.sin(xx[:, 2, 0])
    dir_x, dir_y = np.cos(xx[:, 2, 0] + uu[:, 0, 0]), np.sin(xx[:, 2, 0] + uu[:, 0, 0])
    _, ax = plt.subplots(1, 1)
    ax.plot(pos_x, pos_y, linewidth=0.5)
    dir_scale = 1/5
    ax.quiver(pos_x[::20]-0.5/dir_scale*psi_x[::20],
              pos_y[::20]-0.5/dir_scale*psi_y[::20],
              psi_x[::20], psi_y[::20],
              angles='xy', scale_units='xy', scale=dir_scale, width=0.005, headwidth=1)
    ax.quiver(pos_x[::20]+0.5/dir_scale*psi_x[::20], pos_y[::20]+0.5/dir_scale*psi_y[::20], dir_x[::20], dir_y[::20],
              angles='xy', scale_units='xy', scale=0.25, width=0.005, color='r')
    ax.axis("equal")
    ax.grid()
    plt.show()
    plt.close()

simulate()