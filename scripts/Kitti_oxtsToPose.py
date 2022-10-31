import numpy as np
import geo_transform as geo
import extended_kalman_filter as ekf
import pykitti
import matplotlib.pyplot as plt
import sys
import CAN as can

# path='./data/2011_09_26_2/2011_09_26_drive_0001_sync/oxts/data/'
# filenames = [f for f in listdir(path)]
# filenames.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
# oxts=np.zeros((len(filenames),30))
# for i in range(len(filenames)):
#     oxts[i]=pd.read_csv(path+filenames[i], delimiter=' ', header=None)

kitti_root_dir = './data'
kitti_date = '2011_09_26'
kitti_drive = '0027'


'''GPS'''
dataset = pykitti.raw(kitti_root_dir, kitti_date, kitti_drive)
gt_trajectory_lla = []  # [longitude(deg), latitude(deg), altitude(meter)] x N
gt_yaws = []  # [yaw_angle(rad),] x N
gt_yaw_rates= []  # [vehicle_yaw_rate(rad/s),] x N
gt_forward_velocities = []  # [vehicle_forward_velocity(m/s),] x N
for oxts_data in dataset.oxts:
    packet = oxts_data.packet
    gt_trajectory_lla.append([
        packet.lon,
        packet.lat,
        packet.alt])
    gt_yaws.append(packet.yaw)
    gt_yaw_rates.append(packet.wz)
    gt_forward_velocities.append(packet.vf)

gt_trajectory_lla = np.array(gt_trajectory_lla).T
gt_trajectory_xyz = geo.lla_to_enu(gt_trajectory_lla, gt_trajectory_lla[:,0])
xs, ys, _ = gt_trajectory_xyz

'''IMU'''
timestamps = np.array(dataset.timestamps)
elapsed = np.array(timestamps) - timestamps[0]
ts = [t.total_seconds() for t in elapsed]

initial_yaw = gt_yaws[0] 

pose = np.array([
    gt_trajectory_xyz[0, 0],
    gt_trajectory_xyz[1, 0],
    initial_yaw])
poses=[]
poses.append(np.array([0,0,np.pi]))

for i in range(1,len(ts)):
    # propagate state x
    dt=ts[i]-ts[i-1]
    x, y, theta = pose
    v, omega = gt_forward_velocities[i],gt_yaw_rates[i]
    r = v / omega  # turning radius

    dtheta = omega * dt
    dx = - r * np.sin(theta) + r * np.sin(theta + dtheta)
    dy = + r * np.cos(theta) - r * np.cos(theta + dtheta)

    pose += np.array([dx, dy, dtheta])
    poses.append(list(pose))


x_imu=[p[0] for p in poses]
y_imu=[p[1] for p in poses]


# fig, ax = plt.subplots(1, 1, figsize=(8, 6))
# ax.plot(xs, ys,'.')
# ax.plot(x_imu, y_imu,'.')
# ax.legend(['GPS_data','IMU data'])
# ax.set_xlabel('X [m]')
# ax.set_ylabel('Y [m]')
# ax.grid()
# plt.show()


'''EKF'''
def EKF():
    #vairables 
    initial_yaw_std = np.pi
    yaw_rate_noise_std = 0.02 # standard deviation of yaw rate in rad/s
    xy_obs_noise_std = 7  # standard deviation of observation noise of x and y in meter
    forward_velocity_noise_std = 0.3 # standard deviation of forward velocity in m/s
    pose = np.array([
        gt_trajectory_xyz[0, 0],
        gt_trajectory_xyz[1, 0],
        initial_yaw])

    #initial state covarience 
    P = np.array([
        [xy_obs_noise_std ** 2., 0., 0.],
        [0., xy_obs_noise_std ** 2., 0.],
        [0., 0., initial_yaw_std ** 2.]
    ])

    #Measurement error 
    Q = np.array([
        [xy_obs_noise_std ** 2., 0.],
        [0., xy_obs_noise_std ** 2.]
    ])
    #state transition covarience 
    R = np.array([
        [forward_velocity_noise_std ** 2., 0., 0.],
        [0., forward_velocity_noise_std ** 2., 0.],
        [0., 0., yaw_rate_noise_std ** 2.]
    ])
    # initialize Kalman filter
    kf = ekf.ExtendedKalmanFilter(pose, P)

    # array to store estimated 2d pose [x, y, theta]
    mu_x = [pose[0],]
    mu_y = [pose[1],]
    mu_theta = [pose[2],]

    # array to store estimated error variance of 2d pose
    var_x = [P[0, 0],]
    var_y = [P[1, 1],]
    var_theta = [P[2, 2],]


    for t_idx in range(1, len(ts)):
        dt=ts[t_idx]-ts[t_idx-1]
        
        # get control input `u = [v, omega] + noise`
        u = np.array([gt_forward_velocities[t_idx],gt_yaw_rates[t_idx]])
        
        # because velocity and yaw rate are multiplied with `dt` in state transition function,
        # its noise covariance must be multiplied with `dt**2.`
        R_ = R * (dt ** 2.)
        
        # propagate!
        kf.propagate(u, dt, R)
        
        # get measurement `z = [x, y] + noise`
        z = np.array([
            gt_trajectory_xyz[0, t_idx],
            gt_trajectory_xyz[1, t_idx]
        ])
        
        # update!
        kf.update(z, Q)
        # save estimated state to analyze later
        # print(kf.x[0],kf.x[1])
        mu_x.append(kf.x[0])
        mu_y.append(kf.x[1])
        mu_theta.append(geo.normalize_angles(kf.x[2]))
        
        # save estimated variance to analyze later
        var_x.append(kf.P[0, 0])
        var_y.append(kf.P[1, 1])
        var_theta.append(kf.P[2, 2])
        


    mu_x = np.array(mu_x)
    mu_y = np.array(mu_y)
    mu_theta = np.array(mu_theta)

    var_x = np.array(var_x)
    var_y = np.array(var_y)
    var_theta = np.array(var_theta)

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    ax.plot(xs, ys, lw=2, label='gps trajectory')
    ax.plot(x_imu, y_imu, lw=0, marker='.', markersize=4, alpha=0.8, label='imu observed trajectory')
    ax.plot(mu_x, mu_y,lw=0, marker='.', markersize=4, alpha=0.8,label='EKF integrated trajectory', color='r')
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_title('Kitti GPS and IMU integrated with EKF')
    ax.legend()
    ax.grid()
    plt.show()

def normalise(x):
    return (x-min(x))/(max(x)-min(x))


def attractor_GPS_imu():
    N=200
    scale=[0.25,0.5,1,2,4]
    num_links,excite,activity_mag,inhibit_scale=1,3,0.1051745813,2.96673372e-02
    integratedPos1=[0]
    integratedPos2=[0]
    decodedPos=[0]

    prev_weights=[np.zeros(N), np.zeros(N), np.zeros(N),np.zeros(N), np.zeros(N)]
    net=can.attractorNetwork(N,num_links,excite, activity_mag,inhibit_scale)
    for n in range(len(prev_weights)):
        prev_weights[n][net.activation(N//2)]=net.full_weights(num_links)

    fig = plt.figure(figsize=(6, 6))
    fig_rows,fig_cols=6,1
    ax0 = plt.subplot2grid(shape=(fig_rows, fig_cols), loc=(0, 0), rowspan=5,colspan=1)
    # ax1 = plt.subplot2grid(shape=(fig_rows, fig_cols), loc=(0, 1), rowspan=5,colspan=1)
    # ax2 = plt.subplot2grid(shape=(fig_rows, fig_cols), loc=(0, 2), rowspan=5,colspan=1)
    # axtxt = plt.subplot2grid(shape=(fig_rows, fig_cols), loc=(5, 0), rowspan=1,colspan=1)
    # fig.tight_layout()

    # ax0.plot(np.arange(len(xs)),(xs),label='GPS observed x trajectory'), 
    ax0.set_title(f'IMU and Attractor Network Integrated Positions Drive {kitti_drive}')#, ax0.axis('equal')
    ax0.plot(np.arange(len(x_imu)),(x_imu),label='IMU Observed Trajectory')
    # axtxt.axis('off'), 
    # axtxt.text(0,0,f'Num_links: {num_links}, Excite_radius: {excite}, Activity_magnitude: {activity_mag}, Inhibition_scale: {inhibit_scale}', color='r',fontsize=12)

    for i in range(2,len(x_imu)):
        input1=xs[i]-xs[i-1]
        input2=x_imu[i]-x_imu[i-1]

        delta1 = [(input1/scale[0]), (input1/scale[1]), (input1/scale[2]), (input1/scale[3]), (input1/scale[4])]
        delta2 = [(input2/scale[0]), (input2/scale[1]), (input2/scale[2]), (input2/scale[3]), (input2/scale[4])]
        # print(delta1)
        split_output=np.zeros((len(delta1)))
        '''updating network'''    
        for n in range(len(delta1)):
            prev_weights[n][:]= net.update_weights_dynamics(prev_weights[n][:],delta1[n])
            prev_weights[n][prev_weights[n][:]<0]=0
            prev_weights[n][:]= net.update_weights_dynamics(prev_weights[n][:],delta2[n])
            prev_weights[n][prev_weights[n][:]<0]=0
            split_output[n]=can.activityDecoding(prev_weights[n][:],5,N)
        
        decoded_translation=np.sum((split_output-(N//2))*scale)

        integratedPos1.append(integratedPos1[-1]+input1)
        integratedPos2.append(integratedPos2[-1]+input2)
        decodedPos.append(decoded_translation)   

        print(f"{str(i)}  translation {input1} input output {round(integratedPos1[-1],3)} {round(integratedPos2[-1],3)} {str(decoded_translation )}  ")

    fitness=np.sum(abs(np.array(integratedPos1)-np.array(decodedPos)))*-1
    print(fitness), 
    # axtxt.text(0,-1,f'Error: {fitness}', fontsize=12, c='g')
    # ax1.plot(np.arange(len(integratedPos1)),integratedPos1,np.arange(len(integratedPos2)),integratedPos2), ax1.set_title('Integrated Position')#, ax1.axis('equal')
    ax0.plot((np.array(decodedPos)),label='Decoded Attractor Trajectory')#, ax2.axis('equal')
    ax0.legend()
    
    plt.show()

# EKF()
attractor_GPS_imu()