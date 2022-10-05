import numpy as np
from os import listdir
import pandas as pd 
import geo_transform as geo



def latToScale(lat):
# % compute mercator scale from latitude
    return np.cos(lat * np.pi / 180.0)

def latlonToMercator(lat,lon,scale):
# % converts lat/lon coordinates to mercator coordinates using mercator scale
    er = 6378137
    mx = scale * lon * np.pi * er / 180
    my = scale * er * np.log( np.tan((90+lat) * np.pi / 360) )

    return [mx, my]

def convertOxtsToPose(oxts):
    # % converts a list of oxts measurements into metric poses,
    # % starting at (0,0,0) meters, OXTS coordinates are defined as
    # % x = forward, y = right, z = down (see OXTS RT3000 user manual)
    # % afterwards, pose{i} contains the transformation which takes a
    # % 3D point in the i'th frame and projects it into the oxts
    # % coordinates of the first frame.

    # % compute scale from first lat value
    scale = latToScale(oxts[0][0])

    # % init pose
    pose     = np.zeros((107,3))
    Tr_0_inv = []

    # % for all oxts packets do
    for i in range(1,len(oxts)):
    
        # % if there is no data => no pose
        if not [oxts[i]]:
            pose[i] = []
            pass

        # % translation vector
        t=np.zeros((3,1))
        t[0][0], t[1][0]= latlonToMercator(oxts[i][0],oxts[i][1],scale)
        t[2][0] = oxts[i][2]

        # % rotation matrix (OXTS RT3000 user manual, page 71/92)
        rx = oxts[i][3] #% roll
        ry = oxts[i][4] #% pitch
        rz = oxts[i][5] #% heading 
        Rx = np.asarray([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]]) #% base => nav  (level oxts => rotated oxts)
        Ry = np.asarray([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0,], [-np.sin(ry), 0, np.cos(ry)]]) #% base => nav  (level oxts => rotated oxts)
        Rz = np.asarray([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]]) #% base => nav  (level oxts => rotated oxts)
        R  = Rz@Ry@Rx

        joined=np.asarray([[R[0,0],R[0,1], R[0,2], t[0,0]],[R[1,0],R[1,1], R[1,2], t[1,0]],[R[2,0],R[2,1], R[2,2], t[2,0]],[0, 0, 0, 1]])
        # % normalize translation and rotation (start at 0/0/0)
        if len(Tr_0_inv)==0:
            Tr_0_inv = np.linalg.inv(joined)
        
        # % add pose
        pose[i]= Tr_0_inv*joined
    return pose 


# path='./data/2011_09_26_2/2011_09_26_drive_0001_sync/oxts/data/'
# filenames = [f for f in listdir(path)]
# filenames.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
# oxts=np.zeros((len(filenames),30))
# for i in range(len(filenames)):
#     oxts[i]=pd.read_csv(path+filenames[i], delimiter=' ', header=None)



import numpy as np
import matplotlib.pyplot as plt
import pykitti
import sys

kitti_root_dir = './data'
kitti_date = '2011_09_26'
kitti_drive = '0005'


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


fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.plot(x_imu, y_imu,'.')
ax.plot(xs, ys,'.')
ax.legend(['IMU data', 'GPS_data'])
ax.set_xlabel('X [m]')
ax.set_ylabel('Y [m]')
ax.grid()
plt.show()



# fig, ax = plt.subplots(1, 1, figsize=(8, 6))
# ax.plot(lons-lons[0], lats-lats[0])
# ax.set_xlabel('longitude [deg]')
# ax.set_ylabel('latitude [deg]')
# ax.grid()
# plt.show()