import matplotlib.pyplot as plt
import numpy as np
import math
import os
import pandas as pd
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D 
import time 
from os import listdir
import sys
sys.path.append('./scripts')
from CAN import  attractorNetworkSettling, attractorNetwork, attractorNetworkScaling, attractorNetwork2D
import CAN as can
import pykitti
import json 

'''Kitti Data'''
def data_processing_groundTruth():
    poses = pd.read_csv('./data/dataset/poses/00.txt', delimiter=' ', header=None)
    gt = np.zeros((len(poses), 3, 4))
    for i in range(len(poses)):
        gt[i] = np.array(poses.iloc[i]).reshape((3, 4))
    gt=gt[0::4]
    data_x=gt[:, :, 3][:,0][:100]
    data_y=gt[:, :, 3][:,2][:100]
    mag,rot=np.zeros(len(data_x)),np.zeros(len(data_x))
    for i in range(1,len(data_x)):  
        x1=data_x[i-1]
        x2=data_x[i]
        y1=data_y[i-1]
        y2=data_y[i]
        mag[i]=np.sqrt(((x2-x1)**2)+((y2-y1)**2))
        rot[i]=(math.atan2(y2-y1,x2-x1)) 
    return gt,mag,rot

def data_processing_oxts():
    path='./data/2011_09_26/2011_09_26_drive_0005_sync/oxts/data/'
    filenames = [f for f in listdir(path)]
    filenames.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    '''
    vn:    velocity towards north (m/s)
    ve:    velocity towards east (m/s)
    vf:    forward velocity, i.e. parallel to earth-surface (m/s)
    vl:    leftward velocity, i.e. parallel to earth-surface (m/s)
    vu:    upward velocity, i.e. perpendicular to earth-surface (m/s)'''
    vn,ve,vf,vl,vu=np.zeros(len(filenames)), np.zeros(len(filenames)), np.zeros(len(filenames)), np.zeros(len(filenames)), np.zeros(len(filenames))
    for i in range(len(filenames)):
        cur_file=pd.read_csv(path+filenames[i], delimiter=' ', header=None)
        vn[i]=cur_file[7]
        ve[i]=cur_file[8]
        vf[i]=cur_file[9]
        vl[i]=cur_file[10]
        vu[i]=cur_file[11]
    return vn,ve,vf,vl,vu

def pykitti_loading_data():
    kitti_root_dir = './data'
    kitti_date = '2011_09_26'
    kitti_drive = '0005'

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

'''Citiscape dataset'''
def saveOrLoadNp(filename,array,saveLoad_tag):
    if saveLoad_tag=='save':
        with open(filename, 'wb') as f:
            np.save(f, np.array(array))
    elif saveLoad_tag=='load':
        with open(filename, 'rb') as f:
            return np.load(f)

def storing_speeds_in_npyArray():
    path='./data/train_extra'
    citiscape_files= sorted([f for f in listdir(path)])
    with open(f'./data/train_extra/city_names', 'wb') as f:
            np.save(f, np.array(citiscape_files))


    for i in range(len(citiscape_files)):
        json_files=sorted([f for f in listdir(path+'/'+citiscape_files[i])])
        citiscape_speeds=[]
        for j in range(len(json_files)):
            full_path= open(path+'/'+citiscape_files[i]+'/'+json_files[j])
            data = json.load(full_path)["speed"]
            citiscape_speeds.append(data)
            print(data)
        with open(f'./data/train_extra/citiscape_speed_{i}', 'wb') as f:
            np.save(f, np.array(citiscape_speeds))

def plotSpeedProfileforEachCity():
    city_names=saveOrLoadNp(f'./data/train_extra/city_names',None,'load')
    fig, axs = plt.subplots(5,5)
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.92, top=0.92, hspace=0.5, wspace=0.4)
    axs=axs.flatten()
    axs[-1].axis('off'), axs[-2].axis('off')
    for dataset_num in range(23):
        speeds=saveOrLoadNp(f'./data/train_extra/citiscape_speed_{dataset_num}',None,'load')
        axs[dataset_num].plot(speeds)
        axs[dataset_num].set_title(city_names[dataset_num])
    plt.show()

def plotCombinedSpeedProfile():
    city_names=saveOrLoadNp(f'./data/train_extra/city_names',None,'load')
    all_cities_combined_speeds=np.array([])
    fig, axs = plt.subplots(1,1)
    plt.subplots_adjust(left=0.05, bottom=0.09, right=0.95, top=0.93, hspace=0.5, wspace=0.4)
    # for dataset_num in range(23):
    #     speeds=saveOrLoadNp(f'./data/train_extra/citiscape_speed_{dataset_num}',None,'load')
    #     all_cities_combined_speeds = np.concatenate([all_cities_combined_speeds, speeds])
        # saveOrLoadNp(f'./data/train_extra/speedProfiles_of_all_cities_combined',all_cities_combined_speeds,'save')
    all_cities_combined_speeds=saveOrLoadNp(f'./data/train_extra/speedProfiles_of_all_cities_combined',None,'load')
    axs.plot(all_cities_combined_speeds)
    plt.show()

'''Testing'''
# gt,mag,rot=data_processing_groundTruth()
# vn,ve,vf,vl,vu=data_processing_oxts()