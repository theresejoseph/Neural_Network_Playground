import pandas 
import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.animation import FuncAnimation
from scipy.spatial.distance import cdist
import cv2
import math 
from os import listdir
from PIL import Image 
from CAN import activityDecoding, activityDecodingAngle, attractorNetworkSettling, attractorNetwork

SCALE=1/10
RADIUS=20
MIN_SIM_SCORE=100000
RESIZE_DIM=(72,128)
N=[300,360] #number of neurons
curr_Neuron=[0,0]
prev_weights=[np.zeros(N[0]), np.zeros(N[1])]
num_links=[50,60]
excite=[30,47]
activity_mag=[1,1]
inhibit_scale=[0.005,0.005]
curr_parameter=[0,0]
curr_x,curr_y=0,0
x,y=0,0


def loadImages(path):
    imagesList = sorted(listdir(path))
    loadedImages = []
    for image in imagesList:
        img = np.array(Image.open(path + image))
        imgSmall=cv2.resize(img, RESIZE_DIM, interpolation=cv2.INTER_AREA) #resize
        loadedImages.append(cv2.cvtColor(imgSmall, cv2.COLOR_BGR2GRAY)) #grayscale
    return loadedImages

def distanceMatrix():
    feats_ref=np.zeros((len(imgs00),RESIZE_DIM[0]*RESIZE_DIM[1]))
    feats_qry=np.zeros((len(imgs00),RESIZE_DIM[0]*RESIZE_DIM[1]))
    for i in range(len(imgs00)):
        feats_ref[i,:]=(imgs00[i].flatten())
        feats_qry[i,:]=(imgs01[i].flatten())
    dMat = cdist(feats_ref,feats_qry,'euclidean')
    plt.imshow(dMat)
    plt.show()

def alreadyVisited(library,flatImage):
    bool=False
    distance=np.zeros(len(library))
    for i in range(len(library)):
        distance[i]=np.sum(abs(library[i][0]-flatImage))
    # print(distance[np.argmin(distance)])
    smallestdist=distance[np.argmin(distance)]
    if smallestdist<MIN_SIM_SCORE:
        bool= True

    return bool, np.argmin(distance)

def CAN_VPR_Landmarks():
    '''Visualise Landmarks and Position'''
    fig = plt.figure(figsize=(15, 5))
    ax0= plt.subplot2grid(shape=(2, 3), loc=(0, 0), rowspan=1,colspan=1)
    ax1= plt.subplot2grid(shape=(2, 3), loc=(1, 0), rowspan=1,colspan=1)
    ax11= plt.subplot2grid(shape=(2, 3), loc=(0, 1), rowspan=1,colspan=1)
    ax12= plt.subplot2grid(shape=(2, 3), loc=(1, 1), rowspan=1,colspan=1)
    ax3= plt.subplot2grid(shape=(2, 3), loc=(1, 2), rowspan=1,colspan=1)
    ax2= plt.subplot2grid(shape=(2, 3), loc=(0, 2), rowspan=1,colspan=1)
    imgLib=[]
    delta=[0,0]
    '''Initalise network'''            
    for i in range(len(delta)):
        net=attractorNetwork(N[i],num_links[i],excite[i], activity_mag[i],inhibit_scale[i])
        prev_weights[i][net.activation(delta[i])]=net.full_weights(num_links[i])
    imgLib.append([TwoTraversesImgs[0].flatten(),prev_weights[0][:], prev_weights[1][:]])

    # imgLib.append([TwoTraversesImgs[1].flatten(),lndmrks[1][0], lndmrks[1][1]])
    def animate(i):
        ax2.clear(), ax12.clear(), ax0.clear()#plt.gcf().clear()
        
        '''encoding mangnitude and direction of movement'''
        if i>0 : 
            # x0=data_x[i-2]
            x1=lndmrks[i-1][0]
            x2=lndmrks[i][0]
            
            # y0=data_y[i-2]
            y1=lndmrks[i-1][1]
            y2=lndmrks[i][1]
            
            delta[0]=np.sqrt(((x2-x1)**2)+((y2-y1)**2)) *100           #translation
            delta[1]=np.rad2deg(math.atan2(y2-y1,x2-x1)) % 360          #angle
            
            net1=attractorNetworkSettling(N[0],num_links[0],excite[0], activity_mag[0],inhibit_scale[0])
            net2=attractorNetworkSettling(N[1],num_links[1],excite[1], activity_mag[1],inhibit_scale[1])
            print(i % 5)
            if (i % 5) == 0:
                ax0.clear()
                ax0.imshow(TwoTraversesImgs[i], interpolation='nearest', aspect='auto')
                ax0.set_title('Ground Truth')
                bool,idx=alreadyVisited(imgLib,TwoTraversesImgs[i].flatten())
            
                if bool==True:
                    #add landmark activity to attractor network
                    # ax2.text(0,0,'Image Matched',c='r')
                    # ax2.axis('off')
                    ax2.set_title("Matched Image")
                    ax2.imshow(np.reshape(imgLib[idx][0],(RESIZE_DIM[1],RESIZE_DIM[0])), interpolation='nearest', aspect='auto')

                    prev_weights[0][:]+=imgLib[idx][1]
                    prev_weights[0][:]=prev_weights[0][:]/np.linalg.norm(prev_weights[0][:])
        

                    prev_weights[1][:]+=imgLib[idx][2]
                    prev_weights[1][:]=prev_weights[1][:]/np.linalg.norm(prev_weights[1][:])

                    ax12.clear()
                    ax12.set_title("Stored Landmark Activity")
                    im=np.outer(prev_weights[0][:],prev_weights[1][:])
                    ax12.imshow(im,interpolation='nearest', aspect='auto')

                else: 
                    imgLib.append([TwoTraversesImgs[i].flatten(),prev_weights[0][:], prev_weights[1][:]])
                    ax2.text(0,0,'Image Added',c='r')
                    ax2.axis('off')

            prev_weights[0][:]= net1.update_weights_dynamics(prev_weights[0][:],delta[0])
            prev_weights[0][prev_weights[0][:]<0]=0

            prev_weights[1][:]= net2.update_weights_dynamics(prev_weights[1][:],delta[1])
            prev_weights[1][prev_weights[1][:]<0]=0

            '''decoding mangnitude and direction of movement'''
            trans=activityDecoding(prev_weights[0][:],num_links[0],N[0])/100#-prev_trans
            angle=np.deg2rad(activityDecodingAngle(prev_weights[1][:],num_links[1],N[1]))#-prev_angle

            curr_parameter[0]=curr_parameter[0] + (trans*np.cos(angle))
            curr_parameter[1]=curr_parameter[1]+ (trans*np.sin(angle))
            ax11.text(0,0,f"True:{round(delta[0]/100,2)},   {round(delta[1],2)} _____ Decoded:{round(trans,2)},   {round(np.rad2deg(angle),2)}", c='b')
        

        ax1.scatter(lndmrks[i][0], lndmrks[i][1], marker=".") #landmarks 
        ax1.axis('equal')

        ax11.clear()
        ax11.set_title("Attractor Network")
        im=np.outer(prev_weights[0][:],prev_weights[1][:])
        ax11.imshow(im,interpolation='nearest', aspect='auto')
        
        

        ax3.set_title("Decoded Pose")
        ax3.scatter(curr_parameter[0], curr_parameter[1],s=15)
        ax3.axis('equal')
        
    ani = FuncAnimation(fig, animate, interval=1, frames= len(TwoTraversesImgs),repeat=False)
    plt.show() 

'''Load and store images'''

path00 = "./data/day_right/"
imgs00 = loadImages(path00)[:50]
path01 = "./data/day_left/"
imgs01 = loadImages(path01)[:50]

TwoTraversesImgs=[]
for i in range(len(imgs00)):
    TwoTraversesImgs.append(imgs00[i])
for i in range(len(imgs00)):
    TwoTraversesImgs.append(imgs00[i])

'''Landmark Pseduo Positions'''
mark_x=[]
mark_y=[]
lndmrk_angles=[]

for phi in np.linspace(0,360,len(imgs00)):
    mark_x.append(RADIUS * math.cos(np.deg2rad(phi)) )
    mark_y.append(RADIUS * math.sin(np.deg2rad(phi)))
    lndmrk_angles.append(phi)
for phi in np.linspace(0,360,len(imgs00)):
    mark_x.append((RADIUS) * math.cos(np.deg2rad(phi)) )
    mark_y.append((RADIUS) * math.sin(np.deg2rad(phi)))
    lndmrk_angles.append(phi)
lndmrks=np.stack((np.array(mark_x),np.array(mark_y)),axis=1)
print(len(lndmrks))



CAN_VPR_Landmarks()