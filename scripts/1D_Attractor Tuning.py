import matplotlib.pyplot as plt
import time
import numpy as np
import math
import os
import pandas as pd 


from CAN import activityDecoding, activityDecodingAngle, attractorNetworkSettling, attractorNetwork, multiResolution,attractorNetworkScaling, imageHistogram
from matplotlib.colors import LinearSegmentedColormap
from multiprocessing import Process, Pool
from multiprocessing.dummy import freeze_support
import multiprocessing


'''Parameters'''
N=[200,200,360] #number of neurons
num_links=[4,4,17]
excite=[12,12,7]
activity_mag=[1,1,1]
inhibit_scale=[0.05,0.05,0.005]

# prev_weights_trans=[np.zeros(N[0]), np.zeros(N[0]), np.zeros(N[0]), np.zeros(N[0]),np.zeros(N[0])]
# prev_weights_rot=[np.zeros(N[1]), np.zeros(N[1]), np.zeros(N[1]),np.zeros(N[1]), np.zeros(N[1])]

def data_processing():
    poses = pd.read_csv('./data/dataset/poses/00.txt', delimiter=' ', header=None)
    gt = np.zeros((len(poses), 3, 4))
    for i in range(len(poses)):
        gt[i] = np.array(poses.iloc[i]).reshape((3, 4))
    return gt

def visualiseMultiple_1D_Networks():
    N=[30,30,30] #number of neurons
    prev_weights=[np.zeros(N[0]), np.zeros(N[1]), np.zeros(N[2])]
    '''Initlise Network Activation'''
    for n in range(len(prev_weights)):
        net=attractorNetworkScaling(N[n],num_links[0],excite[0], activity_mag[0],inhibit_scale[0])
        prev_weights[n][net.activation(N[0]//2)]=net.full_weights(num_links[0])

    '''ColourMap with transparent backgroung'''
    # get colormap
    ncolors = 256
    color_array = plt.get_cmap('viridis')(range(ncolors))
    # change alpha values
    color_array[:,-1] = np.linspace(0,1,ncolors)
    # create a colormap object
    map_object = LinearSegmentedColormap.from_list(name='rainbow_alpha',colors=color_array)
    # register this new colormap with matplotlib
    plt.register_cmap(cmap=map_object)

    '''Plotting'''
    fig = plt.figure(figsize=(11, 3))
    ax0 = fig.add_subplot(1, 3, 1)
    ax1 = fig.add_subplot(1, 3, 2)
    ax2 = fig.add_subplot(1, 3, 3, projection='3d')

    '''1D'''
    im1D=imageHistogram(prev_weights[0][:],N[0],1)
    ax0.imshow(im1D,aspect='auto',cmap='rainbow_alpha')
    ax0.invert_yaxis()
    ax0.set_ylim([-40,80])
    ax0.axis('off')
    # ax0.set_title('1D')
    ax0.tick_params(left=False,bottom = False, labelleft = False,labelbottom = False)


    '''2D'''
    im2D=np.outer(prev_weights[0][:],prev_weights[1][:])
    ax1.imshow(im2D, interpolation='nearest', aspect='auto',cmap='rainbow_alpha')
    # ax1.axis('off')
    # ax1.set_title('2D')
    ax1.tick_params(left=False,bottom = False, labelleft = False,labelbottom = False)

    '''3D'''
    im3D=np.zeros((N[0],N[1],N[2]))
    for j in range(N[2]):
        im3D[:,:,j]=prev_weights[2][j]*np.outer(prev_weights[0][:],prev_weights[1][:])
    xx, yy,zz = np.meshgrid(np.linspace(0,1,N[0]), np.linspace(0,1,N[1]), np.linspace(0,1,N[2]))
    ax2.scatter3D(xx, yy, zz, c=im3D, cmap='rainbow_alpha', marker='.')
    ax2.grid(False)
    # ax2.axis('off')
    # ax2.set_title('3D')
    ax2.tick_params(left=False,bottom = False, labelleft = False,labelbottom = False)
    '''removing plot ticks'''
    # for line in ax2.xaxis.get_ticklines():
    #     line.set_visible(False)
    # for line in ax2.yaxis.get_ticklines():
    #     line.set_visible(False)
    # for line in ax2.zaxis.get_ticklines():
    #     line.set_visible(False)

    '''plotting stacked images'''
    # xx, yy = np.meshgrid(np.linspace(0,1,N[0]), np.linspace(0,1,N[1]))
    # cset = ax2.contourf(xx, yy, im2D, 100, zdir='z', offset=0.5)
    # ax2.set_zlim((0.,1.))

    # fig.tight_layout()
    plt.show()


def encodingDecodingMotion(data_x,data_y,excite,inhibit_scale):
    '''Encoding change in translation and rotation (using scale factor to avoid wraparound and very small changes)
       Decoding Attractor Network activity and integrating change in x and y positions 
       Comapring expected and decoded translation, angular velocity and angle'''
    # global prev_weights, num_links, excite, activity_mag,inhibit_scale, curr_parameter
    prev_weights=[np.zeros(N[0]), np.zeros(N[1]), np.zeros(N[2])]
    num_links=[4,4,17]
    activity_mag=[1,1,1]
   

    '''Initalise network'''            
    delta=[0,0,90]
    for i in range(len(delta)):
        net=attractorNetwork(N[i],num_links[i],excite[i], activity_mag[i],inhibit_scale[i])
        prev_weights[i][net.activation(delta[i])]=net.full_weights(num_links[i])
    
    '''Data Storage Parameters'''
    x_gt,y_gt,rot_gt=np.zeros((len(data_x))), np.zeros((len(data_y))), np.zeros((len(data_x)))
    x,y,theta=np.zeros((len(data_x))), np.zeros(len(data_y)), np.zeros((len(data_x)))
    

    for i in range(len(data_x)):
        if i>=2:
            '''encoding mangnitude and direction of movement'''
            x0, x1, x2=data_x[i-2],data_x[i-1], data_x[i]
            y0, y1, y2=data_y[i-2], data_y[i-1], data_y[i]
            
            delta[0]=x2-x1
            delta[1]=y2-y1
            delta[2]=((np.rad2deg(math.atan2(y2-y1,x2-x1)) - np.rad2deg(math.atan2(y1-y0,x1-x0))))#%360     
           
            '''updating network'''
            net0=attractorNetwork(N[0],num_links[0],excite[0], activity_mag[0],inhibit_scale[0])
            net1=attractorNetwork(N[1],num_links[1],excite[1], activity_mag[1],inhibit_scale[1])
            net2=attractorNetwork(N[2],num_links[2],excite[2], activity_mag[2],inhibit_scale[2])

            prev_weights[0][:]= net0.update_weights_dynamics(prev_weights[0][:],delta[0])
            prev_weights[0][prev_weights[0][:]<0]=0
            
            prev_weights[1][:]= net1.update_weights_dynamics(prev_weights[1][:],delta[1])
            prev_weights[1][prev_weights[1][:]<0]=0

            prev_weights[2][:]= net2.update_weights_dynamics(prev_weights[2][:],delta[2])
            prev_weights[2][prev_weights[2][:]<0]=0

            '''decoding mangnitude and direction of movement'''
            x[i]=np.argmax(prev_weights[0][:])#-prev_trans
            y[i]=np.argmax(prev_weights[1][:])
            theta[i]=activityDecoding(prev_weights[2][:],num_links[2],N[2])#-prev_angle
            
            x_gt[i]=data_x[i]
            y_gt[i]=data_y[i]
            rot_gt[i]=np.rad2deg(math.atan2(y1-y0,x1-x0))

    x_error=np.sum(abs(x_gt-x))
    y_error=np.sum(abs(y_gt-y))
    theta_error=np.sum(abs(rot_gt-theta))

    return x_error, y_error, theta_error

def encodingDecodingMotionSearch(excite_trans,excite_rot,inhibit_trans,inhibit_rot):
    '''Encoding change in translation and rotation (using scale factor to avoid wraparound and very small changes)
       Decoding Attractor Network activity and integrating change in x and y positions 
       Comapring expected and decoded translation, angular velocity and angle'''
    # global prev_weights, num_links, excite, activity_mag,inhibit_scale, curr_parameter
    prev_weights=[np.zeros(N[0]), np.zeros(N[1]), np.zeros(N[2])]
    num_links=[4,4,17]
    activity_mag=[1,1,1]
    excite=[excite_trans,excite_trans,excite_rot]
    inhibit_scale=[inhibit_trans,inhibit_trans,inhibit_rot]
   

    '''Initalise network'''            
    delta=[0,0,90]
    for i in range(len(delta)):
        net=attractorNetwork(N[i],num_links[i],excite[i], activity_mag[i],inhibit_scale[i])
        prev_weights[i][net.activation(delta[i])]=net.full_weights(num_links[i])
    
    '''Data Storage Parameters'''
    x_gt,y_gt,rot_gt=np.zeros((len(data_x))), np.zeros((len(data_y))), np.zeros((len(data_x)))
    x,y,theta=np.zeros((len(data_x))), np.zeros(len(data_y)), np.zeros((len(data_x)))
    

    for i in range(len(data_x)):
        if i>=2:
            '''encoding mangnitude and direction of movement'''
            x0, x1, x2=data_x[i-2],data_x[i-1], data_x[i]
            y0, y1, y2=data_y[i-2], data_y[i-1], data_y[i]
            
            delta[0]=x2-x1
            delta[1]=y2-y1
            delta[2]=((np.rad2deg(math.atan2(y2-y1,x2-x1)) - np.rad2deg(math.atan2(y1-y0,x1-x0))))#%360     
           
            '''updating network'''
            net0=attractorNetwork(N[0],num_links[0],excite[0], activity_mag[0],inhibit_scale[0])
            net1=attractorNetwork(N[1],num_links[1],excite[1], activity_mag[1],inhibit_scale[1])
            net2=attractorNetwork(N[2],num_links[2],excite[2], activity_mag[2],inhibit_scale[2])

            prev_weights[0][:]= net0.update_weights_dynamics(prev_weights[0][:],delta[0])
            prev_weights[0][prev_weights[0][:]<0]=0
            
            prev_weights[1][:]= net1.update_weights_dynamics(prev_weights[1][:],delta[1])
            prev_weights[1][prev_weights[1][:]<0]=0

            prev_weights[2][:]= net2.update_weights_dynamics(prev_weights[2][:],delta[2])
            prev_weights[2][prev_weights[2][:]<0]=0

            '''decoding mangnitude and direction of movement'''
            x[i]=np.argmax(prev_weights[0][:])#-prev_trans
            y[i]=np.argmax(prev_weights[1][:])
            theta[i]=activityDecoding(prev_weights[2][:],num_links[2],N[2])#-prev_angle
            
            x_gt[i]=data_x[i]
            y_gt[i]=data_y[i]
            rot_gt[i]=np.rad2deg(math.atan2(y1-y0,x1-x0))

    x_error=np.sum(abs(x_gt-x))
    y_error=np.sum(abs(y_gt-y))
    theta_error=np.sum(abs(rot_gt-theta))

    return inhibit_rot,x_error, y_error, theta_error



def visualiseEncodingDecodingMotion(data_x,data_y,excite,inhibit_scale):
    '''Encoding change in translation and rotation (using scale factor to avoid wraparound and very small changes)
       Decoding Attractor Network activity and integrating change in x and y positions 
       Comapring expected and decoded translation, angular velocity and angle'''
    # global prev_weights, num_links, excite, activity_mag,inhibit_scale, curr_parameter
    prev_weights=[np.zeros(N[0]), np.zeros(N[1]), np.zeros(N[2])]
    num_links=[4,4,17]
    activity_mag=[1,1,1]

    '''Initalise network'''            
    delta=[0,0,90]
    for i in range(len(delta)):
        net=attractorNetwork(N[i],num_links[i],excite[i], activity_mag[i],inhibit_scale[i])
        prev_weights[i][net.activation(delta[i])]=net.full_weights(num_links[i])
    
    '''Data Storage Parameters'''
    x_gt,y_gt,rot_gt=np.zeros((len(data_x))), np.zeros((len(data_y))), np.zeros((len(data_x)))
    x,y,theta=np.zeros((len(data_x))), np.zeros(len(data_y)), np.zeros((len(data_x)))
    

    for i in range(len(data_x)):
        if i>=2:
            '''encoding mangnitude and direction of movement'''
            x0, x1, x2=data_x[i-2],data_x[i-1], data_x[i]
            y0, y1, y2=data_y[i-2], data_y[i-1], data_y[i]
            
            delta[0]=x2-x1
            delta[1]=y2-y1
            delta[2]=((np.rad2deg(math.atan2(y2-y1,x2-x1)) - np.rad2deg(math.atan2(y1-y0,x1-x0))))#%360     
           
            '''updating network'''
            net0=attractorNetwork(N[0],num_links[0],excite[0], activity_mag[0],inhibit_scale[0])
            net1=attractorNetwork(N[1],num_links[1],excite[1], activity_mag[1],inhibit_scale[1])
            net2=attractorNetwork(N[2],num_links[2],excite[2], activity_mag[2],inhibit_scale[2])

            prev_weights[0][:]= net0.update_weights_dynamics(prev_weights[0][:],delta[0])
            prev_weights[0][prev_weights[0][:]<0]=0
            
            prev_weights[1][:]= net1.update_weights_dynamics(prev_weights[1][:],delta[1])
            prev_weights[1][prev_weights[1][:]<0]=0

            prev_weights[2][:]= net2.update_weights_dynamics(prev_weights[2][:],delta[2])
            prev_weights[2][prev_weights[2][:]<0]=0

            '''decoding mangnitude and direction of movement'''
            x[i]=np.argmax(prev_weights[0][:])#-prev_trans
            y[i]=np.argmax(prev_weights[1][:])
            theta[i]=activityDecoding(prev_weights[2][:],num_links[2],N[2])#-prev_angle
            
            x_gt[i]=data_x[i]
            y_gt[i]=data_y[i]
            rot_gt[i]=np.rad2deg(math.atan2(y1-y0,x1-x0))

            # print(f"{str(i)} {x_gt[i]} {y_gt[i]}  {rot_gt[i]}_______  {delta[0]} {delta[1]}_______{x[i]} {y[i]} {theta[i]}")
            
            
    x_error=np.sum(abs(x_gt-x))
    y_error=np.sum(abs(y_gt-y))
    theta_error=np.sum(abs(rot_gt-theta))
    print(f"Final Error x: {x_error}")
    print(f"Final Error y: {y_error}")
    print(f"Final Error Rotate: {theta_error}")

    fig = plt.figure(figsize=(7, 7))
    ax0 = fig.add_subplot(4, 2, 1)
    ax1 = fig.add_subplot(4, 2, 2)
    ax2 = fig.add_subplot(4, 2, 3)
    ax3 = fig.add_subplot(4, 2, 4)
    ax4 = fig.add_subplot(4, 2, 5)
    ax5 = fig.add_subplot(4, 2, 6)
    ax6 = fig.add_subplot(4, 2, 7)
    ax7 = fig.add_subplot(4, 2, 8)

    ax0.set_title('Original')
    ax0.scatter(x_gt, y_gt,c='b',s=5)
    ax0.axis('equal')

    ax1.set_title('Converted')
    ax1.scatter(x, y,c='b',s=5)
    ax1.axis('equal')

    ax2.set_title('Rotation Input')
    ax2.plot(rot_gt,'-k.',markersize=1,linewidth=0.05)
    ax2.set_ylim([0,360])

    ax3.set_title('Rotation Output')
    ax3.plot(theta,'-k.',markersize=1,linewidth=0.05)
    ax3.set_ylim([0,360])

    ax4.set_title('X Input')
    ax4.plot(x_gt,'-m.',markersize=1,linewidth=0.05)

    ax5.set_title('X Output')
    ax5.plot(x,'-m.',markersize=1,linewidth=0.05)

    ax6.set_title('Y Input')
    ax6.plot(y_gt,'-g.',markersize=1,linewidth=0.05)

    ax7.set_title('Y Output')
    ax7.plot(y,'-g.',markersize=1,linewidth=0.05)
    
    fig.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    plt.gcf().text(0.02,0.02,"N= " + str(N[1]) +",  num links= " + str(num_links[1]) + ",  excite links= " + str(excite[1]) + ", inhibition=" + str(inhibit_scale[1]),  fontsize=8)
    plt.show()


def unitTest(data_x, data_y):
    excite=[6,6,12],[3,3,6], [2,2,3]
    inhibit_scale=[0.01,0.01,0.05], [0.001,0.001,0.001], [0.005,0.005,0.005]
    errors,params=[],[]
    for i in range(len(excite)):
        for j in range(len(inhibit_scale)):
            t0=time.time()
            x_error, y_error, theta_error=encodingDecodingMotion(data_x,data_y,excite[i],inhibit_scale[j])
            errors.append([x_error, y_error, theta_error])
            params.append([excite[i],inhibit_scale[j]])
            print(time.time()-t0)


    x_errors=[error[0] for error in errors]
    min_x=np.min(x_errors)       
    min_x_param=params[np.argmin(x_errors)]

    y_errors=[error[1] for error in errors]
    min_y=np.min(y_errors)       
    min_y_param=params[np.argmin(y_errors)]

    theta_errors=[error[2] for error in errors]
    min_theta=np.min(theta_errors)       
    min_theta_param=params[np.argmin(theta_errors)]

    # print(f"Minimum x error {min_x}, Minimum y error {min_y}, Minimum theta error {min_theta}")
    print(f"MinX Excite_Inhibit {min_x_param[0][0]}, {min_x_param[1][0]}, MinY Excite_Inhibit {min_x_param[0][1]}, {min_x_param[1][1]}, MinTheta Excite_Inhibit {min_x_param[0][2]} {min_x_param[1][2]}")
    
    visualiseEncodingDecodingMotion(data_x,data_y,[min_x_param[0][0],min_y_param[0][1],min_theta_param[0][2]],[min_x_param[1][0],min_y_param[1][1],min_theta_param[1][2]])

def gridSearch(data_x,data_y):
    x_error=np.zeros((5,5,5,5))
    y_error=np.zeros((5,5,5,5))
    theta_error=np.zeros((5,5,5,5))
    for excite_trans in list(np.linspace(1,N[0]/2, 5)):
        for excite_rot in list(np.linspace(1,N[2]/2, 5)):
            for inhibit_trans in list(np.linspace(0.005,0.1, 5)):
                #start 5 processes for values of inhibit rot 
                arg_values = [(round(excite_trans),round(excite_rot),inhibit_trans,inhibit_rot) for inhibit_rot in np.linspace(0.005,0.1, 5)]
                with Pool(processes=5) as pool:
                    res = pool.starmap(encodingDecodingMotionSearch, arg_values)


                inhibit_trans_index=list(np.linspace(0.005,0.1, 5)).index(inhibit_trans)
                excite_rot_index=list(np.linspace(1,N[2]/2, 5)).index(excite_rot)
                excite_trans_index=list(np.linspace(1,N[0]/2, 5)).index(excite_trans)
                for i in range(len(res)):
                    inhibit_rot_index=list(np.linspace(0.005,0.1, 5)).index(res[i][0])
                    x_error[excite_trans_index,excite_rot_index,inhibit_trans_index,inhibit_rot_index]=res[i][1]
                    y_error[excite_trans_index,excite_rot_index,inhibit_trans_index,inhibit_rot_index]=res[i][2]
                    theta_error[excite_trans_index,excite_rot_index,inhibit_trans_index,inhibit_rot_index]=res[i][3]
                    print(res[i])

                
    with open('./results/x_error.npy', 'wb') as f:
        np.save(f, np.array(x_error))
    with open('./results/y_error.npy', 'wb') as f:
        np.save(f, np.array(y_error))
    with open('./results/theta_error.npy', 'wb') as f:
        np.save(f, np.array(theta_error))

                
'''Testing'''
sparse_gt=data_processing()#[0::4]
data_x=sparse_gt[:, :, 3][:,0][:200]
data_y=sparse_gt[:, :, 3][:,2][:200]

# data_y=np.zeros(100)
# data_x=np.arange(100)

# visualiseMultiple1DNetworks()
# visualiseEncodingDecodingMotion(data_x,data_y,excite,inhibit_scale)
# unitTest(data_x, data_y)
if __name__=="__main__":
    freeze_support()
    gridSearch(data_x,data_y)

# with open('./results/test.npy', 'rb') as f:
#     a = np.load(f)
# print(a)