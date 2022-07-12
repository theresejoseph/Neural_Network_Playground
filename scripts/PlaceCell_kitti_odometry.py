from xml.sax.xmlreader import InputSource
import matplotlib.pyplot as plt
import numpy as np
import math
import os
import pandas as pd
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D 
from scipy import signal

from CAN import activityDecoding, activityDecodingAngle, attractorNetworkSettling, attractorNetwork, multiResolution


'''Parameters'''
N=[60,720] #number of neurons
curr_Neuron=[0,0]
# prev_weights=[np.zeros(N[0]), np.zeros(N[1])]
prev_weights=[np.zeros(N[0]), np.zeros(N[0]), np.zeros(N[0]), np.zeros(N[0]),np.zeros(N[0])]
num_links=[12,120]
excite=[10,77]
activity_mag=[1,1]
inhibit_scale=[0.01,0.0009]
curr_parameter=[0,0]
curr_x,curr_y=0,0
x,y=0,0
SCALING_FACTOR=50
SCALING_FACTOR2=360


def angdiff( th1, th2):
    d = th1 - th2
    d = np.mod(d+np.pi, 2*np.pi) - np.pi
    return d

def data_processing():
    poses = pd.read_csv('./data/dataset/poses/00.txt', delimiter=' ', header=None)
    gt = np.zeros((len(poses), 3, 4))
    for i in range(len(poses)):
        gt[i] = np.array(poses.iloc[i]).reshape((3, 4))
    return gt

def testing_Conversion(sparse_gt):
    fig = plt.figure(figsize=(13, 4))
    ax0 = fig.add_subplot(1, 2, 1)
    ax1 = fig.add_subplot(1, 2, 2)
    def animate(i):
        global curr_x, curr_y, x, y
        if i>0:
            x1=sparse_gt[:,:,3][i-1,0]
            y1=sparse_gt[:,:,3][i-1,2]

            x2=sparse_gt[:,:,3][i,0]
            y2=sparse_gt[:,:,3][i,2]

            delta1=np.sqrt(((x2-x1)**2)+((y2-y1)**2)) #translation
            if (x2-x1)==0:
                assert False 
                # delta2=np.pi/2
            else:
                delta2=(math.atan2(y2-y1,x2-x1)) #angle

            curr_x=curr_x + (delta1*np.cos(delta2))
            curr_y=curr_y + (delta1*np.sin(delta2))

            x=x+(x2-x1)
            y=y+(y2-y1)

            print(delta1, delta2)

            ax1.set_title('Converted')
            ax1.scatter(curr_x, curr_y,c='b',s=15)
            ax1.set_xlim([-300,300])
            ax1.set_ylim([-100,500])

            ax0.set_title('Original')
            ax0.scatter(x, y,c='b',s=15)
            ax0.set_xlim([-300,300])
            ax0.set_ylim([-100,500])

    ani = FuncAnimation(fig, animate, interval=1,frames=len(sparse_gt),repeat=False)
    plt.show()

def visualise(data_x,data_y):
    fig = plt.figure(figsize=(13, 4))
    ax0 = fig.add_subplot(1, 3, 1)
    ax1 = fig.add_subplot(1, 3, 2)
    ax2 = fig.add_subplot(1, 3, 3)

    '''Initalise network'''            
    delta=[0,0]
    for i in range(len(delta)):
        net=attractorNetwork(delta[i],N[i],num_links[i],excite[i], activity_mag[i],inhibit_scale[i])
        prev_weights[i][net.activation(delta[i])]=net.full_weights(num_links[i])
        prev_weights[i][prev_weights[i][:]<0]=0

    
    def animate(i):
        ax0.set_title("Ground Truth Pose")
        ax0.scatter(data_x[i],data_y[i],s=15)
        ax0.axis('equal')
        # ax0.set_xlim([-300,300])
        # ax0.set_ylim([-100,500])
        # ax0.set_zlim([-50,50])
        # ax0.invert_yaxis()
        # ax0.view_init(elev=39, azim=140)
        
        global prev_weights, num_links, excite, activity_mag,inhibit_scale, curr_parameter
        ax1.clear()
        if i>=2:
            '''encoding mangnitude and direction of movement'''
            x0=data_x[i-2]
            x1=data_x[i-1]
            x2=data_x[i]
            y0=data_y[i-2]
            y1=data_y[i-1]
            y2=data_y[i]
            
            delta[0]=np.sqrt(((x2-x1)**2)+((y2-y1)**2)) *SCALING_FACTOR#translation
            delta[1]=((np.rad2deg(math.atan2(y2-y1,x2-x1)) - np.rad2deg(math.atan2(y1-y0,x1-x0)))%360)*SCALING_FACTOR         #angle
       
            '''updating network'''
             
            net=attractorNetworkSettling(N[0],num_links[0],excite[0], activity_mag[0],inhibit_scale[0])
            prev_weights[0][:]= net.update_weights_dynamics(prev_weights[0][:],delta[0])
            prev_weights[0][prev_weights[0][:]<0]=0

            net=attractorNetworkSettling(N[1],num_links[1],excite[1], activity_mag[1],inhibit_scale[1])
            prev_weights[1][:]= net.update_weights_dynamics(prev_weights[1][:],delta[1])
            prev_weights[1][prev_weights[1][:]<0]=0

            '''decoding mangnitude and direction of movement'''
            trans=activityDecoding(prev_weights[0][:],num_links[0],N[0])/100#-prev_trans
            angle=np.deg2rad(activityDecodingAngle(prev_weights[1][:],num_links[1],N[1]))#-prev_angle

            curr_parameter[0]=curr_parameter[0] + (trans*np.cos(angle))
            curr_parameter[1]=curr_parameter[1]+ (trans*np.sin(angle))
            # curr_z=curr_z+del_z
            
            ax1.set_title("2D Attractor Network")
            im=np.outer(prev_weights[0][:],prev_weights[1][:])
            ax1.imshow(im,interpolation='nearest', aspect='auto')

            # print(delta1, delta2, del_y,del_x)
            ax2.set_title("Decoded Pose")
            ax2.scatter(curr_parameter[0], curr_parameter[1],s=15)
            ax2.axis('equal')
            # ax2.set_xlim([-100,100])
            # ax2.set_ylim([-100,100])
            # ax2.set_zlim([0,N])


            print(str(delta[0])+"__"+str( delta[1])+ "------"+str(trans )+"__"+str(np.rad2deg(angle)))
            # print(x2-x1, y2-y1)
            # print(len(signal.find_peaks(prev_weights_trans)[0]),len(signal.find_peaks(prev_weights_angle)[0]) )
            

    ani = FuncAnimation(fig, animate, interval=1,frames=len(data_x),repeat=False)
    plt.show()

def encodingDecodingMotion(data_x,data_y):
    global prev_weights, num_links, excite, activity_mag,inhibit_scale, curr_parameter

    '''Initalise network'''            
    delta=[0,360]
    for i in range(len(delta)):
        net=attractorNetwork(N[i],num_links[i],excite[i], activity_mag[i],inhibit_scale[i])
        prev_weights[i][net.activation(delta[i])]=net.full_weights(num_links[i])
   
    # [1,:]ev_weights_angle[net.activation(delta2)]=net.full_weights(num_links)
    # prev_weights_z[net.activation(int(delta3))]=net.full_weights(num_links)
    
    curr_x,curr_y=np.zeros((len(data_x))), np.zeros((len(data_y)))
    theta=np.zeros((len(data_x)))
    delta_ang_out,delta_ang=np.zeros((len(data_x))), np.zeros((len(data_x)))
    theta[0]=0
    theta[1]=90 #np.rad2deg(math.atan2(data_y[1]-data_y[0],data_x[1]- data_y[0]))
    tran,rot=np.zeros((len(data_x))), np.zeros((len(data_y)))
    rot[0]=0
    rot[1]=90#np.rad2deg(math.atan2(data_y[1]-data_y[0],data_x[1]- data_y[0]))
    tran_out,rot_out=np.zeros((len(data_x))), np.zeros((len(data_y)))
    for i in range(len(data_x)):
        if i>=2:
            '''encoding mangnitude and direction of movement'''
            x0=data_x[i-2]
            x1=data_x[i-1]
            x2=data_x[i]
            y0=data_y[i-2]
            y1=data_y[i-1]
            y2=data_y[i]
            
            delta[0]=np.sqrt(((x2-x1)**2)+((y2-y1)**2)) *SCALING_FACTOR#translation
            
            delta[1]=((np.rad2deg(math.atan2(y2-y1,x2-x1)) - np.rad2deg(math.atan2(y1-y0,x1-x0)))+SCALING_FACTOR2)#%360     #angle
           
            '''updating network'''
            net0=attractorNetworkSettling(N[0],num_links[0],excite[0], activity_mag[0],inhibit_scale[0])
            net1=attractorNetworkSettling(N[1],num_links[1],excite[1], activity_mag[1],inhibit_scale[1])

            for n in range(2):
                prev_weights[0][:]= net0.update_weights_dynamics(prev_weights[0][:],delta[0])
                prev_weights[0][prev_weights[0][:]<0]=0
                
                prev_weights[1][:]= net1.update_weights_dynamics(prev_weights[1][:],delta[1])
                prev_weights[1][prev_weights[1][:]<0]=0

            '''decoding mangnitude and direction of movement'''
            trans=activityDecoding(prev_weights[0][:],num_links[0],N[0])/SCALING_FACTOR#-prev_trans
            del_angle=(activityDecoding(prev_weights[1][:],num_links[1],N[1]))#-prev_angle
            theta[i]=(theta[i-1]+(del_angle-SCALING_FACTOR2))%360#(2*np.pi)
            curr_x[i]=curr_x[i-1]+ (trans*np.cos(np.deg2rad(theta[i-1])))
            curr_y[i]=curr_y[i-1]+ (trans*np.sin(np.deg2rad(theta[i-1])))
        

            tran[i]=delta[0]/SCALING_FACTOR
            delta_ang[i]=(delta[1]-SCALING_FACTOR2) #%360
            rot[i]=(rot[i-1]+(delta[1]-SCALING_FACTOR2)) % 360

            tran_out[i]=trans 
            delta_ang_out[i]=(del_angle-SCALING_FACTOR2) #% 360
            rot_out[i]=theta[i]
            # (delta[1]/SCALING_FACTOR)%360
            print(str(i)+ "  "+str(delta[0]/SCALING_FACTOR)+"  "+str(delta[1] )+ "_______"+str(trans )+"  "+str(del_angle))

    print(f"Final Error Translate: {np.sum(abs(tran-tran_out))}")
    print(f"Final Error Rotate: {np.sum(abs(delta_ang-delta_ang_out))}")
    fig = plt.figure(figsize=(7, 7))
    ax0 = fig.add_subplot(4, 2, 1)
    ax1 = fig.add_subplot(4, 2, 2)
    ax2 = fig.add_subplot(4, 2, 3)
    ax3 = fig.add_subplot(4, 2, 4)
    ax4 = fig.add_subplot(4, 2, 5)
    ax5 = fig.add_subplot(4, 2, 6)
    ax6 = fig.add_subplot(4, 2, 7)
    ax7 = fig.add_subplot(4, 2, 8)


    ax1.set_title('Converted')
    ax1.scatter(curr_x, curr_y,c='b',s=5)
    ax1.axis('equal')
    # ax1.set_xlim([-300,300])
    # ax1.set_ylim([-100,500])

    ax0.set_title('Original')
    ax0.scatter(data_x, data_y,c='b',s=5)
    ax0.axis('equal')
    # ax0.set_xlim([-300,300])
    # ax0.set_ylim([-100,500])

    ax2.set_title('Traslation Input')
    ax2.plot(tran,'-o',markersize=0.5,linewidth=0.05)
    # ax2.axis('equal')
    ax3.set_title('Traslation Output')
    ax3.plot(tran_out,'-o',markersize=0.5,linewidth=0.05)
    # ax4.axis('equal')

    ax4.set_title('Delta Rotation Input')
    ax4.plot(delta_ang,'g.',markersize=2)
    ax4.set_ylim([-360,360])
    # ax3.axis('equal')
    ax5.set_title('Delta Rotation Output')
    ax5.plot(delta_ang_out,'g.',markersize=2)
    ax5.set_ylim([-360,360])

    ax6.set_title('Rotation Input')
    ax6.plot(rot,'-k.',markersize=1,linewidth=0.05)
    ax6.set_ylim([0,360])
    # ax3.axis('equal')
    ax7.set_title('Rotation Output')
    ax7.plot(rot_out,'-k.',markersize=1,linewidth=0.05)
    ax7.set_ylim([0,360])

    # ax5.axis('equal')
    
    fig.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    plt.gcf().text(0.02,0.02,"N= " + str(N[1]) +",  num links= " + str(num_links[1]) + ",  excite links= " + str(excite[1]) + ", inhibition=" + str(inhibit_scale[1]),  fontsize=8)
    

    plt.show()

def multiResolutionUpdate(input): 
    delta, scale = multiResolution(abs(input))
    split_output=np.zeros((len(delta)))
    
    '''updating network'''    
    net=attractorNetworkSettling(N[0],num_links[0],excite[0], activity_mag[0],inhibit_scale[0])
    for k in range(5):
        for n in range(len(delta)):
            prev_weights[n][:]= net.update_weights_dynamics(prev_weights[n][:],delta[n])
            prev_weights[n][prev_weights[n][:]<0]=0
            split_output[n]=np.argmax(prev_weights[n][:])#-prev_trans
    '''decoding mangnitude and direction of movement'''
    decoded=np.sum(split_output*scale)*np.sign(input)
    
    return decoded        

def multiResolutionTranslation(data_x,data_y):
    global prev_weights, num_links, excite, activity_mag,inhibit_scale, curr_parameter

    input,decoded_output=np.zeros((len(data_x))), np.zeros((len(data_x)))
    for i in range(len(data_x)):
        if i>1:
            '''encoding mangnitude and direction of movement'''
            x0=data_x[i-2]
            x1=data_x[i-1]
            x2=data_x[i]
            y0=data_y[i-2]
            y1=data_y[i-1]
            y2=data_y[i]
            
            rotation=((np.rad2deg(math.atan2(y2-y1,x2-x1)) - np.rad2deg(math.atan2(y1-y0,x1-x0))))#%360     #angle
            input[i]=np.sqrt(((x2-x1)**2)+((y2-y1)**2))#translation
            decoded_output[i]=multiResolutionUpdate(input[i])
            
            print(f"{str(i)}   {str(input[i] )}   {str(decoded_output[i] )}")

    print(f"Final Error: {np.sum(abs(input-decoded_output))}")
    fig = plt.figure(figsize=(7, 7))
    ax2 = fig.add_subplot(1, 2, 1)
    ax3 = fig.add_subplot(1, 2, 2)

    ax2.set_title('Traslation Input')
    ax2.plot(input,'-o',markersize=0.5,linewidth=0.05)
    # ax2.set_ylim([-10,10])
    # ax2.axis('equal')
    ax3.set_title('Traslation Output')
    ax3.plot(decoded_output,'-o',markersize=0.5,linewidth=0.05)
    # ax3.set_ylim([-10,10])
    # ax4.axis('equal')
    
    # fig.tight_layout()
    # plt.subplots_adjust(bottom=0.1)
    # plt.gcf().text(0.02,0.02,"N= " + str(N[1]) +",  num links= " + str(num_links[1]) + ",  excite links= " + str(excite[1]) + ", inhibition=" + str(inhibit_scale[1]),  fontsize=8)

    plt.show()

def visualiseMultiResolution(data_x,data_y):
    fig = plt.figure(figsize=(13, 4))
    ax0 =  plt.subplot2grid(shape=(5, 3), loc=(0, 0), rowspan=5,colspan=1)
    ax10 = plt.subplot2grid(shape=(5, 3), loc=(0, 1), rowspan=1,colspan=1)
    ax11 = plt.subplot2grid(shape=(5, 3), loc=(1, 1), rowspan=1,colspan=1)
    ax12 = plt.subplot2grid(shape=(5, 3), loc=(2, 1), rowspan=1,colspan=1)
    ax13 = plt.subplot2grid(shape=(5, 3), loc=(3, 1), rowspan=1,colspan=1)
    ax14 = plt.subplot2grid(shape=(5, 3), loc=(4, 1), rowspan=1,colspan=1)
    ax2 = plt.subplot2grid(shape=(5, 3), loc=(0, 2), rowspan=5,colspan=1)
    fig.tight_layout()


    def animate(i):
        global prev_weights, num_links, excite, activity_mag,inhibit_scale, curr_parameter
        ax10.clear(),ax11.clear(),ax12.clear(),ax13.clear(),ax14.clear()
        if i>=2:
            '''encoding mangnitude and direction of movement'''
            x0=data_x[i-2]
            x1=data_x[i-1]
            x2=data_x[i]
            y0=data_y[i-2]
            y1=data_y[i-1]
            y2=data_y[i]
            
            input=np.sqrt(((data_x[i]-data_x[i-1])**2)+((data_y[i]-data_y[i-1])**2))
            rotation=((np.rad2deg(math.atan2(y2-y1,x2-x1)) - np.rad2deg(math.atan2(y1-y0,x1-x0))))#%360 
            decoded_output=multiResolutionUpdate(input)
            
            print(f"{str(i)}   {str(input)}   {str(decoded_output )}")
            '''decoding mangnitude and direction of movement'''
            
            ax0.plot(i,input,"k.")
            ax0.set_title("Ground Truth Velocity")
            # ax0.axis('equal')   

            ax10.set_title("Tens Place",fontsize=6)
            ax10.bar(np.arange(N[0]),prev_weights[0][:],color='aqua')
            ax10.axis('off')

            ax11.set_title("Ones Place",fontsize=6)
            ax11.bar(np.arange(N[0]),prev_weights[1][:],color='green')
            ax11.axis('off')

            ax12.set_title("Tenth Place",fontsize=6)
            ax12.bar(np.arange(N[0]),prev_weights[2][:],color='blue')
            ax12.axis('off')

            ax13.set_title("Hundreths Place",fontsize=6)
            ax13.bar(np.arange(N[0]),prev_weights[3][:],color='purple')
            ax13.axis('off')

            ax14.set_title("Thousandths Place",fontsize=6)
            ax14.bar(np.arange(N[0]),prev_weights[4][:],color='pink')
            ax14.axis('off')

            # print(delta1, delta2, del_y,del_x)
            ax2.set_title("Decoded Velocity")
            ax2.plot(i,decoded_output,'r.')

            fig.tight_layout()

            

    ani = FuncAnimation(fig, animate, interval=1,frames=len(data_x),repeat=False)
    plt.show()



'''Test Area'''
# sparse_gt=data_processing()#[0::2]
# data_x=sparse_gt[:, :, 3][:,0][:100]
# data_y=sparse_gt[:, :, 3][:,2][:100]

# data_y=np.concatenate([np.zeros(100), np.arange(100), np.ones(100)*100, np.arange(100,5,-1)])
# data_x=np.concatenate([np.arange(100), np.ones(100)*100, np.arange(100,0,-1), np.zeros(95)])

# data_x=np.arange(200)
# data_y=np.zeros(200)

# data_y=np.zeros(100)
# data_x=np.arange(100)

data_y=np.zeros(50)
data_x=np.concatenate([np.linspace(0,0.1,10), np.linspace(0.2,1,10), np.linspace(2,10,10), np.linspace(20,100,10),np.linspace(110,1000,10)])


# visualise(data_x,data_y)
# encodingDecodingMotion(data_x,data_y)
# multiResolutionTranslation(data_x,data_y)
visualiseMultiResolution(data_x,data_y)

# print(np.rad2deg(math.atan2(0,-1)))

# testing_Conversion(sparse_gt)

# testing_Conversion(sparse_gt)


# print(np.shape(np.transpose(prev_weights[0][:])))

# print(np.shape(prev_weights[1][:]))

# print(np.outer(prev_weights[0][:],prev_weights[1][:]))