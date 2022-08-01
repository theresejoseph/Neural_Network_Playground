import matplotlib.pyplot as plt
import numpy as np
import math
import os
import pandas as pd
from matplotlib.animation import FuncAnimation
from matplotlib.artist import Artist
from mpl_toolkits.mplot3d import Axes3D 
from scipy import signal

from CAN import activityDecoding, activityDecodingAngle, attractorNetworkSettling, attractorNetwork, multiResolution,attractorNetworkScaling

'''Parameters'''
N=[300, 60] #number of neurons
curr_Neuron=[0,0]
prev_weights=[np.zeros(N[0]), np.zeros(N[1])]
split_output=[0,0,0]
num_links=[3,17]
excite=[1,7]
activity_mag=[1,1]
inhibit_scale=[0.05,0.005]
curr_parameter=[0,0]
crossovers=np.zeros(5)

def data_processing():
    poses = pd.read_csv('./data/dataset/poses/00.txt', delimiter=' ', header=None)
    gt = np.zeros((len(poses), 3, 4))
    for i in range(len(poses)):
        gt[i] = np.array(poses.iloc[i]).reshape((3, 4))
    return gt

'''Initiliase Translating Weights'''
prev_weights_trans=[np.zeros(N[0]), np.zeros(N[0]), np.zeros(N[0]),np.zeros(N[0]), np.zeros(N[0])]
for n in range(len(prev_weights_trans)):
    net=attractorNetworkScaling(N[0],num_links[0],excite[0], activity_mag[0],inhibit_scale[0])
    prev_weights_trans[n][net.activation(0)]=net.full_weights(num_links[0])

'''Initiliase Rotating Weights'''
prev_weights_rot=[np.zeros(N[1]), np.zeros(N[1]), np.zeros(N[1]),np.zeros(N[1]), np.zeros(N[1])]
net=attractorNetworkScaling(N[1],num_links[0],excite[0], activity_mag[0],inhibit_scale[0])
for n in range(len(prev_weights_rot)):
    prev_weights_rot[n][net.activation(0)]=net.full_weights(num_links[0])

#filtering method 
def visualiseMultiResolutionFeedthroughTranslation(data_x,data_y):
    # global prev_weights, num_links, excite, activity_mag,inhibit_scale, curr_parameter
    fig = plt.figure(figsize=(13, 6))
    ax0 =  plt.subplot2grid(shape=(6, 3), loc=(0, 0), rowspan=5,colspan=1)
    ax2 =  plt.subplot2grid(shape=(6, 3), loc=(0, 2), rowspan=5,colspan=1)
    ax10 = plt.subplot2grid(shape=(6, 3), loc=(0, 1), rowspan=1,colspan=1)
    ax11 = plt.subplot2grid(shape=(6, 3), loc=(1, 1), rowspan=1,colspan=1)
    ax12 = plt.subplot2grid(shape=(6, 3), loc=(2, 1), rowspan=1,colspan=1)
    ax13 = plt.subplot2grid(shape=(6, 3), loc=(3, 1), rowspan=1,colspan=1)
    ax14 = plt.subplot2grid(shape=(6, 3), loc=(4, 1), rowspan=1,colspan=1)
    axtxt = plt.subplot2grid(shape=(6, 3), loc=(5, 0), rowspan=1,colspan=1)
    axtxt1 = plt.subplot2grid(shape=(6, 3), loc=(5, 1), rowspan=1,colspan=1)

    fig.tight_layout()
    curr_x,curr_y=np.zeros((len(data_x))), np.zeros((len(data_y)))
    theta=np.zeros((len(data_x)))
    decoded_translations=np.zeros((len(data_x)))
    theta[1]=90

    def multiResolutionUpdate(input,prev_weights,net):
        '''Uptate the activity (previous weights) of scaled networks based on input velocity'''
        #initiliaise network and scale
        scale = [(1/(N[0]**2)), (1/N[0]), 1, N[0], N[0]**2]
        split_output=np.zeros((len(scale)))
        net=attractorNetwork(N[0],num_links[0],excite[0], activity_mag[0],inhibit_scale[0])

        #updating network
        prev_weights[0][:],crossovers[0]= net.update_weights_dynamics(prev_weights[0][:],(abs(input)/scale[0]),cross=True)    
        prev_weights[0][prev_weights[0][:]<0]=0
        split_output[0]=np.argmax(prev_weights[0][:])

        for n in range(1,len(scale)):
            prev_weights[n][:],crossovers[n]= net.update_weights_dynamics(prev_weights[n][:],crossovers[n-1],cross=True)
            prev_weights[n][prev_weights[n][:]<0]=0
            split_output[n]=np.argmax(prev_weights[n][:])

        #decoding network and plotting values
        decoded=np.sum(split_output*scale)*np.sign(input)
        return decoded,split_output

    def animate(i):
        global prev_weights_trans,prev_weights_rot, num_links, excite, activity_mag,inhibit_scale
        ax10.clear(),ax11.clear(),ax12.clear(),ax13.clear(),ax14.clear(), axtxt.clear(),axtxt1.clear()

        if i>=2:
            '''encoding mangnitude and direction of movement'''
            x0=data_x[i-2]
            x1=data_x[i-1]
            x2=data_x[i]
            y0=data_y[i-2]
            y1=data_y[i-1]
            y2=data_y[i]
            
            translation=np.sqrt(((x2-x1)**2)+((y2-y1)**2))#translation
            rotation=((np.rad2deg(math.atan2(y2-y1,x2-x1)) - np.rad2deg(math.atan2(y1-y0,x1-x0))))#%360     #angle

            net1=attractorNetwork(N[0],num_links[0],excite[0], activity_mag[0],inhibit_scale[0])
            decoded_translation,split_trans=multiResolutionUpdate(translation,prev_weights_trans,net1)

            net2=attractorNetwork(N[1],num_links[1],excite[1], activity_mag[1],inhibit_scale[1])
            decoded_rotation,split_rot=multiResolutionUpdate(rotation,prev_weights_rot,net2)
    
            # curr_parameter[0]=curr_parameter[0]+decoded_translation

            "Adding translation and rotation to previous posisiton"
            theta[i]=(theta[i-1]+(rotation))%360
            # decoded_translations[i]=decoded_translation-decoded_translations[i-1]
            curr_x[i]=curr_x[i-1]+ (decoded_translations[i]*np.cos(np.deg2rad(theta[i])))
            curr_y[i]=curr_y[i-1]+ (decoded_translations[i]*np.sin(np.deg2rad(theta[i])))
              
            # print(f"{str(i)}  velocity {str(x2)}  {str(decoded_translation )}  ")
            # print(f"{str(i)}  position {str(round(x2))}  decoded {str(round(decoded_translation ))}  ")
           
            axtxt.axis('off'),axtxt1.axis('off')
            axtxt.text(0,0.2,"Input Trans: " + str(round(x2,3)) + " , " + str(round(decoded_translation,3)), c='r')
            # axtxt.text(0,0,"Input Rot: " +str(round(rotation,3))+ " " + str(round(decoded_rotation,3)), c='m')
            axtxt1.text(0,0,"Decoded Position of Each Network: " + str(split_trans), c='r')
            
            ax0.set_title("Ground Truth Translation")
            ax0.plot(x2,0,"g.")
            ax0.axis('equal')

            ax2.plot(decoded_translation,0,'k.')
            ax2.axis('equal')

            ax10.set_title(str(1/(N[0]**2))+" Scale",fontsize=6)
            ax10.bar(np.arange(N[0]),prev_weights_trans[0][:],color='aqua')
            ax10.axis('off')

            ax11.set_title(str(1/N[0])+" Scale",fontsize=6)
            ax11.bar(np.arange(N[0]),prev_weights_trans[1][:],color='green')
            ax11.axis('off')

            ax12.set_title(str(1) + " Scale",fontsize=6)
            ax12.bar(np.arange(N[0]),prev_weights_trans[2][:],color='blue')
            ax12.axis('off')

            ax13.set_title(str(N[0]) + " Scale",fontsize=6)
            ax13.bar(np.arange(N[0]),prev_weights_trans[3][:],color='purple')
            ax13.axis('off')

            ax14.set_title(str(N[0]**2)+" Scale",fontsize=6)
            ax14.bar(np.arange(N[0]),prev_weights_trans[4][:],color='pink')
            ax14.axis('off')    

    ani = FuncAnimation(fig, animate, interval=1,frames=len(data_x),repeat=False)
    plt.show()


# whole input into all networks 
# currently working on modifying network dynamics so new activity outsude the current activity bump dies out
def multiResolutionTranslation(data_x,data_y):
    global prev_weights, num_links, excite, activity_mag,inhibit_scale, curr_parameter
    
    # Initlise netowrk and paramter storage 
    net=attractorNetworkScaling(N[0],num_links[0],excite[0], activity_mag[0],inhibit_scale[0])
    prev_weights[0][net.activation(0)]=net.full_weights(num_links[0])
    for n in range(len(prev_weights_trans)):
        prev_weights_trans[n][net.activation(0)]=net.full_weights(num_links[0])

    input,decoded_output=np.zeros((len(data_x))), np.zeros((len(data_x)))
    
    def multiResolutionUpdate(input,prev_weights,net): 
        scale = [0.01, 0.1, 1, 10, 100]
        delta = [(input/scale[0]), (input/scale[1]), (input/scale[2]), (input/scale[3]), (input/scale[4])]
        split_output=np.zeros((len(delta)))
        
        '''updating network'''    
        for k in range(5):
            for n in range(len(delta)):
                prev_weights[n][:]= net.update_weights_dynamics(prev_weights[n][:],delta[n])
                prev_weights[n][prev_weights[n][:]<0]=0
                split_output[n]=np.argmax(prev_weights[n][:])

        '''decoding mangnitude and direction of movement'''
        print(split_output)
        decoded=np.sum(split_output*scale)*np.sign(input)        
        return decoded  

    #Update Network and Store parameters
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
            net0=attractorNetworkScaling(N[0],num_links[0],excite[0], activity_mag[0],inhibit_scale[0])
            decoded_output[i]=multiResolutionUpdate(input[i],prev_weights_trans,net0)
            
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
    
    fig.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    plt.gcf().text(0.02,0.02,"N= " + str(N[1]) +",  num links= " + str(num_links[1]) + ",  excite links= " + str(excite[1]) + ", inhibition=" + str(inhibit_scale[1]),  fontsize=8)

    plt.show()

# modify network so zero doesnt get decoded as 59
def visualiseMultiResolutionTranslation(data_x,data_y):
    global prev_weights, num_links, excite, activity_mag,inhibit_scale, curr_parameter
    '''initlising network and animate figures'''
    fig = plt.figure(figsize=(13, 4))
    ax0 =  plt.subplot2grid(shape=(5, 3), loc=(0, 0), rowspan=5,colspan=1)
    ax10 = plt.subplot2grid(shape=(5, 3), loc=(0, 1), rowspan=1,colspan=1)
    ax11 = plt.subplot2grid(shape=(5, 3), loc=(1, 1), rowspan=1,colspan=1)
    ax12 = plt.subplot2grid(shape=(5, 3), loc=(2, 1), rowspan=1,colspan=1)
    ax13 = plt.subplot2grid(shape=(5, 3), loc=(3, 1), rowspan=1,colspan=1)
    ax14 = plt.subplot2grid(shape=(5, 3), loc=(4, 1), rowspan=1,colspan=1)
    ax2 = plt.subplot2grid(shape=(5, 3), loc=(0, 2), rowspan=5,colspan=1)
    fig.tight_layout()

    net=attractorNetworkScaling(N[0],num_links[0],excite[0], activity_mag[0],inhibit_scale[0])
    for n in range(len(prev_weights_trans)):
        prev_weights_trans[n][net.activation(0)]=net.full_weights(num_links[0])

    def multiResolutionUpdate(input,prev_weights,net): 
        scale = [0.01, 0.1, 0.5, 1, 10]
        delta = [(input/scale[0]), (input/scale[1]), (input/scale[2]), (input/scale[3]), (input/scale[4])]
        split_output=np.zeros((len(delta)))
        '''updating network'''    
        for k in range(5):
            for n in range(len(delta)):
                prev_weights[n][:]= net.update_weights_dynamics(prev_weights[n][:],delta[n])
                prev_weights[n][prev_weights[n][:]<0]=0
                split_output[n]=np.argmax(prev_weights[n][:])
        print(split_output)
        decoded=np.sum(split_output*scale)*np.sign(input) 

        ax10.set_title(str(scale[0])+" Scale",fontsize=9)
        ax11.set_title(str(scale[1])+" Scale",fontsize=9)
        ax12.set_title(str(scale[2])+" Scale",fontsize=9)
        ax13.set_title(str(scale[3])+" Scale",fontsize=9)
        ax14.set_title(str(scale[4])+" Scale",fontsize=9)

        return decoded  

    def animate(i):
        global prev_weights_trans,prev_weights_rot, num_links, excite, activity_mag,inhibit_scale
        ax10.clear(),ax11.clear(),ax12.clear(),ax13.clear(),ax14.clear()
        
        if i>=2:
            '''encoding mangnitude and direction of movement'''
            x0=data_x[i-2]
            x1=data_x[i-1]
            x2=data_x[i]
            y0=data_y[i-2]
            y1=data_y[i-1]
            y2=data_y[i]
            
            translation=np.sqrt(((x2-x1)**2)+((y2-y1)**2))#translation
            rotation=((np.rad2deg(math.atan2(y2-y1,x2-x1)) - np.rad2deg(math.atan2(y1-y0,x1-x0))))#%360     #angle

            net0=attractorNetworkScaling(N[0],num_links[0],excite[0], activity_mag[0],inhibit_scale[0])
            decoded_translation=multiResolutionUpdate(translation,prev_weights_trans,net0)
    
            curr_parameter[0]=curr_parameter[0]+translation    
            print(f"{str(i)}  translation {translation} input output {round(curr_parameter[0],3)}  {str(decoded_translation )}  ")
            
            ax0.set_title("Ground Truth")
            ax0.plot(curr_parameter[0],0,"k.")
            # ax0.set_xlim([0,len(data_x)])
            # ax0.axis('equal')

            ax2.set_title("Decoded Translation")
            ax2.plot(decoded_translation,0,'g.')
            # ax2.set_xlim([0,len(data_x)])
            # ax2.axis('equal')

            ax10.bar(np.arange(N[0]),prev_weights_trans[0][:],color='aqua')
            ax10.axis('off')

            ax11.bar(np.arange(N[0]),prev_weights_trans[1][:],color='green')
            ax11.axis('off')

            ax12.bar(np.arange(N[0]),prev_weights_trans[2][:],color='blue')
            ax12.axis('off')
        
            ax13.bar(np.arange(N[0]),prev_weights_trans[3][:],color='purple')
            ax13.axis('off')
            
            ax14.bar(np.arange(N[0]),prev_weights_trans[4][:],color='pink')
            ax14.axis('off')

    ani = FuncAnimation(fig, animate, interval=1,frames=len(data_x),repeat=False)
    plt.show()



'''Translation Only'''
data_y=np.zeros(1000)
data_x=np.concatenate([ np.linspace(0,10,100), np.linspace(10,110,100),np.linspace(110,1100,100)])
# data_x=np.linspace(0,9990,1000)

sparse_gt=data_processing()#[0::4]
data_x=sparse_gt[:, :, 3][:,0][:200]
data_y=sparse_gt[:, :, 3][:,2][:200]

# visualiseMultiResolutionFeedthroughTranslation(data_x,data_y)
# multiResolutionTranslation(data_x,data_y)
visualiseMultiResolutionTranslation(data_x,data_y)


# def multiResolutionModulus(input,split_output):
#     rounded=np.round(input,2)*100

#     hundreds=(rounded)%10
#     tens=(rounded//10)%10
#     ones=(rounded//100)

#     scale=[1,0.1,0.01]
#     return [ones,tens,hundreds], scale 

# def visualiseMultiResolutionModulus(data_x,data_y):
#     global prev_weights, num_links, excite, activity_mag,inhibit_scale, curr_parameter
#     # global curr_x,curr_y
#     fig = plt.figure(figsize=(13, 4))
#     ax0 =  plt.subplot2grid(shape=(3, 3), loc=(0, 0), rowspan=5,colspan=1)
#     ax10 = plt.subplot2grid(shape=(3, 3), loc=(0, 1), rowspan=1,colspan=1)
#     ax11 = plt.subplot2grid(shape=(3, 3), loc=(1, 1), rowspan=1,colspan=1)
#     ax12 = plt.subplot2grid(shape=(3, 3), loc=(2, 1), rowspan=1,colspan=1)
#     ax2 = plt.subplot2grid(shape=(3, 3), loc=(0, 2), rowspan=5,colspan=1)
#     fig.tight_layout()

#     curr_x,curr_y=np.zeros((len(data_x))), np.zeros((len(data_y)))
#     theta=np.zeros((len(data_x)))
#     theta[0]=0
#     theta[1]=90
    
   
#     net=attractorNetwork(N[1],num_links[1],excite[1], activity_mag[1],inhibit_scale[1])
#     prev_weights[1][net.activation(theta[1])]=net.full_weights(num_links[1])
    
#     for n in range(len(prev_weights_rot)):
#         prev_weights_rot[n][net.activation(90)]=net.full_weights(num_links[1])


#     def animate(i):
#         global prev_weights_trans,prev_weights_rot, num_links, excite, activity_mag,inhibit_scale,split_output
#         # ax10.clear(),ax11.clear(),ax12.clear(),ax13.clear(),ax14.clear()
        
#         if i>=2:
#             '''encoding mangnitude and direction of movement'''
#             x0=data_x[i-2]
#             x1=data_x[i-1]
#             x2=data_x[i]
#             y0=data_y[i-2]
#             y1=data_y[i-1]
#             y2=data_y[i]
            
#             translation=np.sqrt(((x2-x1)**2)+((y2-y1)**2))#translation
#             rotation=((np.rad2deg(math.atan2(y2-y1,x2-x1)) - np.rad2deg(math.atan2(y1-y0,x1-x0))))#%360     #angle

#             net=attractorNetwork(N[0],num_links[0],excite[0], activity_mag[0],inhibit_scale[0])
#             decoded_translation=multiResolutionUpdate(translation,prev_weights_trans,net)
    
#             # net=attractorNetwork(N[1],num_links[1],excite[1], activity_mag[1],inhibit_scale[1])
#             # prev_weights[1][:]= net.update_weights_dynamics(prev_weights[1][:],rotation)
#             # prev_weights[1][prev_weights[1][:]<0]=0
#             # decoded_rotation=activityDecodingAngle(prev_weights[1][:],num_links[1],N[1])*1.09#-prev_trans
#             # # decoded_rotation=np.argmax(prev_weights[1][:])
            
#             delta, scale = multiResolutionModulus(abs(rotation),split_output)
#             split_output=np.zeros((len(delta)))

#             print(rotation,delta)

#             '''updating network'''    
#             net=attractorNetwork(N[1],num_links[1],excite[1], activity_mag[1],inhibit_scale[1])
            
#             for n in range(len(delta)):
#                 prev_weights_rot[n][:]= net.update_weights_dynamics(prev_weights_rot[n][:],delta[n])
#                 prev_weights_rot[n][prev_weights_rot[n][:]<0]=0
#                 split_output[n]=np.argmax(prev_weights_rot[n][:])#-prev_trans
#             '''decoding mangnitude and direction of movement'''
#             decoded_rotation=np.sum(split_output*scale)*np.sign(rotation)
    
#             # theta[i]=(theta[i-1]+rotation)%360
#             # curr_x[i]=curr_x[i-1]+(translation*np.cos(np.deg2rad(theta[i-1])))
#             # curr_y[i]=curr_y[i-1]+(translation*np.sin(np.deg2rad(theta[i-1])))

            
#             # curr_x[i]=curr_x[i-1]+ (decoded_translation*np.cos(np.deg2rad(theta[i-1])))
#             # curr_y[i]=curr_y[i-1]+ (decoded_translation*np.sin(np.deg2rad(theta[i-1])))
#             curr_x[i]=curr_x[i-1]+ (decoded_translation*np.cos(np.deg2rad(decoded_rotation)))
#             curr_y[i]=curr_y[i-1]+ (decoded_translation*np.sin(np.deg2rad(decoded_rotation)))

        
#             # print(f"{str(i)}   {str(translation)}   {str(decoded_translation )}  ----- {np.rad2deg(math.atan2(y2-y1,x2-x1))}   {decoded_rotation}")
#             '''decoding mangnitude and direction of movement'''
            
#             ax0.plot(x2,y2,"k.")
#             ax0.set_title("Ground Truth Position")
#             ax0.axis('equal')
#             # ax0.axis('equal')  
#             # 
#             ax2.set_title("Attractor Rotation Network")
#             ax2.plot(curr_x[i],curr_y[i],'g.')
#             ax2.axis('equal')

#             # ax10.set_title("Whole Deg",fontsize=6)
#             # ax10.bar(np.arange(N[1]),prev_weights_rot[0][:],color='blue')
#             # ax10.axis('off')

#             # ax11.set_title("1/10 Deg",fontsize=6)
#             # ax11.bar(np.arange(N[1]),prev_weights_rot[1][:],color='purple')
#             # ax11.axis('off')

#             # ax12.set_title("1/100 Deg",fontsize=6)
#             # ax12.bar(np.arange(N[1]),prev_weights_rot[2][:],color='pink')
#             # ax12.axis('off')
            

#     ani = FuncAnimation(fig, animate, interval=1,frames=len(data_x),repeat=False)
#     plt.show()

