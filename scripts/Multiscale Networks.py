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
N=[100, 60] #number of neurons
curr_Neuron=[0,0]
prev_weights=[np.zeros(N[0]), np.zeros(N[1])]
split_output=[0,0,0]
num_links=[3,17]
excite=[1,7]
# activity_mag=[1,1]

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
# for n in range(len(prev_weights_trans)):
#     net=attractorNetworkScaling(N[0],num_links[0],excite[0], activity_mag[0],inhibit_scale[0])
#     prev_weights_trans[n][net.activation(0)]=net.full_weights(num_links[0])

# '''Initiliase Rotating Weights'''
# prev_weights_rot=[np.zeros(N[1]), np.zeros(N[1]), np.zeros(N[1]),np.zeros(N[1]), np.zeros(N[1])]
# net=attractorNetworkScaling(N[1],num_links[0],excite[0], activity_mag[0],inhibit_scale[0])
# for n in range(len(prev_weights_rot)):
#     prev_weights_rot[n][net.activation(0)]=net.full_weights(num_links[0])

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
    N[0]=20
    '''Initiliase Translating Weights'''
    prev_weights_trans=[np.zeros(N[0]), np.zeros(N[0]), np.zeros(N[0]),np.zeros(N[0]), np.zeros(N[0])]
    for n in range(len(prev_weights_trans)):
        net=attractorNetworkScaling(N[0],num_links[0],excite[0], activity_mag[0],inhibit_scale[0])
        prev_weights_trans[n][net.activation(0)]=net.full_weights(num_links[0])

    def multiResolutionUpdate(input,prev_weights,net):
        '''Uptate the activity (previous weights) of scaled networks based on input velocity'''
        #initiliaise network and scale
        scale = [(1/(N[0]**2)), (1/N[0]), 1, N[0], N[0]**2]
        split_output=np.zeros((len(scale)))
        # net=attractorNetwork(N[0],num_links[0],excite[0], activity_mag[0],inhibit_scale[0])

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
        # global prev_weights_trans,prev_weights_rot, num_links, excite, activity_mag,inhibit_scale
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
            axtxt.text(0,0.2,f"Input Trans: {round(x2,3)}, Shift: {round(translation,4)}, Decoded Trans: {round(decoded_translation,3)}", c='r')
            # axtxt.text(0,0,"Input Rot: " +str(round(rotation,3))+ " " + str(round(decoded_rotation,3)), c='m')
            axtxt1.text(0,0,"Decoded Position of Each Network: " + str(split_trans), c='r')
            
            ax0.set_title("Ground Truth Translation")
            ax0.plot(i, x2,"g.")
            # ax0.axis('equal')
            ax2.set_title("Decoded Translation")
            ax2.plot(i,decoded_translation,'k.')
            # ax2.axis('equal')

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
        scale =  [0.25, 0.5, 1, 2, 4]
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


def visualiseMultiResolutionTranslation(data_x,data_y,activity_mag,inhibit_scale):
    # global prev_weights, num_links, excite, activity_mag,inhibit_scale, curr_parameter
    '''initlising network and animate figures'''
    fig = plt.figure(figsize=(6, 6))
    ax10 = plt.subplot2grid(shape=(6, 1), loc=(0, 0), rowspan=1,colspan=1)
    ax11 = plt.subplot2grid(shape=(6, 1), loc=(1, 0), rowspan=1,colspan=1)
    ax12 = plt.subplot2grid(shape=(6, 1), loc=(2, 0), rowspan=1,colspan=1)
    ax13 = plt.subplot2grid(shape=(6, 1), loc=(3, 0), rowspan=1,colspan=1)
    ax14 = plt.subplot2grid(shape=(6, 1), loc=(4, 0), rowspan=1,colspan=1)
    axtxt1 = plt.subplot2grid(shape=(6, 1), loc=(5, 0), rowspan=1,colspan=1)
    fig.tight_layout()

    num_links=[7,17]
    excite=[3,7]
    # activity_mag=0.15
    # inhibit_scale=0.4

    net=attractorNetworkScaling(N[0],num_links[0],excite[0], activity_mag,inhibit_scale)
    for n in range(len(prev_weights_trans)):
        prev_weights_trans[n][net.activation(0)]=net.full_weights(num_links[0])

    def multiResolutionUpdate(input,prev_weights,net): 
        scale =  [0.25, 0.5, 1, 2, 4]
        delta = [(input/scale[0]), (input/scale[1]), (input/scale[2]), (input/scale[3]), (input/scale[4])]
        split_output=np.zeros((len(delta)))
        '''updating network'''    
        for n in range(len(delta)):
            prev_weights[n][:]= net.update_weights_dynamics(prev_weights[n][:],delta[n])
            prev_weights[n][prev_weights[n][:]<0]=0
            split_output[n]=np.argmax(prev_weights[n][:])
        
        decoded=np.sum(split_output*scale)*np.sign(input) 

        ax10.set_title(str(scale[0])+" Scale",fontsize=9)
        ax11.set_title(str(scale[1])+" Scale",fontsize=9)
        ax12.set_title(str(scale[2])+" Scale",fontsize=9)
        ax13.set_title(str(scale[3])+" Scale",fontsize=9)
        ax14.set_title(str(scale[4])+" Scale",fontsize=9)

        return decoded, split_output

    def animate(i):
        # global prev_weights_trans, num_links, excite,inhibit_scale
        ax10.clear(),ax11.clear(),ax12.clear(),ax13.clear(),ax14.clear(),axtxt1.clear()
        
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

            net0=attractorNetworkScaling(N[0],num_links[0],excite[0], activity_mag,inhibit_scale)
            decoded_translation,split_trans=multiResolutionUpdate(translation,prev_weights_trans,net0)
    
            curr_parameter[0]=curr_parameter[0]+translation    
            print(f"{str(i)}  translation {translation} input output {round(curr_parameter[0],3)}  {str(decoded_translation )}  ")
            
            
            axtxt1.text(0,0.5,f"Input Trans: {round(x2,3)}, Shift: {round(translation,4)}, Decoded Trans: {round(decoded_translation,3)}", c='r')
            # axtxt.text(0,0,"Input Rot: " +str(round(rotation,3))+ " " + str(round(decoded_rotation,3)), c='m')
            axtxt1.axis('off')
            axtxt1.text(0,0,"Decoded Position of Each Network: " + str(split_trans), c='r')

            ax10.bar(np.arange(N[0]),prev_weights_trans[0][:],color='aqua')
            ax10.get_xaxis().set_visible(False)
            ax10.spines[['top', 'bottom', 'right']].set_visible(False)

            ax11.bar(np.arange(N[0]),prev_weights_trans[1][:],color='green')
            ax11.get_xaxis().set_visible(False)
            ax11.spines[['top', 'bottom', 'right']].set_visible(False)

            ax12.bar(np.arange(N[0]),prev_weights_trans[2][:],color='blue')
            ax12.get_xaxis().set_visible(False)
            ax12.spines[['top', 'bottom', 'right']].set_visible(False)
        
            ax13.bar(np.arange(N[0]),prev_weights_trans[3][:],color='purple')
            ax13.get_xaxis().set_visible(False)
            ax13.spines[['top', 'bottom', 'right']].set_visible(False)
            
            ax14.bar(np.arange(N[0]),prev_weights_trans[4][:],color='pink')
            ax14.get_xaxis().set_visible(False)
            ax14.spines[['top', 'bottom', 'right']].set_visible(False)

    ani = FuncAnimation(fig, animate, interval=1,frames=len(data_x),repeat=False)
    plt.show()


def MultiResolutionTranslation(data_x,data_y,activity_mag,inhibit_scale,input_idx):
    # parameters
        N=100
        num_links=7
        excite=3
        scale = [0.01, 0.1, 1, 10, 100]
        error=0
        '''initiliase network'''
        net=attractorNetworkScaling(N,num_links,excite, activity_mag,inhibit_scale)
        prev_weights=[np.zeros(N), np.zeros(N), np.zeros(N),np.zeros(N), np.zeros(N)]
        for n in range(len(prev_weights)):
            prev_weights[n][net.activation(0)]=net.full_weights(num_links)
        
        delta_peak=np.zeros((len(data_x),len(scale)))
        split_output=np.zeros((len(data_x),len(scale)))

        for i in range(1,len(data_x)):
            activity_len=[len(np.arange(N)[weights>0]) for weights in prev_weights]
            if 0 in activity_len:
                error = 1000
            else: 
                '''encoding mangnitude movement into multiple scales'''
                x1, x2=data_x[i-1], data_x[i]
                y1, y2= data_y[i-1], data_y[i]
                
                input=np.sqrt(((x2-x1)**2)+((y2-y1)**2))#translation
                delta = [(input/scale[0]), (input/scale[1]), (input/scale[2]), (input/scale[3]), (input/scale[4])]
                
                '''updating network'''    
                for n in range(len(delta)):
                    prev_weights[n][:]= net.update_weights_dynamics(prev_weights[n][:],delta[n])
                    prev_weights[n][prev_weights[n][:]<0]=0
                    split_output[i,n]=np.argmax(prev_weights[n][:])
                delta_peak[i,:]=np.abs(split_output[i,:]-split_output[i-1,:])
                decoded=np.sum(split_output*scale)*np.sign(input) 
                for j in range(len(scale)):
                    if j != input_idx:
                        error+=np.sum([peak[j] for peak in delta_peak])
                    else:
                        error-=delta_peak[i,j]
        return error
            
def gridSearch(filename,n_steps):
    error=np.zeros((n_steps,n_steps))
    inhibit= list(np.linspace(0.005,1,n_steps))
    magnitude= list(np.linspace(0.005,1,n_steps))
    for i,inh in enumerate(inhibit):
        for j,mag in enumerate(magnitude):
            error[i,j]=MultiResolutionTranslation(data_x,data_y,inh,mag,input_idx)
            print(i,j,error[i,j])
    with open(filename, 'wb') as f:
        np.save(f, np.array(error))

def plottingGridSearch(filename,n_steps):
    inhibit= list(np.linspace(0.005,1,n_steps))
    magnitude= list(np.linspace(0.005,1,n_steps))

    with open(filename, 'rb') as f:
        error = np.load(f)
        error[error==1000]=np.nan
        norm_error=error/np.linalg.norm(error)
    plt.figure(figsize=(10, 7))
    ax0=plt.subplot(1,1,1)

    zeros=np.vstack((np.where(error==0)[0],np.where(error==0)[1]))
    # print(inhibit[zeros[0,0]],magnitude[zeros[1,0]])


    ax0.set_title('Error')
    # error[error==0]=np.nan
    ax0.imshow(np.log(error))
    ax0.set_xlabel('Inhibition')
    ax0.set_ylabel('Magnitude')
    ax0.set_xticks(np.arange(n_steps),[round(a,4) for a in inhibit],rotation=90)
    ax0.set_yticks(np.arange(n_steps), [round(a,4) for a in magnitude])
    # ax0.grid(True)

    plt.show()

    return zeros 

       

           

'''Translation Only'''
input_idx=2
data_x=np.arange(0,200,1)#np.concatenate([ np.arange(0,5,0.25), np.arange(5,25,0.5),np.arange(25,50,1),np.arange(50,100,2),np.arange(100,200,4)])
data_y=np.zeros(len(data_x))
# data_x=np.linspace(0,9990,1000)

# sparse_gt=data_processing()#[0::4]
# data_x=sparse_gt[:, :, 3][:,0][:200]
# data_y=sparse_gt[:, :, 3][:,2][:200]

# visualiseMultiResolutionFeedthroughTranslation(data_x,data_y)
# multiResolutionTranslation(data_x,data_y)
# visualiseMultiResolutionTranslation(data_x,data_y)

# f'./results/GridSearch_MultiScale/mutliScale_factor_of_2.npy'
# f'./results/GridSearch_MultiScale/mutli_scale_index_{input_idx}.npy'
# gridSearch(f'./results/GridSearch_MultiScale/mutliScale_factor_test10.npy',10)

zeros=plottingGridSearch( f'./results/GridSearch_MultiScale/mutliScale_factor_test10.npy',10)
# inhibit= list(np.linspace(0.005,1,5))
# magnitude= list(np.linspace(0.005,1,5))
# visualiseMultiResolutionTranslation(data_x,data_y,0.005,0.2537)