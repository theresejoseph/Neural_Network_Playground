import math
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time

class attractorNetwork:
    '''defines 1D attractor network with N neurons, angles associated with each neurons 
    along with inhitory and excitatory connections to update the weights'''
    def __init__(self, N, num_links, excite_radius, activity_mag,inhibit_scale):
        self.excite_radius=excite_radius
        self.N=N  
        self.num_links=num_links
        self.activity_mag=activity_mag
        self.inhibit_scale=inhibit_scale

    def inhibitions(self,id):
        ''' each nueuron inhibits all other nueurons but itself'''
        return np.delete(np.arange(self.N),self.excitations(id))

    def excitations(self,id):
        '''each neuron excites itself and num_links neurons left and right with wraparound connections'''
        excite=[]
        for i in range(-self.excite_radius,self.excite_radius+1):
            excite.append((id + i) % self.N)
        return np.array(excite)

    def activation(self,id):
        '''each neuron excites itself and num_links neurons left and right with wraparound connections'''
        excite=[]
        for i in range(-self.num_links,self.num_links+1):
            excite.append((int(id) + i) % self.N)
        return np.array(excite)

    def full_weights(self,radius):
        x=np.arange(-radius,radius+1)
        return 1/(np.std(x) * np.sqrt(2 * np.pi)) * np.exp( - (x - np.mean(x))**2 / (2 * np.std(x)**2))  

    def fractional_weights(self,non_zero_prev_weights,delta):
        frac=delta%1
        if frac == 0:
            return non_zero_prev_weights
        else: 
            inv_frac=1-frac
            frac_weights=np.zeros((len(non_zero_prev_weights)))
            frac_weights[0]=non_zero_prev_weights[0]*inv_frac
            for i in range(1,len(non_zero_prev_weights)):
                frac_weights[i]=non_zero_prev_weights[i-1]*frac + non_zero_prev_weights[i]*inv_frac
            return frac_weights

    def frac_weights_1D(self, prev_weights, delta):
        mysign=lambda x: 1 if x > 0 else -1
        frac=delta - int(delta)
        inv_frac=1-abs(frac)

        non_zero_idxs=np.nonzero(prev_weights)[0]
        shifted_weights=np.zeros(self.N)
        shifted_weights[(non_zero_idxs+mysign(frac))%self.N]=prev_weights[non_zero_idxs]
        return prev_weights*inv_frac + shifted_weights*abs(frac)

    def update_weights_dynamics(self,prev_weights, delta, moreResults=None, cross=None):
        
        indexes,non_zero_weights,full_shift,inhibit_val=np.arange(self.N),np.zeros(self.N),np.zeros(self.N),0
        non_zero_idxs=indexes[prev_weights>0] # indexes of non zero prev_weights
        '''copied and shifted activity'''
        full_shift_amount=lambda x: np.floor(x) if x > 0 else np.ceil(x)
        full_shift[(non_zero_idxs + int(full_shift_amount(delta)))%self.N]=prev_weights[non_zero_idxs]*self.activity_mag
        # print(int(np.floor(delta)))
        shift=self.frac_weights_1D(full_shift,delta)  #non zero weights shifted by delta
        copy_shift=shift+prev_weights
        shifted_indexes=np.nonzero(copy_shift)[0]
        '''excitation'''
        excitations_store=np.zeros((len(shifted_indexes),self.N))
        excitation_array,excite=np.zeros(self.N),np.zeros(self.N)
        for i in range(len(shifted_indexes)):
            excitation_array[self.excitations(shifted_indexes[i])]=self.full_weights(self.excite_radius)*prev_weights[shifted_indexes[i]]
            excitations_store[i,:]=excitation_array
            excite[self.excitations(shifted_indexes[i])]+=self.full_weights(self.excite_radius)*prev_weights[shifted_indexes[i]]
        '''inhibit'''
        # shift_excite=copy_shift
        non_zero_inhibit=np.nonzero(excite) 
        for idx in non_zero_inhibit[0]:
            inhibit_val+=excite[idx]*self.inhibit_scale
        '''update activity'''
        prev_weights+=(copy_shift+excite-inhibit_val)
        prev_weights=prev_weights/np.linalg.norm(prev_weights)

        if moreResults==True:
           return prev_weights/np.linalg.norm(prev_weights), non_zero_weights,[inhibit_val]*self.N, excitations_store
        else: 
            return prev_weights/np.linalg.norm(prev_weights)

        

class attractorNetworkSettling:
    '''defines 1D attractor network with N neurons, angles associated with each neurons 
    along with inhitory and excitatory connections to update the weights'''
    def __init__(self, N, num_links, excite_radius, activity_mag,inhibit_scale):
        self.excite_radius=excite_radius
        self.N=N  
        self.num_links=num_links
        self.activity_mag=activity_mag
        self.inhibit_scale=inhibit_scale
        
    def inhibitions(self,id):
        ''' each nueuron inhibits all other nueurons but itself'''
        return np.delete(np.arange(self.N),self.excitations(id))

    def excitations(self,id):
        '''each neuron excites itself and num_links neurons left and right with wraparound connections'''
        excite=[]
        for i in range(-self.excite_radius,self.excite_radius+1):
            excite.append((id + i) % self.N)
        return np.array(excite)

    def activation(self,id):
        '''each neuron excites itself and num_links neurons left and right with wraparound connections'''
        excite=[]
        for i in range(-self.num_links,self.num_links+1):
            excite.append((int(id) + i) % self.N)
        return np.array(excite)

    def full_weights(self,radius):
        x=np.arange(-radius,radius+1)
        return 1/(np.std(x) * np.sqrt(2 * np.pi)) * np.exp( - (x - np.mean(x))**2 / (2 * np.std(x)**2))  

    def fractional_weights(self,non_zero_prev_weights,activeNeuron):
        frac=activeNeuron%1
        if frac == 0:
            return non_zero_prev_weights
        else: 
            inv_frac=1-frac
            frac_weights=np.zeros((len(non_zero_prev_weights)))
            frac_weights[0]=non_zero_prev_weights[0]*inv_frac
            for i in range(1,len(non_zero_prev_weights)):
                frac_weights[i]=non_zero_prev_weights[i-1]*frac + non_zero_prev_weights[i]*inv_frac
            return frac_weights

    def update_weights_dynamics(self,prev_weights,activeNeuron,moreResults=None):

        delta=(int(activeNeuron)-np.argmax(prev_weights))%self.N

        indexes,non_zero_weights,non_zero_weights_shifted, inhbit_val=np.arange(self.N),np.zeros(self.N),np.zeros(self.N),0
        # shifted_indexes=self.neuron_update(prev_weights)

        '''copied and shifted activity'''
        non_zero_idxs=indexes[prev_weights>0] # indexes of non zero prev_weights

        if len(prev_weights[non_zero_idxs])==0:
            prev_weights[self.activation(activeNeuron)]=self.full_weights(self.num_links)
            non_zero_idxs=indexes[prev_weights>0] # indexes of non zero prev_weights
        
        non_zero_weights[non_zero_idxs]=prev_weights[non_zero_idxs] 
        
        non_zero_weights_shifted[(non_zero_idxs+delta)%self.N]=self.fractional_weights(prev_weights[non_zero_idxs],activeNeuron) #non zero weights shifted by delta
        
        '''inhibition'''
        for i in range(len(non_zero_weights_shifted)):
            inhbit_val+=non_zero_weights_shifted[i]*self.inhibit_scale
        
        '''excitation'''
        excitations_store=np.zeros((len(non_zero_idxs),self.N))
        excitation_array,excite=np.zeros(self.N),np.zeros(self.N)
        for i in range(len(non_zero_idxs)):
            excitation_array[self.excitations(non_zero_idxs[i])]=self.full_weights(self.excite_radius)*prev_weights[non_zero_idxs[i]]
            excitations_store[i,:]=excitation_array
            excite[self.excitations(non_zero_idxs[i])]+=self.full_weights(self.excite_radius)*prev_weights[non_zero_idxs[i]]

        prev_weights+=(non_zero_weights_shifted+excite-inhbit_val)
        if moreResults==True:
           return prev_weights/np.linalg.norm(prev_weights), non_zero_weights_shifted, excitations_store, inhbit_val
        else:  
           return prev_weights/np.linalg.norm(prev_weights)


class attractorNetworkScaling:
    '''defines 1D attractor network with N neurons, angles associated with each neurons 
    along with inhitory and excitatory connections to update the weights'''
    def __init__(self, N, num_links, excite_radius, activity_mag,inhibit_scale):
        self.excite_radius=excite_radius
        self.N=N  
        self.num_links=num_links
        self.activity_mag=activity_mag
        self.inhibit_scale=inhibit_scale
        
    def inhibitions(self,id):
        ''' each nueuron inhibits all other nueurons but itself'''
        return np.delete(np.arange(self.N),self.excitations(id))

    def excitations(self,id):
        '''each neuron excites itself and num_links neurons left and right with wraparound connections'''
        excite=[]
        for i in range(-self.excite_radius,self.excite_radius+1):
            excite.append((id + i) % self.N)
        return np.array(excite)

    def activation(self,id):
        '''each neuron excites itself and num_links neurons left and right with wraparound connections'''
        excite=[]
        for i in range(-self.num_links,self.num_links+1):
            excite.append((int(id) + i) % self.N)
        return np.array(excite)

    def full_weights(self,radius):
        x=np.arange(-radius,radius+1)
        return 1/(np.std(x) * np.sqrt(2 * np.pi)) * np.exp( - (x - np.mean(x))**2 / (2 * np.std(x)**2))  

    def fractional_weights(self,non_zero_prev_weights,activeNeuron):
        frac=activeNeuron%1
        if frac == 0:
            return non_zero_prev_weights
        else: 
            inv_frac=1-frac
            frac_weights=np.zeros((len(non_zero_prev_weights)))
            frac_weights[0]=non_zero_prev_weights[0]*inv_frac
            for i in range(1,len(non_zero_prev_weights)):
                frac_weights[i]=non_zero_prev_weights[i-1]*frac + non_zero_prev_weights[i]*inv_frac
            return frac_weights

    def update_weights_dynamics(self,prev_weights,delta,moreResults=None):
        indexes,non_zero_weights,non_zero_weights_shifted, inhbit_val=np.arange(self.N),np.zeros(self.N),np.zeros(self.N),0

        # '''initialise'''
        non_zero_idxs=indexes[prev_weights>0] # indexes of non zero prev_weights
        # if len(non_zero_idxs)==0:
        #     prev_weights[self.activation(0)]=self.full_weights(self.num_links)
        #     non_zero_idxs=indexes[prev_weights>0] # indexes of non zero prev_weights    
        '''copied and shifted activity'''
        non_zero_weights[non_zero_idxs]=prev_weights[non_zero_idxs] 
        non_zero_weights_shifted[(non_zero_idxs+round(delta))%self.N]=self.fractional_weights(prev_weights[non_zero_idxs],delta)*self.activity_mag #non zero weights shifted by delta
  
        '''inhibition''' #inhibit shifted neuron 
        for i in range(len(non_zero_weights_shifted)):
            inhbit_val+=non_zero_weights_shifted[i]*self.inhibit_scale
        
        '''excitation''' # excite all active neurons 
        excitations_store=np.zeros((len(non_zero_idxs),self.N))
        excitation_array,excite=np.zeros(self.N),np.zeros(self.N)
        for i in range(len(non_zero_idxs)):
            excitation_array[self.excitations(non_zero_idxs[i])]=self.full_weights(self.excite_radius)*prev_weights[non_zero_idxs[i]]
            excitations_store[i,:]=excitation_array
            excite[self.excitations(non_zero_idxs[i])]+=self.full_weights(self.excite_radius)*prev_weights[non_zero_idxs[i]]

        
        # if abs(delta)<=self.excite_radius:
        prev_weights+=(non_zero_weights_shifted+excite-inhbit_val)
        return prev_weights/np.linalg.norm(prev_weights)
 
 
class attractorNetwork2D:
    '''defines 1D attractor network with N neurons, angles associated with each neurons 
    along with inhitory and excitatory connections to update the weights'''
    def __init__(self, N1, N2, num_links, excite_radius, activity_mag,inhibit_scale):
        self.excite_radius=excite_radius
        self.num_links=num_links
        self.N1=N1
        self.N2=N2  
        self.activity_mag=activity_mag
        self.inhibit_scale=inhibit_scale

    def full_weights(self,radius):
        len=(radius*2)+1
        x, y = np.meshgrid(np.linspace(-1,1,len), np.linspace(-1,1,len))
        sigma, mu = 1.0, 0.0
        return np.exp(-( ((x-mu)**2 + (y-mu)**2) / ( 2.0 * sigma**2 ) ) )

    def inhibitions(self,weights):
        ''' constant inhibition scaled by amount of active neurons'''
        return np.sum(weights[weights>0]*self.inhibit_scale)

    def excitations(self,idx,idy,scale=1):
        '''A scaled 2D gaussian with excite radius is created at given neruon position with wraparound '''
        
        excite_rowvals=[] #wrap around row values 
        excite_colvals=[] #wrap around column values 
        for i in range(-self.excite_radius,self.excite_radius+1):
            excite_rowvals.append((idx + i) % self.N1)
            excite_colvals.append((idy + i) % self.N2)
         

        gauss=self.full_weights(self.excite_radius)# 2D gaussian scaled 
        excite=np.zeros((self.N1,self.N2)) # empty excite array 
        for i,r in enumerate(excite_rowvals):
            for j,c in enumerate(excite_colvals):
                excite[r,c]=gauss[i,j]
        return excite*scale 

    def neuron_activation(self,idx,idy):
        '''A scaled 2D gaussian with excite radius is created at given neruon position with wraparound '''
        
        excite_rowvals=[] #wrap around row values 
        excite_colvals=[] #wrap around column values 
        for i in range(-self.num_links,self.num_links+1):
            excite_rowvals.append((idx + i) % self.N1)
            excite_colvals.append((idy + i) % self.N2)
         

        gauss=self.full_weights(self.num_links)# 2D gaussian scaled 
        excite=np.zeros((self.N1,self.N2)) # empty excite array 
        for i,r in enumerate(excite_rowvals):
            for j,c in enumerate(excite_colvals):
                excite[r,c]=gauss[i,j]
        return excite

    def fractional_weights(self,full_shift,delta_row,delta_col):
        mysign=lambda x: 1 if x > 0 else -1
        frac_row, frac_col=delta_row - int(delta_row),delta_col - int(delta_col)
        inv_frac_row, inv_frac_col=[1-(delta_row%1),1-(delta_col%1)]

        non_zero_weights=np.nonzero(full_shift)
        shifted_row, shifted_col=np.zeros((self.N1,self.N2)), np.zeros((self.N1,self.N2))
        shifted_col[non_zero_weights[0],(non_zero_weights[1]+mysign(frac_col))%self.N2]=full_shift[non_zero_weights]
        shifted_row[(non_zero_weights[0]+mysign(frac_row))%self.N1, non_zero_weights[1]]=full_shift[non_zero_weights]
        
        shifted_rowThencol, shifted_colThenrow=np.zeros((self.N1,self.N2)), np.zeros((self.N1,self.N2))
        non_zero_col, non_zero_row=np.nonzero(shifted_col), np.nonzero(shifted_row)
        shifted_colThenrow[(non_zero_col[0]+ mysign(frac_row))%self.N1, non_zero_col[1]]=shifted_col[non_zero_col]
        shifted_rowThencol[non_zero_row[0], (non_zero_row[1]+ mysign(frac_col))%self.N2]=shifted_row[non_zero_row]

        col=full_shift*inv_frac_col + shifted_col*abs(frac_col)
        colRow=col*inv_frac_row + shifted_colThenrow*abs(frac_row)

        row=full_shift*inv_frac_row + shifted_row*abs(frac_row)
        rowCol=row*inv_frac_col + shifted_rowThencol*abs(frac_col)

        return (rowCol + colRow)/2
        
    def update_weights_dynamics(self,prev_weights, delta_row, delta_col,moreResults=None):
        non_zero_rows, non_zero_cols=np.nonzero(prev_weights) # indexes of non zero prev_weights

        '''copied and shifted activity'''
        full_shift=np.zeros((self.N1,self.N2))
        full_shift[(non_zero_rows + int(np.floor(delta_row)))%self.N1, (non_zero_cols+ int(np.floor(delta_col)))%self.N2]=prev_weights[non_zero_rows, non_zero_cols]
        copy_shift=self.fractional_weights(full_shift,delta_row,delta_col)*self.activity_mag

        '''excitation'''
        copyPaste=prev_weights+copy_shift
        non_zero_copyPaste=np.nonzero(copyPaste)  
        # print(len(non_zero_copyPaste[0]))
        excited=np.zeros((self.N1,self.N2))
        # t=time.time()
        for row, col in zip(non_zero_copyPaste[0], non_zero_copyPaste[1]):
            excited+=self.excitations(row,col,copyPaste[row,col])
        # print(time.time()-t)
        
        # excited=np.sum(excited_array, axis=0)
        # print(np.shape(excited_array), np.shape(excited))
        '''inhibitions'''
        inhibit_val=0
        shift_excite=copy_shift+prev_weights+excited
        non_zero_inhibit=np.nonzero(shift_excite) 
        for row, col in zip(non_zero_inhibit[0], non_zero_inhibit[1]):
            inhibit_val+=shift_excite[row,col]*self.inhibit_scale
        inhibit_array=np.tile(inhibit_val,(self.N1,self.N2))

        '''update activity'''
        # new_weights=np.zeros((self.N1,self.N2))
        prev_weights+=copy_shift+excited-inhibit_val
        prev_weights[prev_weights<0]=0

        if moreResults==True:
            return prev_weights/np.linalg.norm(prev_weights),copy_shift,excited,inhibit_array
        else:
            return prev_weights/np.linalg.norm(prev_weights) if np.sum(prev_weights) > 0 else [np.nan]

'''Tester Functions'''
def visulaiseFractionalWeights():
    fig = plt.figure(figsize=(5, 6))
    nrows=5
    ax0 = fig.add_subplot(nrows, 1, 1)
    ax1 = fig.add_subplot(nrows, 1, 2)
    ax2 = fig.add_subplot(nrows, 1, 3)
    ax3 = fig.add_subplot(nrows, 1, 4)
    ax4 = fig.add_subplot(nrows, 1, 5)
    fig.tight_layout()

    weights=np.array([0,0,0,0,0,2,3,4,5,4,3,2,0,0,0,0])
    N1,num_links,excite_radius,activity_mag,inhibit_scale= len(weights),6, 4, 1, 0.01
    net=attractorNetwork( N1,num_links,excite_radius,activity_mag,inhibit_scale)


    idx=np.nonzero(weights)[0]
    shifted_right,shifted_left=np.zeros(len(weights)),np.zeros(len(weights))
    shifted_right[idx+1]=weights[idx]
    shifted_left[idx-1]=weights[idx]
    frac=weights*0.25 + shifted_right*0.75
    frac2=weights*0.01 + shifted_right*0.99


    ax0.bar(np.arange(len(weights)),weights,color='r')
    ax0.set_ylim([0,10])
    ax0.set_title('Orginal')

    ax1.bar(np.arange(len(weights)),net.frac_weights_1D(weights,0.012),color='m')
    ax1.set_title('0.25 unit copy paste')
    ax1.set_ylim([0,10])

    ax2.bar(np.arange(len(weights)),net.frac_weights_1D(weights,0.5),color='m')
    ax2.set_title('0.5 unit copy paste')
    ax2.set_ylim([0,10])

    
    ax3.bar(np.arange(len(weights)),net.frac_weights_1D(weights,0.75), color='m')
    ax3.set_title('0.75 unit copy paste')
    ax3.set_ylim([0,10])

    ax4.bar(np.arange(len(weights)), shifted_right, color='b')
    ax4.set_title('1 unit Right')
    ax4.set_ylim([0,10])
    plt.show()

def visulaiseDeconstructed2DAttractor():
    fig = plt.figure(figsize=(13, 4))
    ax0 = fig.add_subplot(1, 4, 1)
    ax1 = fig.add_subplot(1, 4, 2)
    ax2 = fig.add_subplot(1, 4, 3)
    ax3 = fig.add_subplot(1, 4, 4)
    fig.tight_layout()

    N1,N2,excite_radius,activity_mag,inhibit_scale=  10, 10, 1, 1, 0.01
    delta_row, delta_col = 0.167, 0.33
    net=attractorNetwork2D( N1,N2,excite_radius,activity_mag,inhibit_scale)
    old_weights=net.excitations(0,9)

    ax0.imshow(old_weights)
    ax0.set_title('Previous Activity')

    old_weights, copy, excite,inhibit_array = net.update_weights_dynamics(old_weights,delta_row,delta_col,moreResults=True)
    ax1.imshow(copy)
    non_zero_copy=np.nonzero(copy)
    print(copy[non_zero_copy[0],non_zero_copy[1]])
    ax1.set_title('Copied and Shifted Activity')

    ax2.imshow(excite)
    ax2.set_title('Exctied Activity')

    ax3.imshow(old_weights)
    ax3.set_title('Inhibited Activity')
    plt.show()

def visulaise2DFractions():
    fig = plt.figure(figsize=(13, 4))
    ax0 = fig.add_subplot(1, 2, 1)
    ax3 = fig.add_subplot(1, 2, 2)
    fig.tight_layout()

    N1,N2,excite_radius,activity_mag,inhibit_scale=  100, 100, 1, 1, 0.01
    net=attractorNetwork2D( N1,N2,num_links,excite_radius,activity_mag,inhibit_scale)


    ax0.imshow(prev_weights)
    ax0.set_title('Previous Activity')
    ax0.invert_yaxis()

    def animate(i):
        ax3.clear(), ax0.clear()
        global prev_weights, another_prev_weights
        another_prev_weights=net.update_weights_dynamics(another_prev_weights,0.9,0.9)
        ax0.imshow(another_prev_weights)
        ax0.set_title('Copied and Shifted 0.9 Column 0.9 Row')
        ax0.invert_yaxis()

        prev_weights=net.update_weights_dynamics(prev_weights,0.1,0.1)
        ax3.imshow(prev_weights)
        ax3.set_title('Copied and Shifted 0.1 Column 0.1 Row')
        ax3.invert_yaxis()
    ani = FuncAnimation(fig, animate, interval=100,frames=1000,repeat=False)
    plt.show()

# visulaiseFractionalWeights()
# visulaiseDeconstructed2DAttractor()

# N1,N2,num_links,excite_radius,activity_mag,inhibit_scale=  100, 100,6, 4, 1, 0.01
# net=attractorNetwork2D( N1,N2,num_links,excite_radius,activity_mag,inhibit_scale)
# prev_weights=net.neuron_activation(5,5)
# another_prev_weights=net.neuron_activation(5,5)
# visulaise2DFractions()


# class attractorNetworkSettlingLandmark:
#     '''defines 1D attractor network with N neurons, angles associated with each neurons 
#     along with inhitory and excitatory connections to update the weights'''
#     def __init__(self, N, num_links, excite_radius, activity_mag,inhibit_scale):
#         self.excite_radius=excite_radius
#         self.N=N  
#         self.num_links=num_links
#         self.activity_mag=activity_mag
#         self.inhibit_scale=inhibit_scale
        
#     def inhibitions(self,id):
#         ''' each nueuron inhibits all other nueurons but itself'''
#         return np.delete(np.arange(self.N),self.excitations(id))

#     def excitations(self,id):
#         '''each neuron excites itself and num_links neurons left and right with wraparound connections'''
#         excite=[]
#         for i in range(-self.excite_radius,self.excite_radius+1):
#             excite.append((id + i) % self.N)
#         return np.array(excite)

#     def activation(self,id):
#         '''each neuron excites itself and num_links neurons left and right with wraparound connections'''
#         excite=[]
#         for i in range(-self.num_links,self.num_links+1):
#             excite.append((int(id) + i) % self.N)
#         return np.array(excite)

#     def full_weights(self,radius):
#         x=np.arange(-radius,radius+1)
#         return 1/(np.std(x) * np.sqrt(2 * np.pi)) * np.exp( - (x - np.mean(x))**2 / (2 * np.std(x)**2))  

#     def fractional_weights(self,non_zero_prev_weights,activeNeuron):
#         frac=activeNeuron%1
#         if frac == 0:
#             return non_zero_prev_weights
#         else: 
#             inv_frac=1-frac
#             frac_weights=np.zeros((len(non_zero_prev_weights)))
#             frac_weights[0]=non_zero_prev_weights[0]*inv_frac
#             for i in range(1,len(non_zero_prev_weights)):
#                 frac_weights[i]=non_zero_prev_weights[i-1]*frac + non_zero_prev_weights[i]*inv_frac
#             return frac_weights

#     def update_weights_dynamics(self,prev_weights,activeNeuron,moreResults=None,Landmark=None):

#         delta=(int(activeNeuron)-np.argmax(prev_weights))%self.N

#         indexes,non_zero_weights,non_zero_weights_shifted, inhbit_val, non_zero_weights_shifted_land=np.arange(self.N),np.zeros(self.N),np.zeros(self.N),0, np.zeros(self.N)
#         # shifted_indexes=self.neuron_update(prev_weights)

#         '''copied and shifted activity'''
#         non_zero_idxs=indexes[prev_weights>0] # indexes of non zero prev_weights

#         if len(prev_weights[non_zero_idxs])==0:
#             prev_weights[self.activation(activeNeuron)]=self.full_weights(self.num_links)
#             non_zero_idxs=indexes[prev_weights>0] # indexes of non zero prev_weights
        
#         non_zero_weights[non_zero_idxs]=prev_weights[non_zero_idxs] 
        
#         non_zero_weights_shifted[(non_zero_idxs+delta)%self.N]=self.fractional_weights(prev_weights[non_zero_idxs],activeNeuron) #non zero weights shifted by delta
        
#         '''inhibition'''
#         for i in range(len(non_zero_weights_shifted)):
#             inhbit_val+=non_zero_weights_shifted[i]*self.inhibit_scale
        
#         '''excitation'''
#         excitations_store=np.zeros((len(non_zero_idxs),self.N))
#         excitation_array,excite=np.zeros(self.N),np.zeros(self.N)
#         for i in range(len(non_zero_idxs)):
#             excitation_array[self.excitations(non_zero_idxs[i])]=self.full_weights(self.excite_radius)*prev_weights[non_zero_idxs[i]]
#             excitations_store[i,:]=excitation_array
#             excite[self.excitations(non_zero_idxs[i])]+=self.full_weights(self.excite_radius)*prev_weights[non_zero_idxs[i]]

        
#         if Landmark is not None:
#             delta_land=(int(activeNeuron)-np.argmax(prev_weights))%self.N
#             non_zero_weights_shifted_land[(non_zero_idxs+delta_land)%self.N]=self.fractional_weights(prev_weights[non_zero_idxs],Landmark) #non zero weights shifted by delta
#             prev_weights+=(non_zero_weights_shifted+non_zero_weights_shifted_land+excite-inhbit_val)
#         else:
#             prev_weights+=(non_zero_weights_shifted+excite-inhbit_val)
            
#         if moreResults==True:
#            return prev_weights/np.linalg.norm(prev_weights), non_zero_weights_shifted, excitations_store, inhbit_val
#         else:  
#            return prev_weights/np.linalg.norm(prev_weights)

def activityDecoding(prev_weights,radius,N):
    '''Isolating activity at a radius around the peak to decode position'''
    # if np.argmax(prev_weights)==0:
    #     return 0
    # else:
    neurons=np.arange(N)
    peak=np.argmax(prev_weights) 
    local_activity=np.zeros(N)
    local_activity_idx=[]
    for i in range(-radius,radius+1):
        local_activity_idx.append((peak + i) % N)
    local_activity[local_activity_idx]=prev_weights[local_activity_idx]

    x,y=local_activity*np.cos(np.deg2rad(neurons*360/N)), local_activity*np.sin(np.deg2rad(neurons*360/N))
    vect_sum=np.rad2deg(math.atan2(sum(y),sum(x))) % 360
    weighted_sum = N*(vect_sum/360)

    if weighted_sum==N:
        weighted_sum=0

    return weighted_sum


def activityDecodingAngle(prev_weights,radius,N):
    '''Isolating activity at a radius around the peak to decode position'''
    neurons = np.arange(N)
    peak=np.argmax(prev_weights) 
    local_activity=np.zeros(N)
    local_activity_idx=[]
    for i in range(-radius,radius+1):
        local_activity_idx.append((peak + i) % N)
    local_activity[local_activity_idx]=prev_weights[local_activity_idx]

    x,y=local_activity*np.cos(np.deg2rad(neurons*360/N)), local_activity*np.sin(np.deg2rad(neurons*360/N))
    vect_sum=np.rad2deg(math.atan2(sum(y),sum(x))) % 360
    return vect_sum


def multiResolution(val):
    '''Representing up to 2 whole numbers and 3 decimal places: assumes input is a float with a decimal point eg. 0 = 0.0'''
    rounded_val=round(val,3)
    values=str(rounded_val).split('.')
    whole, dec= np.zeros(2), np.zeros(3)
    whole_len, dec_len=len(values[0]), len(values[1])

    #grouping the whole number strings into two
    whole[1]=int(values[0][-1])
    if whole_len==1:
        whole[0]=0
    elif whole_len>2:
        whole[0]=int(values[0][0:-1])
    else:
        whole[0]=int(values[0][0])

    #storing decimal strings based on the length 
    for i in range(dec_len):
        dec[i]=int(values[1][i])

    placeValue=np.concatenate((whole, dec))
    scale = [10,1,0.1,0.01,0.001]
    return placeValue, scale

def imageHistogram(prev_weights,N,val):
      prev_weights[prev_weights<0]=0
      prev_weights/np.linalg.norm(prev_weights)
      height=50
     
      hist1=np.zeros((height,N*2))
      for n in range(N*2):
         if n%2==0:
            coloured=int(np.round(prev_weights[n//2],2)*100)
            if coloured != 0:
               hist1[:coloured, n]=[val]*coloured
      return hist1