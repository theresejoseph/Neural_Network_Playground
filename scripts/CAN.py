from cgi import print_arguments
import math
import numpy as np 

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

    def update_weights_dynamics(self,prev_weights, delta, moreResults=None, cross=None):
        indexes,non_zero_weights,non_zero_weights_shifted, inhbit_val=np.arange(self.N),np.zeros(self.N),np.zeros(self.N),0
        '''crossover'''
        crossover=0
        if np.argmax(prev_weights)+delta >= self.N:
            crossover=(np.argmax(prev_weights)+int(delta))//self.N
        elif np.argmax(prev_weights)+delta < 0:
            crossover=(np.argmax(prev_weights)+int(delta))//self.N

        '''copied and shifted activity'''
        non_zero_idxs=indexes[prev_weights>0] # indexes of non zero prev_weights

        shifted_indexes=(non_zero_idxs+ int(delta)) % self.N

        # if len(prev_weights[non_zero_idxs])==0:
        #     prev_weights[self.activation(0)]=self.full_weights(self.num_links)
        #     non_zero_idxs=indexes[prev_weights>0] # indexes of non zero prev_weights
        
        non_zero_weights[non_zero_idxs]=prev_weights[non_zero_idxs] 
        
        non_zero_weights_shifted[shifted_indexes]=prev_weights[non_zero_idxs]  #non zero weights shifted by delta
        
        intermediate_activity=non_zero_weights_shifted+non_zero_weights
        '''inhibition'''
        for i in range(len(non_zero_weights_shifted)):
            inhbit_val+=non_zero_weights_shifted[i]*self.inhibit_scale
        
        '''excitation'''
        excitations_store=np.zeros((len(shifted_indexes),self.N))
        excitation_array,excite=np.zeros(self.N),np.zeros(self.N)
        for i in range(len(shifted_indexes)):
            excitation_array[self.excitations(shifted_indexes[i])]=self.full_weights(self.excite_radius)*prev_weights[shifted_indexes[i]]
            excitations_store[i,:]=excitation_array
            excite[self.excitations(shifted_indexes[i])]+=self.full_weights(self.excite_radius)*prev_weights[shifted_indexes[i]]

        '''update activity'''
        for k in range(10):
            prev_weights+=(non_zero_weights_shifted+excite-inhbit_val)

        

        if moreResults==True:
           return prev_weights/np.linalg.norm(prev_weights), non_zero_weights, non_zero_weights_shifted, intermediate_activity,[inhbit_val]*self.N, excitations_store
        elif cross==True:  
           return prev_weights/np.linalg.norm(prev_weights) ,crossover
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
        # '''initialise'''
        # if len(prev_weights>0)==0:
        #     prev_weights[self.activation(0)]=self.full_weights(self.num_links)
        #     non_zero_idxs=indexes[prev_weights>0] # indexes of non zero prev_weights    
        
     
        indexes,non_zero_weights,non_zero_weights_shifted, inhbit_val=np.arange(self.N),np.zeros(self.N),np.zeros(self.N),0

        '''copied and shifted activity'''
        non_zero_idxs=indexes[prev_weights>0] # indexes of non zero prev_weights
        non_zero_weights[non_zero_idxs]=prev_weights[non_zero_idxs] 
        non_zero_weights_shifted[(non_zero_idxs+round(delta))%self.N]=prev_weights[non_zero_idxs] #non zero weights shifted by delta
  
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

        

        if abs(delta)<self.excite_radius:
            prev_weights+=(non_zero_weights_shifted+excite-inhbit_val)
            return prev_weights/np.linalg.norm(prev_weights)
        else:  
            return prev_weights


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

