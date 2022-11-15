import random
from cv2 import Algorithm
import numpy as np
import sys
sys.path.append('./scripts')
from CAN import attractorNetworkScaling,attractorNetwork2D, attractorNetwork
import CAN as can
from TwoModesofMultiscale import scale_selection, hierarchicalNetwork
from DataHandling import saveOrLoadNp
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import math

'''GA Fitness Functions'''
#1D input velocities comparing movement wihtin undesired scale 
def MultiResolutionTranslation(genome):
    N=100
    num_links=int(genome[0])
    excite=int(genome[1])
    activity_mag=genome[2]
    inhibit_scale=genome[3]

    # data_x=np.concatenate([np.arange(0,0.51,0.01), np.arange(0.51,5.61,0.1), np.arange(5.61,56.61,1), np.arange(56.61,566.61,10), np.arange(566.61,5666.61,100)])
    # data_x=np.concatenate([np.arange(0,5000,100), np.arange(5000,5500,10), np.arange(5500,5550,1), np.arange(5550,5555,0.1), np.arange(5555,5555.5,0.01)])
    # data_y=np.zeros(len(data_x))
    # velocities=np.concatenate([np.array([100]*25), np.array([10]*25), np.array([1]*25), np.array([0.1]*25), np.array([0.01]*25), np.array([0.1]*25),  np.array([1]*25),  np.array([10]*25),  np.array([100]*25)])
    velocities=np.concatenate([np.random.normal(100,10,25),np.random.normal(10,1,25), np.random.normal(1,0.1,25), np.random.normal(0.1,0.01,25), np.random.normal(0.01,0.001,25)])
    scale = [0.01, 0.1, 1, 10, 100]
    fitness=0

    '''initiliase network'''
    net=attractorNetwork(N,num_links,excite, activity_mag,inhibit_scale)
    prev_weights=[np.zeros(N), np.zeros(N), np.zeros(N),np.zeros(N), np.zeros(N)]
    for n in range(len(prev_weights)):
        prev_weights[n][net.activation(0)]=net.full_weights(num_links)
    
    delta_peak=np.zeros((len(velocities),len(scale)))
    split_output=np.zeros((len(velocities),len(scale)))

    for i in range(1,len(velocities)):
        activity_len=[len(np.arange(N)[weights>0]) for weights in prev_weights]
        if 0 in activity_len:
            fitness = -100000
        else: 
            '''encoding mangnitude movement into multiple scales'''
            input_idx=np.argmin(np.abs(np.asarray(scale)-velocities[i]))
            
            input=velocities[i]#translation
            delta = [(input/scale[0]), (input/scale[1]), (input/scale[2]), (input/scale[3]), (input/scale[4])]
            
            '''updating network'''    
            for n in range(len(delta)):
                prev_weights[n][:]= net.update_weights_dynamics(prev_weights[n][:],delta[n])
                prev_weights[n][prev_weights[n][:]<0]=0
                split_output[i,n]=np.argmax(prev_weights[n][:])
            
            '''finding fitness based on activity velocity'''
            delta_peak[i,:]=np.abs(split_output[i,:]-split_output[i-1,:])
            decoded=np.sum(split_output*scale)*np.sign(input) 
            for j in range(len(scale)):
                if j != input_idx:
                    fitness-=np.sum([peak[j] for peak in delta_peak])
                else:
                    fitness+=delta_peak[i,j]
    return fitness

#1D input velocites comparing input output
def MultiResolution1D_Decoding(genome):
    N=100
    num_links=int(genome[0])
    excite=int(genome[1])
    activity_mag=genome[2]
    inhibit_scale=genome[3]

    velocities=np.concatenate([np.array([100]*25), np.array([10]*25), np.array([1]*25), np.array([0.1]*25), np.array([0.01]*25)])

    scale = [0.01, 0.1, 1, 10, 100]
    fitness=0

    '''initiliase network'''
    prev_weights=[np.zeros(N), np.zeros(N), np.zeros(N),np.zeros(N), np.zeros(N)]
    net=attractorNetwork(N,num_links,excite, activity_mag,inhibit_scale)
    for n in range(len(prev_weights)):
        prev_weights[n][net.activation(0)]=net.full_weights(num_links)

    integratedPos,decodedPos=[0],[0]
    for i in range(2,len(velocities)):
        activity_len=[len(np.arange(N)[weights>0]) for weights in prev_weights]
        if 0 in activity_len:
            fitness = -100000
            break
        else: 
            '''encoding mangnitude movement into multiple scales'''
            input=velocities[i]
            delta = [(input/scale[0]), (input/scale[1]), (input/scale[2]), (input/scale[3]), (input/scale[4])]
            
            '''updating network''' 
            split_output=np.zeros((len(delta)))   
            for n in range(len(delta)):
                prev_weights[n][:]= net.update_weights_dynamics(prev_weights[n][:],delta[n])
                prev_weights[n][prev_weights[n][:]<0]=0
                split_output[n]=np.argmax(prev_weights[n][:])
            
            '''finding fitness based on activity velocity'''
            decoded=np.sum(split_output*scale)
            integratedPos.append(integratedPos[-1]+input)
            decodedPos.append(decoded)  

    fitness=np.sum(abs(np.array(integratedPos)-np.array(decodedPos)))*-1
    return fitness 

#2D input positions 
def MultiResolution2D(genome):
    N1=100
    N2=100
    num_links1=int(genome[0])
    excite1=int(genome[1])
    activity_mag1=genome[2]
    inhibit_scale1=genome[3]

    num_links2=int(genome[4])
    excite2=int(genome[5])
    activity_mag2=genome[6]
    inhibit_scale2=genome[7]

    num_links3=int(genome[8])
    excite3=int(genome[9])
    activity_mag3=genome[10]
    inhibit_scale3=genome[11]

    data_x=np.concatenate([ np.arange(0,10.1,0.1), np.arange(10.1,101.1,1), np.arange(101.1,1111.1,10)])
    data_y=np.concatenate([ np.arange(0,10.1,0.1), np.arange(10.1,101.1,1), np.arange(101.1,1111.1,10)])

    # data_x=np.arange(1,100,1)
    # data_y=np.arange(1,100,1)

    scale = [ 0.1, 1, 10]
    fitness=0
    
    '''initiliase network'''
    net=[attractorNetwork2D(N1,N2,num_links1,excite1,activity_mag1,inhibit_scale1),attractorNetwork2D(N1,N2,num_links2,excite2,activity_mag2,inhibit_scale2),attractorNetwork2D(N1,N2,num_links3,excite3,activity_mag3,inhibit_scale3)]
    prev_weights=[net[0].neuron_activation(0,0), net[1].neuron_activation(0,0), net[2].neuron_activation(0,0)]

    delta_peak_rows, delta_peak_cols=np.zeros((len(data_x),len(scale))), np.zeros((len(data_x),len(scale)))
    split_output_row,split_output_col=np.zeros((len(data_x),len(scale))), np.zeros((len(data_x),len(scale)))

    for i in range(1,len(data_x)):
        activity_sum=[np.sum(prev_weights[0][:]), np.sum(prev_weights[1][:]), np.sum(prev_weights[2][:])]
        # print(activity_sum)
        if True in np.isnan(activity_sum):
            fitness = -100000
            break
        else: 
            '''encoding mangnitude movement into multiple scales'''
            x1, x2=data_x[i-1], data_x[i]
            y1, y2= data_y[i-1], data_y[i]

            input_idx=np.argmin(np.abs(np.asarray(scale)-(x2-x1)))

            delta_col = [((x2-x1)/scale[0]), ((x2-x1)/scale[1]), ((x2-x1)/scale[2])]
            delta_row = [((y2-y1)/scale[0]), ((y2-y1)/scale[1]), ((y2-y1)/scale[2])]
     
            '''updating network'''    
            for n in range(len(delta_col)):
                prev_weights[n][:]= net[n].update_weights_dynamics(prev_weights[n][:],delta_row[n],delta_col[n])
                prev_weights[n][prev_weights[n][:]<0]=0
                row_index,col_index=np.unravel_index(np.argmax(prev_weights[n][:]), np.shape(prev_weights[n][:]))
                split_output_row[i,n]=row_index
                split_output_col[i,n]=col_index

            delta_peak_rows[i,:]=np.abs(split_output_row[i,:]-split_output_row[i-1,:])
            delta_peak_cols[i,:]=np.abs(split_output_col[i,:]-split_output_col[i-1,:])

            # decoded_row=np.sum(split_output_row*scale)*np.sign((y2-y1)) 
            # decoded_col=np.sum(split_output_col*scale)*np.sign((x2-x1)) 
            
            for j in range(len(scale)):
                if j != input_idx:
                    fitness-=np.sum([peak[j] for peak in delta_peak_rows])
                    fitness-=np.sum([peak[j] for peak in delta_peak_cols])
                else:
                    fitness+=delta_peak_rows[i,j]
                    fitness+=delta_peak_cols[i,j]
    # print(fitness)
    return fitness

#input to outuput accuracy
def CAN_tuningShiftAccuracy(genome):
    N=100
    num_links=int(genome[0])
    excite=int(genome[1])
    activity_mag=genome[2]
    inhibit_scale=genome[3]
    iterations=int(genome[4])
    
    prev_weights=np.zeros(N)
    net=attractorNetwork(N,num_links,excite, activity_mag,inhibit_scale)
    prev_weights[net.activation(0)]=net.full_weights(num_links)
    inputs=np.concatenate([np.linspace(0,0.25,15), np.linspace(0.25,1,15), np.array([4]*5),  np.array([16]*2)])
    outputs=np.zeros(len(inputs))
    peaks=np.zeros(len(inputs))

    for i in range(1,len(inputs)):
        for iter in range(iterations):
            prev_weights=net.update_weights_dynamics(prev_weights,inputs[i])
            prev_weights[prev_weights<0]=0
        
        peaks[i]=can.activityDecoding(prev_weights,3,N)
        outputs[i]=(abs(can.activityDecoding(prev_weights,4,N)-peaks[i-1]))
    
    return (np.sum(abs(outputs-inputs)))*-1
# tuning with wraparound 
def CAN_tuningShiftAccuracywithWraparound(genome):
    #genome parameters 
    N=100
    num_links=int(genome[0]) #int
    excite=int(genome[1]) #int
    activity_mag=genome[2] #uni
    inhibit_scale=genome[3] #uni
    iterations=int(genome[4]) #int
    wrap_iterations= int(genome[5]) #int
    wrap_mag=genome[6] #uni
    wrap_inhi=genome[7] #uni
    
    #initialising network
    scales=[0.25,1,4,16,100,10000]
    prev_weights=np.zeros(N)
    net=attractorNetwork(N,num_links,excite, activity_mag,inhibit_scale)
    prev_weights[net.activation(0)]=net.full_weights(num_links)
    inputs=np.concatenate([np.array([16]*110),np.linspace(100,300,100)])
    integratedPos=[0]
    decodedPos=[0]

    for i in range(1,len(inputs)):
        hierarchicalNetwork(integratedPos,decodedPos,net,inputs[i],N,iterations,wrap_iterations, wrap_mag, wrap_inhi)

    
    return (np.sum(abs(np.array(decodedPos)-np.array(integratedPos))))*-1



'''Implementation'''
class GeneticAlgorithm:
    def __init__(self,num_gens,population_size,filename,fitnessFunc, ranges,mutate_amount):
        self.num_gens=num_gens
        self.population_size=population_size
        self.filename=filename
        self.fitnessFunc=fitnessFunc
        self.ranges=ranges
        self.mutate_amount=mutate_amount

    def rand(self,range_idx,intOruni):
        # return random integer or float from uniform distribution  within the allowed range of each parameter 
        if intOruni=='int':
            return random.randint(self.ranges[range_idx][0],self.ranges[range_idx][1])
        elif intOruni=='uni':
            return random.uniform(self.ranges[range_idx][0],self.ranges[range_idx][1])

    def initlisePopulation(self,numRandGenomes):
        population=[]
        for i in range(numRandGenomes):
            # genome=[self.rand(0,'int'), self.rand(1,'int'),self.rand(2,'uni'),self.rand(3,'uni'), self.rand(4,'int'), self.rand(5,'int'),self.rand(6,'uni'),self.rand(7,'uni')]
            genome=[self.rand(0,'int'), self.rand(1,'int'),self.rand(2,'uni'),self.rand(3,'uni'), self.rand(4,'int')]
            population.append(genome)
        return population 

    def mutate(self, genome):
        # if random value is greater than 1-probabilty, then mutate the gene
        # if no genes are mutated then require one (pick randomly)
        # amount of mutation = value + gaussian (with varience)
        mutate_prob=np.array([random.random() for i in range(len(genome))])
        mutate_indexs=np.argwhere(mutate_prob<=0.2)
        
        new_genome=np.array(genome)
        new_genome[mutate_indexs]+=self.mutate_amount[mutate_indexs]
        return new_genome 
    
    def checkMutation(self,genome):
        # check mutated genome exists within the specified range 
        g=self.mutate(genome)
        while([self.ranges[i][0] <= g[i] <= self.ranges[i][1] for i in range(len(genome))]!=[True]*len(genome)):
            g=self.mutate(genome)
        return g

    def sortByFitness(self,population,topK):
        # fitness for each genome 
        # sort genomes by fitness
        fitness=np.zeros(len(population))
        for i in range (len(population)):
            fitness[i]=self.fitnessFunc(population[i])
        idxs=np.argsort(fitness)[::-1]
        return fitness[idxs[:topK]],idxs[:topK]
    
    def selection(self,population):
        # parent = take the best 5 (add to new population)
        # for every parent  make 3 children and add to new population 
        num_parents=self.population_size//4
        num_children_perParent=(self.population_size-num_parents)//num_parents

        fitnesses,indexes=self.sortByFitness(population,num_parents)
        print('Finsihed Checking Fitness of old population, now mutating parents to make a new generation')
        # new_population=[population[idx] for idx in indexes] #parents are added to the new population 
        '''Add 5 random genomes into the population'''
        new_population=self.initlisePopulation(num_parents)
        
        '''Make 15 Children from the fittest parents'''
        for i in range(num_parents):
            for j in range(num_children_perParent):
                new_population.append(self.checkMutation(population[indexes[i]]))
        return new_population
    
    def implimentGA(self):
        population=self.initlisePopulation(self.population_size)
        print(population)
        fitnesses=np.zeros((self.population_size,1))
        order_population=np.zeros((self.num_gens,self.population_size,len(population[0])+1))

        '''iterate through generations'''
        for i in range(self.num_gens):
            print(f'Current Generation {i}')
            population=np.array(self.selection(population))
            print('Finsihed making new populaiton  through mutation, now evaluting fitness and sorting')
            fitnesses,indexes=self.sortByFitness(population,self.population_size)
            order_population[i,:,:] = np.hstack((np.array(population[indexes]), fitnesses[:,None]))

            current_fitnesses=[max(fit) for fit in np.array(order_population)[:,:,-1]]
            stop_val=5
            if i>=stop_val and current_fitnesses[-stop_val]==current_fitnesses[-1]*stop_val:
                break
            print(fitnesses)

        with open(self.filename, 'wb') as f:
            np.save(f, np.array(order_population))

def runGA1D(plot=False):
    #[num_links, excitation width, activity magnitude,inhibition scale]
    filename=f'./results/GA_MultiScale/20_gens_20pop_wraparound.npy'
    # mutate_amount=np.array([int(np.random.normal(0,1)), int(np.random.normal(0,1)), np.random.normal(0,0.05), np.random.normal(0,0.05), int(np.random.normal(0,1)), int(np.random.normal(0,1)), np.random.normal(0,0.05), np.random.normal(0,0.05)])
    # ranges = [[1,10],[1,10],[0.1,4],[0,0.1],[1,10],[1,10],[0.1,4],[0,0.1]]

    mutate_amount=np.array([int(np.random.normal(0,1)), int(np.random.normal(0,1)), np.random.normal(0,0.05), np.random.normal(0,0.05), int(np.random.normal(0,1))])
    ranges = [[1,10],[1,10],[0.1,4],[0,0.1],[1,10]]
    fitnessFunc=CAN_tuningShiftAccuracy
    num_gens=20
    population_size=32

    if plot==True:
        with open(filename, 'rb') as f:
            data = np.load(f)
        plt.plot([max(fit) for fit in data[:,:,-1]], 'g*-')
        plt.title('Best Fitness over 20 Generation')
        plt.show()
        print(data[:,1,:])
    else:
        GeneticAlgorithm(num_gens,population_size,filename,fitnessFunc,ranges,mutate_amount).implimentGA()


runGA1D(plot=False)
runGA1D(plot=True)

# def decodedPosAfterupdate(weights,input):





'''Test Area'''
#np.[number or neurons , num_links, excitation width, activity magnitude,inhibition scale]

# mutate_amount=np.array([int(np.random.normal(0,5)), int(np.random.normal(0,5)), int(np.random.normal(0,1)), np.random.normal(0,0.1), np.random.normal(0,0.05)])
'''1D'''
def visualiseMultiResolutionTranslation(genome):
    N=int(genome[0])
    num_links=int(genome[1])
    excite=int(genome[2])
    activity_mag=genome[3]
    inhibit_scale=genome[4]

    data_x=np.arange(0,200,1)
    data_y=np.zeros(len(data_x))

    curr_parameter=[0,0]
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


    net=attractorNetworkScaling(N,num_links,excite, activity_mag,inhibit_scale)
    prev_weights_trans=[np.zeros(N), np.zeros(N), np.zeros(N),np.zeros(N), np.zeros(N)]
    for n in range(len(prev_weights_trans)):
        prev_weights_trans[n][net.activation(0)]=net.full_weights(num_links)

    def multiResolutionUpdate(input,prev_weights,net): 
        scale = [0.01, 0.1, 1, 10, 100]
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

            net0=attractorNetworkScaling(N,num_links,excite, activity_mag,inhibit_scale)
            decoded_translation,split_trans=multiResolutionUpdate(translation,prev_weights_trans,net0)
    
            curr_parameter[0]=curr_parameter[0]+translation    
            print(f"{str(i)}  translation {translation} input output {round(curr_parameter[0],3)}  {str(decoded_translation )}  ")
            
            
            axtxt1.text(0,0.5,f"Input Trans: {round(x2,3)}, Shift: {round(translation,4)}, Decoded Trans: {round(decoded_translation,3)}", c='r')
            # axtxt.text(0,0,"Input Rot: " +str(round(rotation,3))+ " " + str(round(decoded_rotation,3)), c='m')
            axtxt1.axis('off')
            axtxt1.text(0,0,"Decoded Position of Each Network: " + str(split_trans), c='r')

            ax10.bar(np.arange(N),prev_weights_trans[0][:],color='aqua')
            ax10.get_xaxis().set_visible(False)
            ax10.spines[['top', 'bottom', 'right']].set_visible(False)

            ax11.bar(np.arange(N),prev_weights_trans[1][:],color='green')
            ax11.get_xaxis().set_visible(False)
            ax11.spines[['top', 'bottom', 'right']].set_visible(False)

            ax12.bar(np.arange(N),prev_weights_trans[2][:],color='blue')
            ax12.get_xaxis().set_visible(False)
            ax12.spines[['top', 'bottom', 'right']].set_visible(False)
        
            ax13.bar(np.arange(N),prev_weights_trans[3][:],color='purple')
            ax13.get_xaxis().set_visible(False)
            ax13.spines[['top', 'bottom', 'right']].set_visible(False)
            
            ax14.bar(np.arange(N),prev_weights_trans[4][:],color='pink')
            ax14.get_xaxis().set_visible(False)
            ax14.spines[['top', 'bottom', 'right']].set_visible(False)

    ani = FuncAnimation(fig, animate, interval=1,frames=len(data_x),repeat=False)
    plt.show()

# fittest=[1.48000000e+02, 1.20000000e+01, 1.00000000e+00, 7.07215044e-02, 4.74091433e-01]
# fittest=[1.67000000e+02, 2.00000000e+00, 3.00000000e+00, 9.23424531e-02, 5.20268520e-01]
# visualiseMultiResolutionTranslation(fittest)
# print(MultiResolutionTranslation(fittest))
# genome=[1,4,1.00221581e-01,1.29876096e-01]
# print(MultiResolution1D_Decoding(genome))

'''2D''' 

#np.[nnum_links, excitation width, activity magnitude,inhibition scale]*3
# mutate_amount=np.array([int(np.random.normal(0,2)), int(np.random.normal(0,1)), np.random.normal(0,0.05), np.random.normal(0,0.005), int(np.random.normal(0,2)), int(np.random.normal(0,1)), np.random.normal(0,0.05), np.random.normal(0,0.005), int(np.random.normal(0,2)), int(np.random.normal(0,1)), np.random.normal(0,0.05), np.random.normal(0,0.005)])
# ranges = [[1,10],[1,10],[0.001,1],[0.0005,0.005],[1,10],[1,10],[0.001,1],[0.00005,0.005],[1,10],[1,10],[0.001,1],[0.00005,0.005]]
# fitnessFunc=MultiResolution2D
# num_gens=20
# population_size=20
# filename=f'./results/GA_MultiScale/20_gens_2D_1net_20pop_300points_3paramSet.npy'
# '''initiliase'''
# population=[]
# for i in range(population_size):
#     genome=[random.randint(ranges[0][0],ranges[0][1]), random.randint(ranges[1][0],ranges[1][1]), random.uniform(ranges[2][0],ranges[2][1]), random.uniform(ranges[3][0],ranges[3][1]), random.randint(ranges[0][0],ranges[0][1]), random.randint(ranges[1][0],ranges[1][1]), random.uniform(ranges[2][0],ranges[2][1]), random.uniform(ranges[3][0],ranges[3][1]), random.randint(ranges[0][0],ranges[0][1]), random.randint(ranges[1][0],ranges[1][1]), random.uniform(ranges[2][0],ranges[2][1]), random.uniform(ranges[3][0],ranges[3][1])]
#     population.append(genome)
# GeneticAlgorithm(population,num_gens,population_size,filename,fitnessFunc,ranges,mutate_amount)


def visualiseMultiResolutionTranslation2D(genome):
    global error
    '''initlising network and animate figures'''
    fig = plt.figure(figsize=(8, 4))
    ax10 = plt.subplot2grid(shape=(6, 3), loc=(0, 0), rowspan=5,colspan=1)
    ax11 = plt.subplot2grid(shape=(6, 3), loc=(0, 1), rowspan=5,colspan=1)
    ax12 = plt.subplot2grid(shape=(6, 3), loc=(0, 2), rowspan=5,colspan=1)
    axtxt1 = plt.subplot2grid(shape=(6,3), loc=(5, 0), rowspan=1,colspan=3)
    fig.tight_layout()

    N1=100
    N2=100
    num_links1=int(genome[0])
    excite1=int(genome[1])
    activity_mag1=genome[2]
    inhibit_scale1=genome[3]

    num_links2=int(genome[4])
    excite2=int(genome[5])
    activity_mag2=genome[6]
    inhibit_scale2=genome[7]

    num_links3=int(genome[8])
    excite3=int(genome[9])
    activity_mag3=genome[10]
    inhibit_scale3=genome[11]

    data_x=np.concatenate([ np.arange(0,10.1,0.1), np.arange(10.1,101.1,1), np.arange(101.1,1111.1,10)])
    data_y=np.concatenate([ np.arange(0,10.1,0.1), np.arange(10.1,101.1,1), np.arange(101.1,1111.1,10)])

    # data_x=np.arange(1,1000,1)
    # data_y=np.arange(1,1000,1)

    scale = [ 0.1, 1, 10]
    error=0
    
    '''initiliase network'''
    net=[attractorNetwork2D(N1,N2,num_links1,excite1,activity_mag1,inhibit_scale1),attractorNetwork2D(N1,N2,num_links2,excite2,activity_mag2,inhibit_scale2),attractorNetwork2D(N1,N2,num_links3,excite3,activity_mag3,inhibit_scale3)]
    prev_weights=[net[0].neuron_activation(0,0), net[1].neuron_activation(0,0), net[2].neuron_activation(0,0)]

    delta_peak_rows, delta_peak_cols=np.zeros((len(data_x),len(scale))), np.zeros((len(data_x),len(scale)))
    split_output_row,split_output_col=np.zeros((len(data_x),len(scale))), np.zeros((len(data_x),len(scale)))

    def animate(i):
        global error
        i=i+2
        ax10.clear(),ax11.clear(),ax12.clear(), axtxt1.clear()
    
        '''encoding mangnitude movement into multiple scales'''
        x1, x2=data_x[i-1], data_x[i]
        y1, y2= data_y[i-1], data_y[i]

        input_idx=np.argmin(np.abs(np.asarray(scale)-(x2-x1)))

        delta_col = [((x2-x1)/scale[0]), ((x2-x1)/scale[1]), ((x2-x1)/scale[2])]
        delta_row = [((y2-y1)/scale[0]), ((y2-y1)/scale[1]), ((y2-y1)/scale[2])]
        
        '''updating network'''    
        for n in range(len(delta_col)):
            prev_weights[n][:]= net[n].update_weights_dynamics(prev_weights[n][:],delta_row[n],delta_col[n])
            prev_weights[n][prev_weights[n][:]<0]=0
            row_index,col_index=np.unravel_index(np.argmax(prev_weights[n][:]), np.shape(prev_weights[n][:]))
            split_output_row[i,n]=row_index
            split_output_col[i,n]=col_index

        delta_peak_rows[i,:]=np.abs(split_output_row[i,:]-split_output_row[i-1,:])
        delta_peak_cols[i,:]=np.abs(split_output_col[i,:]-split_output_col[i-1,:])

        decoded_row=np.sum(split_output_row*scale)*np.sign((y2-y1)) 
        decoded_col=np.sum(split_output_col*scale)*np.sign((x2-x1)) 
        
        for j in range(len(scale)):
            if j != input_idx:
                error+=np.sum([peak[j] for peak in delta_peak_rows])
                error+=np.sum([peak[j] for peak in delta_peak_cols])
            else:
                error-=delta_peak_rows[i,j]
                error-=delta_peak_cols[i,j]
        
        
        ax10.set_title(str(scale[0])+" Scale",fontsize=9)
        ax11.set_title(str(scale[1])+" Scale",fontsize=9)
        ax12.set_title(str(scale[2])+" Scale",fontsize=9)
        axtxt1.text(0,1,f"Input Trans: {round(x2,3), round(y2,3)}, Shift: {round(x2-x1,2)},{round(y2-y1,2)}, Decoded Trans: {round(decoded_col), round(decoded_row)}", c='r')
        # axtxt.text(0,0,"Input Rot: " +str(round(rotation,3))+ " " + str(round(decoded_rotation,3)), c='m')
        axtxt1.axis('off')
        axtxt1.text(0,0,f"Decoded Position of Each Network: {split_output_col[i]}, {split_output_row[i]}", c='r')

        ax10.imshow(prev_weights[0][:])
        ax10.invert_yaxis()

        ax11.imshow(prev_weights[1][:])
        ax11.invert_yaxis()

        ax12.imshow(prev_weights[2][:])
        ax12.invert_yaxis()
        
    # return error
    ani = FuncAnimation(fig, animate, interval=100,frames=len(data_x)-2,repeat=False)
    plt.show()


'''2D'''
# fittest=[7.60000000e+01,9.70000000e+01,5.00000000e+00,1.36576574e-01, 5.19946747e-03,1.20000000e+01]
# fittest2=[8.40000000e+01,9.80000000e+01,2.00000000e+00,3.76026794e-02,8.27480338e-03,-4.00000000e+01]
# fittest3=[ 7.40000000e+01,7.80000000e+01,2.00000000e+00,1.07922584e-01,7.02904729e-03,-8.80000000e+01]
# fittest=[5.00000000e+00,3.00000000e+00,2.23220121e-02,3.62736020e-03,1.10000000e+01,3.00000000e+00,2.39987795e-02,5.51246640e-04,4.00000000e+00,2.00000000e+00,2.73989154e-01,2.03594051e-03,-4.34000000e+02]
# visualiseMultiResolutionTranslation2D(fittest)
