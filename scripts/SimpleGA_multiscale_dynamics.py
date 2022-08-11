# represent genome 
#np.[number or neurons , num_links, excitation width, activity magnitude,inhibition scale]

import random
import numpy as np
from CAN import attractorNetworkScaling
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import math


def MultiResolutionTranslation(genome):
    N=int(genome[0])
    num_links=int(genome[1])
    excite=int(genome[2])
    activity_mag=genome[3]
    inhibit_scale=genome[4]

    data_x=np.arange(0,200,1)
    data_y=np.zeros(len(data_x))

    scale = [0.01, 0.1, 1, 10, 100]
    input_idx=scale.index(1)
    fitness=0

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
            fitness = -100000
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
            
            '''finding fitness based on activity velocity'''
            delta_peak[i,:]=np.abs(split_output[i,:]-split_output[i-1,:])
            decoded=np.sum(split_output*scale)*np.sign(input) 
            for j in range(len(scale)):
                if j != input_idx:
                    fitness-=np.sum([peak[j] for peak in delta_peak])
                else:
                    fitness+=delta_peak[i,j]
    return fitness

def mutate(genome):
    # if random value is greater than 1-probabilty, then mutate the gene
    # if no genes are mutated then require one (pick randomly)
    # amount of mutation = value + gaussian (with varience)
    mutate_prob=np.array([random.random() for i in range(5)])
    mutate_indexs=np.argwhere(mutate_prob<=0.2)
    mutate_amount=np.array([int(np.random.normal(0,10)), int(np.random.normal(0,5)), int(np.random.normal(0,5)), np.random.normal(0,0.05), np.random.normal(0,0.05)])
    
    new_genome=np.array(genome)
    new_genome[mutate_indexs]+=mutate_amount[mutate_indexs]
    return new_genome 

def checkMutation(genome):
    ranges = [[100,200],[1,50],[1,50],[0.01,1],[0.01,1]]
    g=mutate(genome)
    while([ranges[i][0] <= g[i] <= ranges[i][1] for i in range(5)]!=[True]*5):
        g=mutate(genome)
    return g

def sortByFitness(population,topK):
    # fitness for each genome 
    # sort genomes by fitness
    fitness=np.zeros(len(population))
    for i in range (len(population)):
        fitness[i]=MultiResolutionTranslation(population[i])
    idxs=np.argsort(fitness)[::-1]

    return fitness[idxs[:topK]],idxs[:topK]

def selection(population):
    # parent = take the best 5 (add to new population)
    # for every parent  make 3 children and add to new population 
    fitnesses,indexes=sortByFitness(population,5)
    new_population=[population[idx] for idx in indexes]
    for i in range(5):
        for j in range(3):
            new_population.append(checkMutation(population[indexes[i]]))

    return new_population

def GeneticAlgorithm(num_gens,population_size,filename):
    '''initiliase'''
    ranges = [[100,200],[1,50],[1,50],[0.01,1],[0.01,1]]
    
    population=[]
    for i in range(population_size):
        genome=[random.randint(ranges[0][0],ranges[0][1]), random.randint(ranges[1][0],ranges[1][1]), random.randint(ranges[2][0],ranges[2][1]), (random.random()*0.9)+0.1, (random.random()*0.9)+0.1]
        population.append(genome)

    fitnesses=np.zeros((population_size,1))
    order_population=np.zeros((num_gens,population_size,6))
    for i in range(num_gens):
        population=np.array(selection(population))
        fitnesses,indexes=sortByFitness(population,population_size)
        order_population[i,:,:] = np.hstack((np.array(population[indexes]), fitnesses[:,None]))
        print(fitnesses)

    with open(filename, 'wb') as f:
        np.save(f, np.array(order_population))

num_gens=30
population_size=20
filename=f'./results/GA_MultiScale/30_gens_factor10.npy'
GeneticAlgorithm(num_gens,population_size,filename)

# with open(filename, 'rb') as f:
#     data = np.load(f)


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
# visualiseMultiResolutionTranslation(fittest)
# print(MultiResolutionTranslation(fittest))