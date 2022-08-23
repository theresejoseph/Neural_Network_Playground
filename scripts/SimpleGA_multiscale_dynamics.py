import random
import numpy as np
from CAN import attractorNetworkScaling,attractorNetwork2D
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import math


def MultiResolutionTranslation(genome):
    N=int(genome[0])
    num_links=int(genome[1])
    excite=int(genome[2])
    activity_mag=genome[3]
    inhibit_scale=genome[4]

    data_x=np.concatenate([  np.arange(0,0.51,0.01), np.arange(0.51,5.61,0.1), np.arange(5.61,56.61,1), np.arange(56.61,566.61,10), np.arange(566.61,5666.61,100)])
    data_y=np.zeros(len(data_x))

    scale = [0.01, 0.1, 1, 10, 100]
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

            input_idx=np.argmin(np.abs(np.asarray(scale)-(x2-x1)))
            
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

def MultiResolution2D(genome):
    N1=int(genome[0])
    N2=int(genome[1])
    excite=int(genome[2])
    activity_mag=genome[3]
    inhibit_scale=genome[4]

    # data_x=np.concatenate([ np.arange(0,5.1,0.1), np.arange(5.1,56.1,1), np.arange(55,566.1,10)])
    # data_y=np.concatenate([ np.arange(0,5.1,0.1), np.arange(5.1,56.1,1), np.arange(55,566.1,10)])

    data_x=np.arange(1,100,1)
    data_y=np.arange(1,100,1)

    scale = [ 0.1, 1, 10]
    fitness=0
    
    '''initiliase network'''
    net=attractorNetwork2D(N1,N2,excite,activity_mag,inhibit_scale)
    prev_weights=[net.excitations(0,0), net.excitations(0,0), net.excitations(0,0)]

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
                prev_weights[n][:]= net.update_weights_dynamics(prev_weights[n][:],delta_row[n],delta_col[n])
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

def mutate(genome,mutate_amount):
    # if random value is greater than 1-probabilty, then mutate the gene
    # if no genes are mutated then require one (pick randomly)
    # amount of mutation = value + gaussian (with varience)
    mutate_prob=np.array([random.random() for i in range(5)])
    mutate_indexs=np.argwhere(mutate_prob<=0.2)
    
    new_genome=np.array(genome)
    new_genome[mutate_indexs]+=mutate_amount[mutate_indexs]
    return new_genome 

def checkMutation(genome,ranges, mutate_amount):
    g=mutate(genome,mutate_amount)
    while([ranges[i][0] <= g[i] <= ranges[i][1] for i in range(5)]!=[True]*5):
        g=mutate(genome,mutate_amount)
    return g

def sortByFitness(population,topK,fitnessFunc):
    # fitness for each genome 
    # sort genomes by fitness
    fitness=np.zeros(len(population))
    for i in range (len(population)):
        fitness[i]=fitnessFunc(population[i])
    idxs=np.argsort(fitness)[::-1]

    return fitness[idxs[:topK]],idxs[:topK]

def selection(population_size,population,fitnessFunc,ranges, mutate_amount):
    # parent = take the best 5 (add to new population)
    # for every parent  make 3 children and add to new population 
    num_parents=population_size//4
    num_children_perParent=(population_size-num_parents)//num_parents

    fitnesses,indexes=sortByFitness(population,num_parents,fitnessFunc)
    print('Finsihed Checking Fitness of old population, now mutating parents to make a new generation')
    
    '''Add 5 random genomes into the population'''
    new_population=[]
    for i in range(num_parents):
        genome=[random.randint(ranges[0][0],ranges[0][1]), random.randint(ranges[1][0],ranges[1][1]), random.randint(ranges[2][0],ranges[2][1]), (random.random()*0.9)+0.1, (random.random()*0.01)+0.0005]
        new_population.append(genome)
    '''Make 15 Children from the fittest parents'''
    for i in range(num_parents):
        for j in range(num_children_perParent):
            new_population.append(checkMutation(population[indexes[i]],ranges, mutate_amount))

    return new_population

def GeneticAlgorithm(num_gens,population_size,filename,fitnessFunc,ranges,mutate_amount):
    '''initiliase'''
    population=[]
    for i in range(population_size):
        genome=[random.randint(ranges[0][0],ranges[0][1]), random.randint(ranges[1][0],ranges[1][1]), random.randint(ranges[2][0],ranges[2][1]), (random.random()*0.9)+0.1, (random.random()*0.01)+0.0005]
        population.append(genome)
    print(population)
    fitnesses=np.zeros((population_size,1))
    order_population=np.zeros((num_gens,population_size,6))

    '''iterate through generations'''
    for i in range(num_gens):
        population=np.array(selection(population_size,population, fitnessFunc,ranges, mutate_amount))
        print('Finsihed making new populaiton  through mutation, now evaluting fitness and sorting')
        fitnesses,indexes=sortByFitness(population,population_size,fitnessFunc)
        order_population[i,:,:] = np.hstack((np.array(population[indexes]), fitnesses[:,None]))
        if i>5 and [max(fit) for fit in order_population[:,:,5]][-5]==[max(fit) for fit in order_population[:,:,5]][-1]*5:
            break
        print(fitnesses)

    with open(filename, 'wb') as f:
        np.save(f, np.array(order_population))

'''Test Area'''
#np.[number or neurons , num_links, excitation width, activity magnitude,inhibition scale]
# mutate_amount=np.array([int(np.random.normal(0,5)), int(np.random.normal(0,5)), int(np.random.normal(0,1)), np.random.normal(0,0.1), np.random.normal(0,0.05)])
mutate_amount=np.array([int(np.random.normal(0,10)), int(np.random.normal(0,10)), int(np.random.normal(0,1)), np.random.normal(0,0.05), np.random.normal(0,0.005)])
ranges = [[50,100],[50,100],[1,10],[0.01,1],[0.0005,0.01]]
fitnessFunc=MultiResolution2D
num_gens=20
population_size=32
filename=f'./results/GA_MultiScale/20_gens_2D_1net_32pop_100data.npy'
GeneticAlgorithm(num_gens,population_size,filename,fitnessFunc,ranges,mutate_amount)

# with open(filename, 'rb') as f:
#     data = np.load(f)
# plt.plot([max(fit) for fit in data[:,:,5]])
# plt.title('Best Fitness over 30 Generation')
# plt.show()
# print(data)


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

def visualiseMultiResolutionTranslation2D(genome):
    global error
    '''initlising network and animate figures'''
    fig = plt.figure(figsize=(8, 4))
    ax10 = plt.subplot2grid(shape=(6, 3), loc=(0, 0), rowspan=5,colspan=1)
    ax11 = plt.subplot2grid(shape=(6, 3), loc=(0, 1), rowspan=5,colspan=1)
    ax12 = plt.subplot2grid(shape=(6, 3), loc=(0, 2), rowspan=5,colspan=1)
    axtxt1 = plt.subplot2grid(shape=(6,3), loc=(5, 0), rowspan=1,colspan=3)
    fig.tight_layout()

    N1=int(genome[0])
    N2=int(genome[1])
    excite=int(genome[2])
    activity_mag=genome[3]
    inhibit_scale=genome[4]

    # data_x=np.concatenate([ np.arange(0,5.1,0.1), np.arange(5.1,56.1,1), np.arange(55,566.1,10)])
    # data_y=np.concatenate([ np.arange(0,5.1,0.1), np.arange(5.1,56.1,1), np.arange(55,566.1,10)])

    data_x=np.arange(1,1000,1)
    data_y=np.arange(1,1000,1)

    scale = [ 0.1, 1, 10]
    error=0
    
    '''initiliase network'''
    net=attractorNetwork2D(N1,N2,excite,activity_mag,inhibit_scale)
    prev_weights=[net.excitations(0,0), net.excitations(0,0), net.excitations(0,0)]

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
            prev_weights[n][:]= net.update_weights_dynamics(prev_weights[n][:],delta_row[n],delta_col[n])
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


'''1D'''
# fittest=[1.48000000e+02, 1.20000000e+01, 1.00000000e+00, 7.07215044e-02, 4.74091433e-01]
# fittest=[1.67000000e+02, 2.00000000e+00, 3.00000000e+00, 9.23424531e-02, 5.20268520e-01]
# visualiseMultiResolutionTranslation(fittest)
# print(MultiResolutionTranslation(fittest))

'''2D'''
# fittest=[7.60000000e+01,9.70000000e+01,5.00000000e+00,1.36576574e-01, 5.19946747e-03,1.20000000e+01]
fittest2=[8.40000000e+01,9.80000000e+01,2.00000000e+00,3.76026794e-02,8.27480338e-03,-4.00000000e+01]
# visualiseMultiResolutionTranslation2D(fittest2)