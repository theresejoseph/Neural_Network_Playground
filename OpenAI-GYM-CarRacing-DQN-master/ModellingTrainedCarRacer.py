import time 
import numpy as np 
import math

from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import matplotlib.widgets as wig
from matplotlib.patches import Rectangle 


from multiprocessing.dummy import freeze_support
import multiprocessing

import argparse
import gym
from collections import deque
from CarRacingDQNAgent import CarRacingDQNAgent

import sys
sys.path.append('./scripts')
from CAN import attractorNetwork2D



def carRacerExecution(queue):
    train_model = './OpenAI-GYM-CarRacing-DQN-master/trial_600.h5' #args.model
    play_episodes = 1#args.episodes

    env = gym.make('CarRacing-v0')
    agent = CarRacingDQNAgent(epsilon=0) # Set epsilon to 0 to ensure all actions are instructed by the agent
    agent.load(train_model)

    def process_state_image(state):
        state=np.dot(state[...,:3], [0.2989, 0.5870, 0.1140])
        state = state.astype(float)
        state /= 255.0
        return state

    def generate_state_frame_stack_from_queue(deque):
        frame_stack = np.array(deque)
        # Move stack dimension to the channel dimension (stack, x, y) -> (x, y, stack)
        return np.transpose(frame_stack, (1, 2, 0))


    for e in range(play_episodes):
        init_state = env.reset()
        init_state = process_state_image(init_state)

        total_reward = 0
        punishment_counter = 0
        state_frame_stack_queue = deque([init_state]*agent.frame_stack_num, maxlen=agent.frame_stack_num)
        time_frame_counter = 1
        
        while True:
            env.render()

            current_state_frame_stack = generate_state_frame_stack_from_queue(state_frame_stack_queue)
            action = agent.act(current_state_frame_stack)
            next_state, reward, done, info = env.step(action)

            total_reward += reward

            next_state = process_state_image(next_state)
            state_frame_stack_queue.append(next_state)

            posX= env.car.hull.position[0]
            posY= env.car.hull.position[1]
            linVx,linVy=env.car.hull.linearVelocity[0],env.car.hull.linearVelocity[1]
            # angV=env.car.hull.angularVelocity

            queue.put((posX,posY,linVx,linVy))
            time.sleep(0.5)

            

            if done:
                print('Episode: {}/{}, Scores(Time Frames): {}, Total Rewards: {:.2}'.format(e+1, play_episodes, time_frame_counter, float(total_reward)))
                break
            time_frame_counter += 1
    env.close()

def plottingPosition(queue):
    global int_x, int_y
    fig = plt.figure(figsize=(13, 4))
    fig_rows,fig_cols=2,4
    ax0 = plt.subplot2grid(shape=(fig_rows, fig_cols), loc=(0, 0), rowspan=2,colspan=1)
    ax11 = plt.subplot2grid(shape=(fig_rows, fig_cols), loc=(0, 1))
    ax12 = plt.subplot2grid(shape=(fig_rows, fig_cols), loc=(0, 2))
    ax13 = plt.subplot2grid(shape=(fig_rows, fig_cols), loc=(0, 3))

    ax21 = plt.subplot2grid(shape=(fig_rows, fig_cols), loc=(0, 1))
    ax22 = plt.subplot2grid(shape=(fig_rows, fig_cols), loc=(0, 2))
    ax23 = plt.subplot2grid(shape=(fig_rows, fig_cols), loc=(0, 3))


    # ax0 = fig.add_subplot(fig_rows, fig_cols, 1)
    # ax1 = fig.add_subplot(fig_rows, fig_cols, 2)
    # ax2 = fig.add_subplot(fig_rows, fig_cols, 3)
    # ax3 = fig.add_subplot(fig_rows, fig_cols, 4)
    # ax4 = fig.add_subplot(1, fig_cols, 5)
    fig.tight_layout()

    curr_x, curr_y=[],[]
    int_x, int_y=0,0
    '''initiliase network'''
    scale=[0.01,0.1,1]
    genome=[7.40000000e+01,2,2.00000000e+00,1.07922584e-01,7.02904729e-03,-8.80000000e+01]
    N1=int(genome[0])
    N2=int(genome[0])
    num_links=int(genome[1])
    excite=int(genome[2])
    activity_mag=genome[3]
    inhibit_scale=genome[4]

    net=attractorNetwork2D(N1,N2,num_links,excite,activity_mag,inhibit_scale)
    prev_weights=[net.neuron_activation(0,0), net.neuron_activation(0,0), net.neuron_activation(0,0)]
    
    def animate(i):
        global int_x, int_y
        #ax1.clear(),ax2.clear(),ax3.clear()
        '''True Position and Linear Velocities'''
        while not queue.empty():
            posX,posY,linVx,linVy= queue.get() 
            if len(curr_x)>0:
                curr_x.append(posX-curr_x[0])
                curr_y.append(posY-curr_y[0]) 
            else:
                curr_x.append(posX)
                curr_y.append(posY)
            
            linVx_adjust,linVy_adjust=linVx/50,linVy/50
            int_x+=(linVx_adjust)
            int_y+= (linVy_adjust)
            # print((linVx/50),(linVy/50))

            '''encoding mangnitude movement into multiple scales'''
            delta_col = [(linVx_adjust/scale[0]), (linVx_adjust/scale[1]), (linVx_adjust/scale[2])]
            delta_row = [(linVy_adjust/scale[0]), (linVy_adjust/scale[1]), (linVy_adjust/scale[2])]
            
            '''updating network'''  
            row_index,col_index=np.zeros(len(delta_col)),np.zeros(len(delta_col))
            for n in range(len(delta_col)):
                prev_weights[n][:]= net.update_weights_dynamics(prev_weights[n][:],delta_row[n],delta_col[n])
                prev_weights[n][prev_weights[n][:]<0]=0
                row_index[n],col_index[n]=np.unravel_index(np.argmax(prev_weights[n][:]), np.shape(prev_weights[n][:]))
                
            '''Plotting'''
            ax0.clear(),ax11.clear(),ax12.clear(),ax13.clear()#,ax21.clear(),ax22.clear(),ax23.clear()
            ax0.scatter(curr_x[1:],curr_y[1:],s=1)
            ax0.set_title('True Car Racer Position')
            ax0.set_xlabel(f"Input: {round(linVx_adjust,4)}, {round(linVy_adjust,4)} Position of Each Network: {row_index}, {col_index}", c='r')

            ax21.scatter(col_index[0],row_index[0],s=1,color='r')
            ax22.scatter(col_index[1],row_index[1],s=1,color='g')
            ax23.scatter(col_index[2],row_index[2],s=1,color='b')
            ax21.set_xlim([0,N1]),ax21.set_ylim([0,N2]),ax22.set_xlim([0,N1]),ax22.set_ylim([0,N2]),ax23.set_xlim([0,N1]),ax23.set_ylim([0,N2])
            
            ax11.imshow(prev_weights[0][:])
            ax12.imshow(prev_weights[1][:])
            ax13.imshow(prev_weights[2][:])
            ax11.invert_yaxis(),ax12.invert_yaxis(),ax13.invert_yaxis()

            ax1.set_title(str(scale[0])+" Scale",fontsize=9)
            ax2.set_title(str(scale[1])+" Scale",fontsize=9)
            ax3.set_title(str(scale[2])+" Scale",fontsize=9)

        

    ani = FuncAnimation(fig, animate, interval=1)
    plt.show()



if __name__=="__main__":
    freeze_support()
    queue = multiprocessing.Queue()
    process_1 = multiprocessing.Process(target=carRacerExecution, args=(queue,))
    process_2 = multiprocessing.Process(target=plottingPosition, args=(queue,))
    process_1.start()
    process_2.start()
    process_1.join()
    process_2.join()