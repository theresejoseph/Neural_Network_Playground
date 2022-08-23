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


def process_state_image(state):
    state=np.dot(state[...,:3], [0.2989, 0.5870, 0.1140])
    state = state.astype(float)
    state /= 255.0
    return state

def generate_state_frame_stack_from_queue(deque):
    frame_stack = np.array(deque)
    # Move stack dimension to the channel dimension (stack, x, y) -> (x, y, stack)
    return np.transpose(frame_stack, (1, 2, 0))

def carRacerExecution(queue):
    train_model = './trial_600.h5' #args.model
    play_episodes = 1#args.episodes

    env = gym.make('CarRacing-v0')
    agent = CarRacingDQNAgent(epsilon=0) # Set epsilon to 0 to ensure all actions are instructed by the agent
    agent.load(train_model)

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
            time.sleep(0.1)

            

            if done:
                print('Episode: {}/{}, Scores(Time Frames): {}, Total Rewards: {:.2}'.format(e+1, play_episodes, time_frame_counter, float(total_reward)))
                break
            time_frame_counter += 1
    env.close()

def plottingPosition(queue):
    global int_x, int_y
    fig = plt.figure(figsize=(13, 4))
    ax0 = fig.add_subplot(1, 2, 1)
    ax3 = fig.add_subplot(1, 2, 2)
    fig.tight_layout()

    curr_x, curr_y=[],[]
    int_x, int_y=0,0
    
    def animate(i):
        global int_x, int_y
        ax0.clear()

        while not queue.empty():
            posX,posY,linVx,linVy= queue.get() 
            if len(curr_x)>0:
                curr_x.append(posX-curr_x[0])
                curr_y.append(posY-curr_y[0])
                
            else:
                curr_x.append(posX)
                curr_y.append(posY)
            
            int_x+=(linVx/50)
            int_y+= (linVy/50)
            print((linVx/50),(linVy/50))
            

        
        ax0.scatter(curr_x[1:],curr_y[1:])
        ax0.set_title('True Car Racer Position')

        ax3.scatter(int_x,int_y)
        

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