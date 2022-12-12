import numpy as np 
import math 

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.widgets as wig
import sys

sys.path.append('/home/therese/Documents/Neural_Network_Playground/scripts')
from CAN import attractorNetwork2D

N1,N2,excite_radius,activity_mag,inhibit_scale=  50, 50, 1, 1, 0.01
net=attractorNetwork2D(N1,N2,excite_radius,activity_mag,inhibit_scale)
prev_weights=net.excitations(0,0)

#simulation
sim_speed=10
iters=400
pause=False
resetDone=True

def plotting_CAN_dynamics():
    fig1 = plt.figure(figsize=(8, 6))
    gs = fig1.add_gridspec(32,24)
    ax0 = plt.subplot(gs[0:24, 0:24])
    plt.subplots_adjust(bottom=0.1)
    # fig1.tight_layout()

    '''Slider for Parameters'''
    button_ax = plt.axes([.05, .05, .05, .04]) # x, y, width, height
    button2_ax = plt.axes([.05, .12, .05, .04]) # x, y, width, height
    exciteax = plt.axes([0.25, 0.15, 0.65, 0.03])
    delta1ax = plt.axes([0.25, 0.1, 0.65, 0.03])
    delta2ax = plt.axes([0.25, 0.05, 0.65, 0.03])
    inhax = plt.axes([0.25, 0.0, 0.65, 0.03])
    # Create a slider from 0.0 to 20.0 in axes axfreq with 3 as initial value
    start_stop=wig.Button(button_ax,label='$\u25B6$')
    reset=wig.Button(button2_ax,'Reset')
    inhibit_scale=wig.Slider(inhax, 'Scale of Inhibition', 0, 0.01, 0.005)
    excite = wig.Slider(exciteax, 'Excitation Radius', 1, 10, 2, valstep=1)
    delta1 = wig.Slider(delta1ax, 'Delta 1', -10, 10, 0)
    delta2 = wig.Slider(delta2ax, 'Delta 2', -10, 10, 0)

    def animate(i):
        global prev_weights
        ax0.clear()
        '''distributed weights with excitations and inhibitions'''
        net=attractorNetwork2D( N1,N2,int(excite.val),activity_mag,inhibit_scale.val)
        prev_weights= net.update_weights_dynamics(prev_weights,delta1.val,delta2.val)
        prev_weights[prev_weights<0]=0

        ax0.imshow(prev_weights)
        ax0.invert_yaxis()
    

    def update(val):
        global prev_weights
        '''distributed weights with excitations and inhibitions'''
        net=attractorNetwork2D(N1,N2,int(excite.val),activity_mag,inhibit_scale.val)
        prev_weights= net.update_weights_dynamics(prev_weights,delta1.val,delta2.val)
        prev_weights[prev_weights<0]=0

    def onClick(event):
        global pause, prev_weights, resetDone 
        (xm,ym),(xM,yM) = start_stop.label.clipbox.get_points()
        if xm < event.x < xM and ym < event.y < yM:
            pause ^= True

        (xn,yn),(xN,yN) = reset.label.clipbox.get_points()
        if xn < event.x < xN and yn < event.y < yN:
            delta=[0,0]
            net=attractorNetwork2D(N1,N2,int(excite.val),activity_mag,inhibit_scale.val)
            prev_weights=net.excitations(0,0)
            prev_weights= net.update_weights_dynamics(prev_weights,delta[0],delta[1])
            prev_weights[prev_weights<0]=0

            resetDone ^= False
            pause ^=True

    '''animation for Place Cells'''
    excite.on_changed(update)
    delta1.on_changed(update)
    delta2.on_changed(update)
    inhibit_scale.on_changed(update)
    fig1.canvas.mpl_connect('button_press_event', onClick)
    ani = FuncAnimation(fig1, animate)
    plt.show() 

def fixedShift_visulaisation():
    N1,N2,excite_radius,activity_mag,inhibit_scale=  60, 60, 2, 1, 0.005
    delta_row, delta_col = 1,1
    net=attractorNetwork2D( N1,N2,excite_radius,activity_mag,inhibit_scale)
    prev_weights=net.excitations(30,30)

    fig = plt.figure(figsize=(13, 4))
    ax0 = fig.add_subplot(1, 1, 1)
    # ax1 = fig.add_subplot(1, 4, 2)
    # ax2 = fig.add_subplot(1, 4, 3)
    # ax3 = fig.add_subplot(1, 4, 4)

    def animate(i):
        global prev_weights
        ax0.clear()
        ax0.imshow(prev_weights)
        ax0.set_title('Previous Activity')

        prev_weights, copy, excite,inhibit_array = net.update_weights_dynamics(prev_weights,delta_row,delta_col,moreResults=True)
        # ax1.imshow(copy)
        # ax1.set_title('Copied and Shifted Activity')

        # ax2.imshow(excite)
        # ax2.set_title('Exctied Activity')

        # ax3.imshow(prev_weights)
        # ax3.set_title('Inhibited Activity')

    ani = FuncAnimation(fig, animate, interval=1,frames=1000,repeat=False)
    plt.show()
'''Testing''' 
  
plotting_CAN_dynamics()