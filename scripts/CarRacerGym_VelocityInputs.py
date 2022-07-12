import time 
import numpy as np 
import math

from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.widgets as wig
from matplotlib.patches import Rectangle 

from multiprocessing import Process, Pool
from multiprocessing.dummy import freeze_support
import multiprocessing

from carRacer import  FrictionDetector, CarRacing
from CAN import activityDecoding, activityDecodingAngle, attractorNetworkSettling, attractorNetwork, multiResolution


'''Parameters'''
N=[60,60] #number of neurons
neurons=[np.arange(0,N[0]), np.arange(0,N[1])]
curr_Neuron=[0,0]
num_links=[30,30]
excite=[20,47]
activity_mag=[1,1]
inhibit_scale=[0.005,0.005]
curr_parameter=[0,0]
SCALING_FACTOR=1


def driving_func(queue):
   a = np.array([0.0, 0.0, 0.0])
   import pygame

   def register_input():
      for event in pygame.event.get():
         if event.type == pygame.KEYDOWN:
               if event.key == pygame.K_LEFT:
                  a[0] = -1.0
               if event.key == pygame.K_RIGHT:
                  a[0] = +1.0                 
               if event.key == pygame.K_UP:
                  a[1] = +1.0     
               if event.key == pygame.K_DOWN:
                  a[2] = +0.8  # set 1.0 for wheels to block to zero rotation
               if event.key == pygame.K_RETURN:
                  global restart
                  restart = True
                  

         if event.type == pygame.KEYUP:
               if event.key == pygame.K_LEFT:
                  a[0] = 0
               if event.key == pygame.K_RIGHT:
                  a[0] = 0
               if event.key == pygame.K_UP:
                  a[1] = 0
               if event.key == pygame.K_DOWN:
                  a[2] = 0
               
   env = CarRacing()
   env.render()
   isopen = True
   
   while isopen:
      env.reset()
      total_reward = 0.0
      steps = 0
      global restart
      restart = False
      while True:
         t=time.time()
         register_input()
         
         posX= env.car.hull.position[0]
         posY= env.car.hull.position[1]

         linV=np.sqrt(np.square(env.car.hull.linearVelocity[0])+ np.square(env.car.hull.linearVelocity[1]))
         linV_x=env.car.hull.linearVelocity[0]
         linV_y=env.car.hull.linearVelocity[1]
         angV=env.car.hull.angularVelocity
         

         s, r, done, info = env.step(a)
         total_reward += r
         
         if done:
               print("\naction " + str([f"{x:+0.2f}" for x in a]))
               print(f"step {steps} total_reward {total_reward:+0.2f}")
         steps += 1
         isopen = env.render()

         queue.put((posX,posY,linV,linV_x, linV_y,angV,done,restart,total_reward, 0.02))

         time.sleep(0.1)
         
         
         if done or restart or isopen is False:
               break
   env.close()


def matplotlib_func(queue):
   # matplotlib stuff
   global curr_x, curr_y,decoded_pose, pause, prev_weights, angVel,decoded_true_pose, theta
   curr_x, curr_y=[],[]
   theta=[0]
   # decoded_x,decoded_y=[],[]
   decoded_pose=[(0,0,0)]
   decoded_true_pose=[[0,0,0]]
   # decoded_y=[0]
   angVel=[]
   pause=False 

   figw, figh = 7, 6
   fig = plt.figure(figsize=(figw, figh))
   fig.patch.set_facecolor('dimgrey')
   # plt.get_current_fig_manager().window.setGeometry(500,0,800,800)
   ax0 =  plt.subplot2grid(shape=(9, 16), loc=(2, 0), rowspan=6,colspan=7)
   # axy2 =  plt.subplot2grid(shape=(9, 16), loc=(8, 0), rowspan=1,colspan=7)
   axx =  plt.subplot2grid(shape=(9, 16), loc=(0, 0), rowspan=2, colspan=7)
   axy =  plt.subplot2grid(shape=(9, 16), loc=(2, 7), rowspan=6, colspan=2)
   ax1 = plt.subplot2grid(shape=(9, 16), loc=(0, 10), colspan=6, rowspan = 4, facecolor="#15B01A")
   ax2 = plt.subplot2grid(shape=(9, 16), loc=(8, 0), colspan=7, rowspan = 1, facecolor="dimgrey")
   ax3 = plt.subplot2grid(shape=(9, 16), loc=(5, 10), colspan=6, rowspan = 4, facecolor="#15B01A")
   

   # ax1 = plt.subplot(1,2,2, facecolor="#15B01A")
   # ax0 = plt.subplot(1,2,1)

   # gs = fig.add_gridspec(15,32)
   # #place cell
   # ax0 = plt.subplot(gs[4:14, 10:20])
   # axx = plt.subplot(gs[1:4, 10:20])
   # axy = plt.subplot(gs[4:14, 20:24])
   
   # #deconstructed CANN
   # axy1 = plt.subplot(gs[0:5, 0:8])
   # axy2 = plt.subplot(gs[5:10, 0:8])
   # axy3 = plt.subplot(gs[10:15, 0:8])

   # ax1 = plt.subplot(gs[0:15, 24:32],facecolor="#15B01A")

   # plt.subplots_adjust(bottom=0.25)
   # # fig.tight_layout()

   # '''Slider for Parameters'''
   # button_ax = plt.axes([.05, .03, .05, .04], facecolor='white') # x, y, width, height
   # # button2_ax = plt.axes([.05, .03, .05, .04], facecolor='white')
   # # Nax = plt.axes([0.25, 0.17, 0.65, 0.03], facecolor='white')
   # exciteax = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor='white')
   # inhax = plt.axes([0.25, 0.03, 0.65, 0.03], facecolor='white')
   
   # inhax.tick_params(axis='x', colors='white')    #setting up X-axis tick color to red
   # inhax.tick_params(axis='y', colors='white') 

   # # Create a slider from 0.0 to 20.0 in axes axfreq with 3 as initial value
   # start_stop=wig.Button(button_ax,label='$\u25B6$')
   # # reset=wig.Button(button2_ax,'Total Score')
   # inhibit_scale=wig.Slider(inhax, 'Inhibition', 0, 0.05, 0.01, color='#008000', track_color='white')
   # inhibit_scale.valtext.set_color("white")
   # inhibit_scale.label.set_color('white')

   # excite = wig.Slider(exciteax, 'Excitation', 0, 40, 10, valstep=2, color='#008000',track_color='white')
   # excite.valtext.set_color("white")
   # excite.label.set_color('white')

   # N = wig.Slider(Nax, 'Neurons', 20, 400, 300, valstep=10, color='#008000',track_color='white')
   # N.valtext.set_color("white")
   # N.label.set_color('white')   
   # delta2 = wig.Slider(delta2ax, 'Delta 2', -10, 10, 0, valstep=1)

   '''Initalise network'''            
   delta=[0,0]
   prev_weights=[np.zeros(N[0]), np.zeros(N[0]), np.zeros(N[0]), np.zeros(N[0]),np.zeros(N[0])]

   # for i in range(len(delta)):
   #    net=attractorNetworkSettling(int(N.val),num_links[i],int(excite.val), activity_mag[i],inhibit_scale.val)
   #    prev_weights[i][net.activation(delta[i])]=net.full_weights(num_links[i])

   def imageHistogram(prev_weights,N,val):
      prev_weights[prev_weights<0]=0
      prev_weights/np.linalg.norm(prev_weights)
      height=100
     
      hist1=np.zeros((height,N*2))
      for n in range(N*2):
         if n%2==0:
            coloured=int(np.round(prev_weights[n//2],2)*100)
            if coloured != 0:
               hist1[:coloured, n]=[val]*coloured
      return hist1

   # def imageHistogramMultiple(excite_store,N,val):
   #    # prev_weights[prev_weights<0]=0
   #    # prev_weights/np.linalg.norm(prev_weights)
   #    height=100
   #    hist1=np.zeros((height,N*2))
   #    for n in range(N*2):
   #       if n%2==0 and np.all((hist1[:, n]==0)):
   #          coloured=int(np.round(prev_weights[n//2],2)*100)
   #          if coloured != 0:
   #             hist1[:coloured, n]=[val]*coloured
   #    return hist1

   def poseUpdate(pose, linV, angV, dt):
      theta_old=pose[2]

      xdot=linV*np.cos(theta_old)
      ydot=linV*np.sin(theta_old)
      theta_dot=angV

      x=dt*xdot + pose[0]
      y=dt*ydot + pose[1]
      theta= dt*theta_dot + pose[2]

      return (x,y,theta)
   
   def multiResolutionUpdate(input,N,num_links,excite, activity_mag,inhibit_scale): 
      delta, scale = multiResolution(abs(input))
      split_output=np.zeros((len(delta)))
      
      '''updating network'''    
      net=attractorNetworkSettling(N,num_links,excite, activity_mag,inhibit_scale)
      for k in range(5):
         for n in range(len(delta)):
            prev_weights[n][:]= net.update_weights_dynamics(prev_weights[n][:],delta[n])
            prev_weights[n][prev_weights[n][:]<0]=0
            split_output[n]=np.argmax(prev_weights[n][:])#-prev_trans
      '''decoding mangnitude and direction of movement'''
      decoded=np.sum(split_output*scale)*np.sign(input)
      
      return decoded  

   def animate(i):
      t = time.time()
      global curr_x, curr_y, prev_weights,decoded_pose, decoded_true_pose, pause, angVel,theta
      while not queue.empty() and not pause:
         posX,posY,linV,linV_x, linV_y,angV,done,restart,reward,del_t = queue.get() 
         curr_x.append(posX)
         curr_y.append(posY)
         angVel.append(angV)

         if done or restart:
            curr_x,curr_y=[],[]
            ax2.clear()
            ax1.clear()
 
         if i>1 and len(curr_x)>2:
            delta[0]=linV
            delta[1]=angVel[-1]
            decoded=[0,0]
            '''updating network'''
            for j in range(len(delta)):
               decoded[j]=multiResolutionUpdate(delta[j],N[j],num_links[j],excite[j], activity_mag[j],inhibit_scale[j])

            # decoded_true_pose.append([delta[0]/SCALING_FACTOR+decoded_true_pose[-1][0],delta[1]/SCALING_FACTOR+decoded_true_pose[-1][1],theta[-1]+decoded_true_pose[-1][2]])
            decoded_true_pose.append(poseUpdate(decoded_true_pose[-1], linV, angV, del_t))
            
            decoded_pose.append(poseUpdate(decoded_pose[-1],  decoded[0], decoded[1], del_t))

      
            '''plotting'''
            ax2.clear()
            ax2.axis('off')
            ax2.text(0,0, "Total Score: " + str(np.round(reward,2)), c='r')

            # ax0.clear()   
            # # ax0.set_title("Attractor Network", color='white')
            # ax0.imshow(im, interpolation='nearest', aspect='auto')
            # # ax0.axis('off')
            # ax0.invert_yaxis()
            # ax0.spines[['top', 'right', 'left', 'bottom']].set_color('white')   
            # ax0.tick_params(axis='x', colors='white')    #setting up X-axis tick color to red
            # ax0.tick_params(axis='y', colors='white') 

            
            # mapelites_colours = np.vstack((np.array([105/255,105/255,105/255,1]), plt.get_cmap('Set3')(np.arange(256))))
            # mapelites_colours = ListedColormap(mapelites_colours, name='mapelites', N = mapelites_colours.shape[0])
            # simple_colour= ListedColormap(['dimgrey', "#C79FEF"])
            # simple_colour2= ListedColormap(['dimgrey',"#E3E4FA"])
            # color = ListedColormap(['dimgrey', "#728FCE", "#000080","#3090C7","#E3E4FA","#C79FEF","#48D1CC","#045F5F","#50C878","#808000","#254117","#B2C248","#E2F516","#E3F9A6","#FFFFE0","#FFE4C4","#FFE87C","#FBB917","#C8B560","#C19A6B","#C88141","#665D1E","#513B1C","#C04000","#E78A61","#9F000F","#810541","#7E354D","#FDD7E4","#FC6C85","#B048B5","#4E387E","#DCD0FF","#CC6600"])

            # axx.clear()
            # axx.set_title("Attractor Network", color='white')
            # axx.imshow(imageHistogram(prev_weights[0][:],N.val,1),interpolation='nearest', aspect='auto',cmap=simple_colour)
            # axx.invert_yaxis()
            # axx.axis('off')

            # axy.clear()
            # axy.imshow(np.transpose(imageHistogram(prev_weights[1][:],N.val,1)),interpolation='nearest', aspect='auto',cmap=simple_colour)
            # axy.invert_yaxis()
            # axy.axis('off')


            # decoded_pose.append(poseUpdate(decoded_pose[-1], linV, angV, del_t))
            

            # ax1.clear()
            ax1.set_title('True CarRacer Position', color='white')
            ax1.axis('equal')
            ax1.spines[['top', 'right', 'left', 'bottom']].set_color('white')   
            ax1.tick_params(axis='x', colors='white')    #setting up X-axis tick color to red
            ax1.tick_params(axis='y', colors='white') 
            ax1.scatter(decoded_true_pose[-1][0], decoded_true_pose[-1][1],s=10,c='r')

            ax3.set_title('Decoded CarRacer Position', color='white')
            ax3.axis('equal')
            ax3.spines[['top', 'right', 'left', 'bottom']].set_color('white')   
            ax3.tick_params(axis='x', colors='white')    #setting up X-axis tick color to red
            ax3.tick_params(axis='y', colors='white') 
            ax3.scatter(decoded_pose[-1][0], decoded_pose[-1][1],s=10,c='r')

            # print(str(decoded_true_pose[-1][0])+"  "+str( decoded_true_pose[-1][1])+ "_______"+str(decoded_pose[-1][0] )+"  "+str(decoded_pose[-1][1]))
            # print(str(linV)+"  "+str( angVel[-1])+ "_______"+str(x )+"  "+str(y))

            # axy3.clear(), axy3.spines[['top', 'left', 'right']].set_visible(False), axy3.spines.bottom.set_color('white')
            # axy3.tick_params(axis='y',which='both', left=False, right=False, labelleft=False), axy3.tick_params(axis='x', colors='white')
            # axy3.imshow(imageHistogram(exciteInhi,N.val,1),interpolation='nearest', aspect='auto',cmap=simple_colour), axy3.invert_yaxis()

            # excite_color = np.linspace(0, 1, len(excite_store))
            # excite_hist=np.zeros((100,N.val*2))
            # for k in range(len(excite_store)):
            #    if not np.all((excite_store[k])== 0):
            #       excite_hist=imageHistogramMultiple(excite_store[k],N.val,excite_color[k],excite_hist)
            # # print(excite_hist)

            # axy2.clear(), axy2.spines[['top', 'left', 'right']].set_visible(False), axy2.spines.bottom.set_color('white')
            # axy2.tick_params(axis='y',which='both', left=False, right=False, labelleft=False), axy2.tick_params(axis='x', colors='white')
            # axy2.imshow(excite_hist,interpolation='nearest', aspect='auto',cmap=color), axy2.invert_yaxis()

            # axy1.clear(), axy1.spines[['top', 'left', 'right']].set_visible(False), axy1.spines.bottom.set_color('white')
            # axy1.tick_params(axis='y',which='both', left=False, right=False, labelleft=False), axy1.tick_params(axis='x', colors='white')
            # axy1.imshow(imageHistogram(prev_weights[1][:],N.val,1),interpolation='nearest', aspect='auto',cmap=simple_colour2), axy1.invert_yaxis()

            
            


   # def update(val):
   #    global prev_weights, num_links, activity_mag
   #    '''distributed weights with excitations and inhibitions'''
   #    # delta=[int(delta1.val),int(delta2.val)]
   #    prev_weights=[np.zeros(N[0]), np.zeros(N[1])]
   #    for j in range(len(delta)):
   #       net=attractorNetworkSettling(N[j],num_links[j],int(excite.val), activity_mag[j],inhibit_scale.val)
   #       prev_weights[j][:]= net.update_weights_dynamics(prev_weights[j][:],delta[j])
   #       prev_weights[j][prev_weights[j][:]<0]=0

   # def onClick(event):
   #    global pause, prev_weights # resetDone
   #    (xm,ym),(xM,yM) = start_stop.label.clipbox.get_points()
   #    if xm < event.x < xM and ym < event.y < yM:
   #       pause ^= True

            
   # '''animation for Place Cells'''
   # excite.on_changed(update)
   # # delta1.on_changed(update)
   # # N.on_changed(update)
   # inhibit_scale.on_changed(update)
   # fig.canvas.mpl_connect('button_press_event', onClick)
   ani = FuncAnimation(fig, animate, interval=1)
   plt.show() 
       
            
   
if __name__=="__main__":
    freeze_support()
    #conn1, conn2 = multiprocessing.Pipe()
    queue = multiprocessing.Queue()
    process_1 = multiprocessing.Process(target=driving_func, args=(queue,))
    process_2 = multiprocessing.Process(target=matplotlib_func, args=(queue,))
    process_1.start()
    process_2.start()
    process_1.join()
    process_2.join()


# def my_func(is_matplotlib):
#     if is_matplotlib:   
#         #matplotlib stuff    
#     else:
#         #carRacer stuff

# if __name__=="__main__":
#     freeze_support()

#     with Pool(processes=2) as pool:
#         values = [[True],[False]]
#         res = pool.starmap(my_func, values)
#         for r in res:
#             while True: pass
