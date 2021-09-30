################################################################
# 
# Owen Blair
# ECE351-52
# Lab #4
# 9/30/2021
# 
# 
################################################################
import numpy as np
import matplotlib.pyplot as plt
#import scipy.signal as sig
import math

#------------FUNCTIONS----------------------------------------#

def cosine(t): # The only variable sent to the function is t
    y = np.zeros(t.shape) # initialze y(t) as an array of zeros
    for i in range(len(t)): # run the loop once for each index of t
        y[i] = np.cos(t[i])
            
    #Return the cosine function
    return y

#Make a step function using an array t, stepTime, and stepHeight
def stepFunc(t, startTime, stepHeight):
    y = np.zeros(t.shape)
    for i in range(len(t)):
        if(t[i] >= startTime):
            y[i] = stepHeight
    return y

#Make a ramp function using an array t, startTime, and slope
def rampFunc(t, startTime, slope):
    y = np.zeros(t.shape)
    for i in range(len(t)):
        if(t[i] >= startTime):
            y[i] = t[i]-startTime
    y = y * slope
    return y

#Make e^at function using array t, startTime, and a (alpha)
def eExpo(t,amplatude,alpha):
    y = np.zeros(t.shape)
    for i in range(len(t)):
        y[i]=amplatude * math.exp(alpha * (t[i]))
    
    return y

#Time reversal using t and a function plot
def timeReversal(ary):

        #Make an array to return time reversal plot
    timeReverse = np.zeros(ary.shape)
    
            #Goes from index 0 to index len(f)-1
    for i in range(0, len(ary)-1):
        timeReverse[i] = ary[(len(ary) - 1)-i]
            
        #Return the time reversed array
    return timeReverse

#Convolution function!!!
def convolve(f1, f2):
    
    #Both functions need to be the same size!
    Nf1 = len(f1)
    Nf2 = len(f2)
    
    #Debug statements
    #print(Nf1)
    #print(Nf2)
    
    f1Extend = np.append(f1,np.zeros((1,Nf2-1)))
    f2Extend = np.append(f2,np.zeros((1,Nf1-1)))
    
    y = np.zeros(f1Extend.shape)
    
    for i in range(Nf1 + Nf2 - 2):
        y[i] = 0
        
        for j in range(Nf1):
            if (i-j+1 >0):
                try:
                    y[i] += f1Extend[j] * f2Extend[i-j+1]
                
                #Where am I running out of space in my array?
                except:
                    print(i,j)
    return y
#--------------END FUNCTIONS-----------------------------#



#---------------------PART 1-----------------------------#

    #Define step size
steps = 1e-2

    #t for part 1
start = -10
stop = 10
    #Define a range of t_pt1. Start @ 0 go to 20 (+a step) w/
    #a stepsize of step
t_pt1 = np.arange(start, stop + steps, steps)


    #Make h1(t) = e^(-2t)[u(t)-u(t-3)]
h1 = eExpo(t_pt1, 1, -2) * (stepFunc(t_pt1, 0, 1) - stepFunc(t_pt1, 3, 1))

    #Make h2(t) = u(t-2) - u(t-6)
h2 = stepFunc(t_pt1, 2, 1) - stepFunc(t_pt1, 6, 1)

    #Make h3(t) = cos(wt)*u(t) --> frequency = 0.25
    #This means that w is about 1.5707963 (0.25 * 2 * pi)
h3 = cosine(1.5707963 * t_pt1) * stepFunc(t_pt1, 0, 1)

#-----MAKE PLOTS--------#
    #Make plot and then show it
    #Make plots for Part 1-----------#
plt.figure(figsize=(10,7))
plt.subplot(3,1,1)
plt.plot(t_pt1,h1)
plt.grid()
plt.ylabel('h1 Output')
plt.title('Plots of h1, h2, and h3')

plt.subplot(3,1,2)
plt.plot(t_pt1,h2)
plt.grid()
plt.ylabel('h2 Output')

plt.subplot(3,1,3)
plt.plot(t_pt1,h3)
plt.grid()
plt.ylabel('h3 Output')


#---------------------PART 2-----------------------------#
    #t for part 2
start = -10
stop = 10
    #Define a range of t_pt1. Start @ -10 go to 10 (+a step) w/
    #a stepsize of step
t_pt2 = np.arange(start, stop + steps, steps)


    #Make h1(t) = e^(-2t)[u(t)-u(t-3)]
h1 = eExpo(t_pt2, 1, -2) * (stepFunc(t_pt2, 0, 1) - stepFunc(t_pt2, 3, 1))

    #Make h2(t) = u(t-2) - u(t-6)
h2 = stepFunc(t_pt2, 2, 1) - stepFunc(t_pt2, 6, 1)

    #Make h3(t) = cos(wt)*u(t) --> frequency = 0.25
    #This means that w is about 1.5707963 (0.25 * 2 * pi)
h3 = cosine(1.5707963 * t_pt2) * stepFunc(t_pt2, 0, 1)



    #Make step response to convolve with --> f(t) = u(t)
forceFunc = stepFunc(t_pt1, 0, 1)

    #Make convolutions for step response!
h1StepResponse = convolve(h1, forceFunc) * steps

h2StepResponse = convolve(h2, forceFunc) * steps

h3StepResponse = convolve(h3, forceFunc) * steps

#Make a tConv range to pot the convolve functions!
#This should be the same as the size for all convolutions for this lab
tConv = np.arange(0, ((len(h3StepResponse) -1) * steps) + steps, steps) -20


#-----MAKE PLOTS--------#
    #Make plot and then show it
    #Make plots for Part 2-----------#
plt.figure(figsize=(10,7))
plt.subplot(3,1,1)
plt.plot(tConv,h1StepResponse)
plt.grid()
plt.ylabel('Output')
plt.title('h1, h2, and h3 Step Response')

plt.subplot(3,1,2)
plt.plot(tConv,h2StepResponse)
plt.grid()
plt.ylabel('Output')


plt.subplot(3,1,3)
plt.plot(tConv,h3StepResponse)
plt.grid()
plt.ylabel('Output')



#------------SHOW ME WHAT YOU GOT!-----------------------#
plt.show()