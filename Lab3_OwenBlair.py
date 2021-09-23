################################################################
# 
# Owen Blair
# ECE351-52
# Lab #3
# 9/23/2021
# Other info
# 
################################################################
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
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
def eExpo(t,startTime,amplatude,alpha):
    y = np.zeros(t.shape)
    for i in range(len(t)):
        if(t[i] >= startTime):
            y[i]=amplatude * math.exp(alpha * (t[i]-startTime))
    
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

    #Define step size
steps = 1e-2

    #Define a range of t. Start @ 0 go to 10 (+a step) w/ a stepsize of step
t = np.arange(0, 20+ steps, steps)

#Make stuff to plot!
func1 = stepFunc(t,2,1) - stepFunc(t, 9, 1)
func2 = eExpo(t,0,1,-1)
func3 = (rampFunc(t,2,1) * (stepFunc(t, 2, 1) - stepFunc(t, 3, 1))) + ((rampFunc(t, 3, -1) + 1) * (stepFunc(t, 3, 1) - stepFunc(t, 4, 1)) ) #x * y

#Make convolutions by hand!
f1Convolvef2 = convolve(func1, func2)
f2Convolvef3 = convolve(func2, func3)
f1Convolvef3 = convolve(func1, func3)

#Make convolutios using scipy.siganl.convolve!
f1Conf2Lib = sig.convolve(func1, func2)
f2Conf3Lib = sig.convolve(func2, func3)
f1Conf3Lib = sig.convolve(func1, func3)

#Make a t range to pot the convolve functions!
#This should be the same as the size for all convolutions for this lab
tConv = np.arange(0, len(f1Convolvef2) * steps, steps)

    #Make plot and then show it
plt.figure(figsize=(10,7))
plt.subplot(3,1,1)
plt.plot(t,func1)
plt.grid()
plt.ylabel('Output')
plt.title('Function 1, Function 2, Function 3')

plt.subplot(3,1,2)
plt.plot(t,func2)
plt.grid()
plt.ylabel('Output')

plt.subplot(3,1,3)
plt.plot(t,func3)
plt.grid()
plt.ylabel('Output')

#New plot with subplots for convolve!
    #f1 convolve f2
plt.figure(figsize=(10,7))
plt.subplot(3,2,1)
plt.plot(tConv, f1Convolvef2)
plt.grid()
plt.ylabel("Output f1 conv. f2")
plt.title("User Convolve")

plt.subplot(3,2,2)
plt.plot(tConv, f1Conf2Lib)
plt.grid()
plt.ylabel("Output f1 conv. f2")
plt.title("Lib. Convolve")

    #f2 convolve f3
plt.subplot(3,2,3)
plt.plot(tConv, f2Convolvef3)
plt.grid()
plt.ylabel("Output f2 conv. f3")

plt.subplot(3,2,4)
plt.grid()
plt.plot(tConv, f2Conf3Lib)

    #f1 convolve f3
plt.subplot(3,2,5)
plt.plot(tConv, f1Convolvef3)
plt.grid()
plt.ylabel("Output f1 conv. f3")

plt.subplot(3,2,6)
plt.grid()
plt.plot(tConv, f1Conf3Lib)

plt.show()
