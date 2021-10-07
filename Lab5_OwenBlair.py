################################################################
# 
# Owen Blair
# ECE351-52
# Lab #5
# 9/30/2021
# 
################################################################
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import math as m

#------------FUNCTIONS----------------------------------------#

def cosine(t): # The only variable sent to the function is t
    y = np.zeros(t.shape) # initialze y(t) as an array of zeros
    for i in range(len(t)): # run the loop once for each index of t
        y[i] = np.cos(t[i])
            
    #Return the cosine function
    return y

def sine(t):    #Like cosine function but with sine instead!
    y=np.zeros(t.shape)
    for i in range(len(t)): #For each index of t
        y[i] = np.sin(t[i])
        
    return y #Return a value

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
        y[i]=amplatude * m.exp(alpha * (t[i]))
    
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

def func2Plot(t, R, L, C):
    
    X = (1/(R*C))
    tmpX = -0.5 * ((1/(R*C))**2)
    tmpY = 0.5 * X * (m.sqrt((X**2)) - (4/(L*C)))
    
    magOfG = m.sqrt( (tmpX**2) + (tmpY**2))
    
    tmpTanX = 0.5 * (m.sqrt((X**2)) - (4/(L*C)))
    tmpTanY = -0.5 * ((1/(R*C))**2)
    
    degOfG = m.atan(tmpTanX / tmpTanY)
    
    omega = 0.5 * np.sqrt(X**2 - 4*(1/np.sqrt(L*C))**2 + 0*1j)
    alpha = (-0.5) * X
    
    y = (magOfG/np.abs(omega)) * np.exp(alpha * t) * np.sin(np.abs(omega) * t + degOfG) * stepFunc(t, 0, 1)
    
    return y
#--------------END FUNCTIONS-----------------------------#


    #Make steps for t! From 0 to 1.2 ms this would be 
steps = 1e-6

    #t for part 1
start = 0
stop = 1.2e-3
    #Define a range of t_pt1. Start @ 0 go to 20 (+a step) w/
    #a stepsize of step
t = np.arange(start, stop + steps, steps)



#-------STUF TO MAKE THE FUNCTION FOR PT1-----------

    #For circuit-------------RLC DEF----------
R = 1e3
L = 27e-3
C = 100e-9

y = func2Plot(t, R, L, C)

num = [0, 1/(R*C), 0] #Creates a matrix for the numerator
den = [1, 1/(R*C), (1/(L*C))] #Creates a matrix for the denominator

tout , ySig = sig.impulse ((num , den), T = t)

ySig = ySig * (4e-4)
    #Make plots
plt.figure(figsize=(10,7))
plt.subplot(2,1,1)
plt.plot(t,y)
plt.grid()
plt.ylabel('y(t) Output')
plt.title('Plots of y(t) and sig.impulse Function')

plt.subplot(2,1,2)
plt.plot(t,ySig)
plt.grid()
plt.ylabel('sig.impulse Output')

#-----------------------PT2 CODE-----------
tout2 , yStep = sig.step ((num , den), T = t)

#---------------PT 2 PLOTS----------------
plt.figure(figsize=(10,7))
plt.subplot(2,1,1)
plt.plot(t,ySig)
plt.grid()
plt.ylabel('ySig Output')
plt.title('Plots of sig.impulse and sig.step Function')

plt.subplot(2,1,2)
plt.plot(t,yStep)
plt.grid()
plt.ylabel('sig.step Output')

plt.show()








