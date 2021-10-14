################################################################
# 
# Owen Blair
# ECE351-52
# Lab #6
# 10/7/2021
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

#--------------END FUNCTIONS-----------------------------#

#---------PT 1-------------------------------------------#
    #Define step size
steps = 1e-2

    #t for part 1
start = 0
stop = 2
    #Define a range of t_pt1. Start @ 0 go to 20 (+a step) w/
    #a stepsize of step
t = np.arange(start, stop + steps, steps)

    #Plot prelab results
h = (0.5 + (-0.5 * np.exp(-4 * t)) + (np.exp(-6 * t))) * stepFunc(t, 0, 1)

    #Make the H(s) using the sig.step() function!!
    
num = [1, 6, 12] #Creates a matrix for the numerator
den = [1, 10, 24] #Creates a matrix for the denominator

tout , yStep = sig.step((num , den), T = t)

den_residue = [1, 10, 24, 0]

    #Make and print the partial fraction decomp
roots, poles, _ = sig.residue(num, den_residue)

print("Partial frac decomp Roots")
print(roots)
print("")
print("Partial frac decomp Poles")
print(poles)


    #Make plots for pt1
plt.figure(figsize=(10,7))
plt.subplot(2,1,1)
plt.plot(t,h)
plt.grid()
plt.ylabel('Hand Solved Output')
plt.title('Plots of h(t) and sig.step Function')

plt.subplot(2,1,2)
plt.plot(t,yStep)
plt.grid()
plt.ylabel('sig.step Output')


#------------PART 2-------------------------

    #Define step size
steps = 1e-2

    #t for part 1
start = 0
stop = 4.5
    #Define a range of t_pt1. Start @ 0 go to 20 (+a step) w/
    #a stepsize of step
t_pt2 = np.arange(start, stop + steps, steps)


#System is:
#y^(5)(t) + 18y^(4)(t) + 218y^(3)(t) + 2036y^(2)(t) + 9085y^(1)(t) + 25250y(t)
#        = 25250x(t)
#
#The ^(number) signifies the diritive of the function y(t). I.e. y^(6)(t) would
#be the 6th diritive of the funciton y(t)


    #Make numerator and denomentaor for sig.residue()
num_pt2 = [25250]
den_pt2 = [1, 18, 218, 2036, 9085, 25250, 0]

roots_pt2, poles_pt2, _2 = sig.residue(num_pt2, den_pt2)

print("Roots and Poles for pt 2")
print("Roots_pt2")
print(roots_pt2)
print("")
print("Poles_pt2")
print(poles_pt2)

    #COSINE METHOD! Using the poles found previously
ytCosineMethod = 0

    #Range iterates through each root
for i in range(len(roots_pt2)):
    angleK = np.angle(roots_pt2[i])
    magOfK = np.abs(roots_pt2[i])
    W = np.imag(poles_pt2[i])
    a = np.real(poles_pt2[i])
    
        #DEBUG!!!
    print("angle = ", angleK)
    print("magnatude = ", magOfK)
        #END DEBUG
        
    ytCosineMethod += magOfK * np.exp(a * t_pt2) * np.cos(W * t_pt2 + angleK) * stepFunc(t_pt2, 0, 1)
    
#Make the lib generated step response
den_pt2_step = [1, 18, 218, 2036, 9085, 25250]
tStep_pt2, yStep_pt2 = sig.step((num_pt2,den_pt2_step), T = t_pt2)

    #Show Plots
plt.figure(figsize=(10,7))
plt.subplot(2,1,1)
plt.plot(t_pt2, ytCosineMethod)
plt.grid()
plt.xlabel('t')
plt.ylabel('y(t)')
plt.title('Cosine Method vs. Lib sig.step Method')


plt.subplot(2,1,2)
plt.plot(tStep_pt2, yStep_pt2)
plt.grid()
plt.xlabel('t')
plt.ylabel('sig.step y(t)')

#-------------SHOW ALL PLOTS-----------------
plt.show()
