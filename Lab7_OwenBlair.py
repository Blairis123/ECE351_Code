################################################################
# 
# Owen Blair
# ECE351-52
# Lab #7
# 10/21/2021
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

        #THIS WILL BE THE SAME FOR THE ENTIRE LAB!

    #Define step size
steps = 1e-2

    #t for part 1
start = 0
stop = 10
    #Define a range of t_pt1. Start @ 0 go to 20 (+a step) w/
    #a stepsize of step
t = np.arange(start, stop + steps, steps)

#--------------PART 1, OPEN-LOOP------------------------------------#
    #Find the roots and the poles
A_num = [1, 4]
A_den = [1, 4, 3]

B_num = [1, 26, 168]
B_den = [1]

G_num = [1, 4]
G_den = [1, -2, -40, -64]

zeros_A, pole_A, gain_A = sig.tf2zpk(A_num, A_den)
zeros_B, pole_B, gain_B = sig.tf2zpk(B_num, B_den)
zeros_G, pole_G, gain_G = sig.tf2zpk(G_num, G_den)

print("Zeros for A")
print(zeros_A)
print("")
print("Poles for A")
print(pole_A)
print("")
print("Zeros for B")
print(zeros_B)
print("")
print("Poles for B")
print(pole_B)
print("")
print("Zeros for G")
print(zeros_G)
print("")
print("Poles for G")
print(pole_G)
print("")

H_open_num = [1, 9]
H_open_den = [1, -2, -37, -82, -48]

zero_H_open, pole_H_open, gain_H_open = sig.tf2zpk(H_open_num, H_open_den)

print("Poles of the open loop transfer function")
print(pole_H_open)
print("")

stepTOpen, stepHOpen = sig.step((H_open_num, H_open_den), T = t)

plt.figure(figsize=(10,7))
plt.plot(stepTOpen, stepHOpen)
plt.grid()
plt.xlabel("Time")
plt.ylabel("Output")
plt.title("Open Loop Transfer Function Step Response")

#-------------PART 2, CLOSED-LOOP--------------------------------------------#

    #Finding the Numerator and Denomenator of transfer function
H_clo_num = sig.convolve([1,4],[1,9])

part_den = sig.convolve([1,1],[1,3])
H_clo_den = sig.convolve(part_den,[2,33,362,1448])

print("Closed loop numerator = ", H_clo_num)
print("Closed loop denemonator = ", H_clo_den)
print()

    #This is what the num and den should look like
#H_clo_num = [1, 13, 36]
#H_clo_den = [2, 41, 500, 2995, 6878, 4344]

zeros_H_clo, pole_H_clo, gain_H_clo = sig.tf2zpk(H_clo_num, H_clo_den)

print("Poles of the closed loop transfer function:")
print(pole_H_clo)
print("")

    #Define transfer function!
stepTClosed, stepHClosed = sig.step((H_clo_num, H_clo_den), T = t)

    #Make plots for pt1
plt.figure(figsize=(10,7))
plt.plot(stepTClosed, stepHClosed)
plt.grid()
plt.xlabel("Time")
plt.ylabel("Output")
plt.title("Closed Loop Transfer Function Step Response")

#--------SHOW PLOTS--------------------#
plt.show()