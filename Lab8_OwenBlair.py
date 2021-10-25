################################################################
#
# Owen Blair
# ECE351-52
# Lab #8
# 10/21/2021
#
################################################################
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import math as m

#------------FUNCTIONS----------------------------------------#

# Make a step function using an array t, stepTime, and stepHeight


def stepFunc(t, startTime, stepHeight):
    y = np.zeros(t.shape)
    for i in range(len(t)):
        if(t[i] >= startTime):
            y[i] = stepHeight
    return y


# Make a ramp function using an array t, startTime, and slope
def rampFunc(t, startTime, slope):
    y = np.zeros(t.shape)
    for i in range(len(t)):
        if(t[i] >= startTime):
            y[i] = t[i]-startTime
    y = y * slope
    return y


# Make e^at function using array t, startTime, and a (alpha)
def eExpo(t, amplatude, alpha):
    y = np.zeros(t.shape)
    for i in range(len(t)):
        y[i] = amplatude * m.exp(alpha * (t[i]))

    return y


# Time reversal using t and a function plot
def timeReversal(ary):

    # Make an array to return time reversal plot
    timeReverse = np.zeros(ary.shape)

    # Goes from index 0 to index len(f)-1
    for i in range(0, len(ary)-1):
        timeReverse[i] = ary[(len(ary) - 1)-i]

        # Return the time reversed array
    return timeReverse


# Convolution function!!!
def convolve(f1, f2):

    # Both functions need to be the same size!
    Nf1 = len(f1)
    Nf2 = len(f2)

    # Debug statements
    # print(Nf1)
    # print(Nf2)

    f1Extend = np.append(f1, np.zeros((1, Nf2-1)))
    f2Extend = np.append(f2, np.zeros((1, Nf1-1)))

    y = np.zeros(f1Extend.shape)

    for i in range(Nf1 + Nf2 - 2):
        y[i] = 0

        for j in range(Nf1):
            if (i-j+1 > 0):
                try:
                    y[i] += f1Extend[j] * f2Extend[i-j+1]

                # Where am I running out of space in my array?
                except:
                    print(i, j)
    return y

#--------------END FUNCTIONS-----------------------------#


#---------PT 1-------------------------------------------#

# Finding b_n for fourier estimation given an n
def b_n(n):
    b = (-2/((n)*np.pi)) * (np.cos((n) * np.pi) - 1)
    return b


def W(period):
    return ((2*np.pi)/period)


def xFourier(t, period, n):
    x_t = 0
    for i in np.arange(1, n+1):
        x_t += (np.sin(i * W(period) * t) * b_n(i))
        
    return x_t


"""
 a_0 = 0 because function has a DC offset of 0
 a_n = 0 Because function is a reflection along the line y=-x
     This means that the function will always have non-zero coefficients
     in the sine terms.
"""

# Define step size
#steps = 1e-2
steps = 1e-1

# t for part 1
start = 0
stop = 20
# Define a range of t_pt1. Start @ 0 go to 20 (+a step) w/
# a stepsize of step
t = np.arange(start, stop + steps, steps)


# Calculate b_n for testing
n1 = 1
#print(b_n1)

# Make arrays to plot against t
x_1 = xFourier(t, 8, 1)
x_3 = xFourier(t, 8, 3)
x_15 = xFourier(t, 8, 15)
x_50 = xFourier(t, 8, 50)
x_150 = xFourier(t, 8, 150)
x_1500 = xFourier(t, 8, 1500)

#Define plot size!
plt.figure(figsize=(10, 12))
# set the spacing between subplots
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9,
                    top=1.0, wspace=0.4, hspace=0.4)

plt.subplot(6, 1, 1)
plt.plot(t, x_1)
plt.title("Estimation when N=1")
plt.ylabel("x(t) Output")
plt.grid()

plt.subplot(6, 1, 2)
plt.plot(t, x_3)
plt.title("Estimation when N=3")
plt.ylabel("x(t) Output")
plt.grid()

plt.subplot(6, 1, 3)
plt.plot(t, x_15)
plt.title("Estimation when N=15")
plt.ylabel("x(t) Output")
plt.grid()

plt.subplot(6, 1, 4)
plt.plot(t, x_50)
plt.title("Estimation when N=50")
plt.ylabel("x(t) Output")
plt.grid()

plt.subplot(6, 1, 5)
plt.plot(t, x_150)
plt.title("Estimation when N=150")
plt.ylabel("x(t) Output")
plt.grid()

plt.subplot(6, 1, 6)
plt.plot(t, x_1500)
plt.title("Estimation when N=1500")
plt.ylabel("x(t) Output")
plt.grid()

plt.show()

print("a0 and all values for a are zero. This is")
print("because the wave is odd and has no  offset.")
print()
for i in range(1,4):
    print("B%d is" %i)
    print(b_n(i))