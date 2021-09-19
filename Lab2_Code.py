################################################################
# 
# Owen Blair
# ECE351-52
# Lab #2
# 9/16/2021
# 
################################################################
import numpy as np
import matplotlib.pyplot as plt

#------------FUNCTIONS----------------------------------------#

#Make a cosine wave using an array t (time)
def cosine(t): # The only variable sent to the function is t
    y = np.zeros(t.shape) # initialze y(t) as an array of zeros
    for i in range(len(t)): # run the loop once for each index of t
        y[i] = np.cos(t[i])
            
    #Return the cosine function
    return y

#Make a step function using an array t, stepTime, and stepHeight
def stepFunc(t, startTime, stepHeight):
    y= np.zeros(t.shape)
    for i in range(len(t)):
        if(t[i]>=startTime):
            y[i] = stepHeight
    return y

#Make a ramp function using an array t, startTime, and slope
def rampFunc(t, startTime, slope):
    y=np.zeros(t.shape)
    for i in range(len(t)):
        if(t[i]>=startTime):
            y[i]=slope * (t[i]-startTime)
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

#Time shift of a plot
def timeShift(timePlot, shift):
    
    timePlot += shift
    return timePlot

#Time scale
def timeScale(t, scale):
    for i in range(0, len(t)-1):
        t[i]=t[i] * scale
        
    return t
#--------------END FUNCTIONS-----------------------------#

    #Define step size
steps = 1e-2

#Part 1, Task 2------------------------------------
    #Define a range of t. Start @ 0 go to 10 (+a step) w/ a stepsize of step
t = np.arange(0, 10+ steps, steps)

    #Call function
func1 = cosine(t)

    #Make plot and then show it
plt.figure(figsize=(10, 7))
plt.plot(t,func1)
plt.grid()
plt.ylabel('cos(t)')
plt.xlabel('time')
plt.title('Part 1, Task 2 Cosine Plot')


#Part 2, Task 2------------------------------------
    #Define a range of t. Start @ -2 go to 5 (+a step) w/ a stepsize of step
t = np.arange(-2, 5+ steps, steps)

    #Step functions
y=stepFunc(t, 0, 1)
yNeg2=stepFunc(t, -1, -2)

    #Ramp functions
yRamp=rampFunc(t, 0, 1)
yNegRamp=rampFunc(t, 2, -2.5)

    #Make the plot for step functions
plt.figure(figsize=(10, 7))

    #Plot the 1st chart for step function
plt.subplot(1, 2, 1)
plt.plot(t, y)
#plt.plot(t,yNeg2) #Proof that I can change height and time start
plt.title('Part 2, Task 2 - Step Function Output')
plt.ylabel('Step Function Output')
plt.xlabel('Time (s)')
plt.grid()

    #Plot 2nd chart for ramp function
plt.subplot(1,2,2)
plt.plot(t, yRamp)
#plt.plot(t,yNegRamp) #Proof that I can change slope and time start
plt.title('Part 2, Task 2 - Ramp Function Output')
plt.ylabel('Ramp Function Output')
plt.xlabel('Time')
plt.grid()

#Part 2, Task 3------------------------------------
    #Define a range of t. Start @ -5 go to 10 (+a step) w/ a stepsize of step
t = np.arange(-5, 10+ steps, steps)

    #Get an array of the function to plot
    #Make my function!
ramp1 = rampFunc(t, 0, 1)
negRamp = rampFunc(t, 3, -1)
fiveStep = stepFunc(t, 3, 5)
negTwoStep = stepFunc(t, 6, -2)
negTwoRamp = rampFunc(t, 6, -2)

func2Plot = ramp1 + negRamp + fiveStep + negTwoStep + negTwoRamp
    #Ploting the functionToPlot for part 2, task 3


plt.figure(figsize=(10,7))
plt.plot(t, func2Plot)
plt.title('Part 2, Task 3')
plt.ylabel('Function Output')
plt.xlabel('Time')
plt.grid()

#Part 3, Task 1------------------------------------
    #Apply time reversal
reverseTimeFunction = timeReversal(func2Plot)

    #Ploting reverseTimeFunction
plt.figure(figsize=(10, 7))
plt.plot(t, reverseTimeFunction)
plt.ylabel('Function Output')
plt.xlabel('Time')
plt.title('Part 3, Task 1 - Time Reversal of Function')
plt.grid()

#Part3, Task 2------------------------------------
#tScale = np.arange(-5 + 4, 10 + steps + 4, steps)
tScale = timeShift(t,4)

    #Ploting f(t-4)
plt.figure(figsize=(10, 7))
plt.subplot(1,2,1)
plt.plot(tScale, func2Plot)
plt.ylabel('Function Output')
plt.xlabel('Time')
plt.title('Part 3, Task 2 - f(t-4)')
plt.grid()

plt.subplot(1,2,2)
plt.plot(tScale,reverseTimeFunction)
plt.xlabel("Time")
plt.title("Part 3, Task 2 - f(-t-4)")
plt.grid()

#Part 3, Task 3------------------------------------
    #Timescales!

    #Define a range of t. Start @ -5 go to 10 (+a step) w/ a stepsize of step
#time scale of t/2!
#steps = ((5+10)/1501)
#tScaleHalf = np.arange(-5, 5 + (1/150), (1/150))
t = np.arange(-5, 10+ steps, steps)
tScaleHalf = t * 0.5

    #Ploting f(0.5t)
plt.figure(figsize=(10, 7))
plt.subplot(1,2,1)
plt.plot(tScaleHalf, func2Plot)
plt.ylabel('Function Output')
plt.xlabel('Time')
plt.title('Part 3, Task 2 - f(t/2)')
plt.grid()

#time scale of 2t!
tScale2 = t * 2
#tScale2 = timeScale(t,2)

    #Ploting f(2t)
plt.subplot(1,2,2)
plt.plot(tScale2, func2Plot)
plt.ylabel('Function Output')
plt.xlabel('Time')
plt.title('Part 3, Task 2 - f(2t)')
plt.grid()

#Calculate and plot the diritive of func2Plot
func2PlotDir = np.diff(func2Plot)
tMod = np.arange(-5, 10, steps)
plt.figure(figsize=(10, 7))
plt.plot(tMod, func2PlotDir)
plt.ylabel('Function Output')
plt.xlabel('Time')
plt.title("Part 3, Task  - f'(t)")
plt.grid()

    #Show the plot! Uncomment to show plots
plt.show()
