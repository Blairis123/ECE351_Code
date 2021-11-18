################################################################
#
# Owen Blair
# ECE351-52
# Lab #10
# 11/11/2021
#
################################################################
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import scipy.fftpack as fft
import math as m

# The following package is not included with the Anaconda distribution
# and needed to be installed separately. The control package also has issues
# working on macs and a PC or a linux distribution is needed
import control as con


#---------PART 1-----------------------------------#
    #Define step size
steps = 1

    #t for part 1
start = 1e3
stop = 1e6
    #Define a range of w, with a stepsize of step
w = np.arange(start, stop, steps)

#------------------------------TASK 1, Part 1---------------#
    # Data from RLC circuit
R=1e3
L=27e-3
C=100e-9
    # Magnatude function and phase function. Keep in mind that
    # np.arctan will return a value in radians
H_mag = (w/(R*C)) / np.sqrt( w**4 + ( (1/(R*C))**2 - (2/(L*C))) * (w**2) + (1/(L*C))**2)
H_phi = (np.pi/2) - np.arctan( (w/(R*C)) / (-1*(w**2) + (1/(L*C))) )

    # Convert to Db and deg
H_magDb = 20 * np.log10(H_mag)
H_phiDeg = (180/np.pi) * H_phi


    # loop to fix the jump in the phase plot
for i in range(0,len(H_phiDeg)):
    #Do stuff
    if(H_phiDeg[i]>90):
        H_phiDeg[i] = H_phiDeg[i] - 180

    # Plot the magnatude
plt.figure(figsize=(10, 7))
plt.figure(constrained_layout=True)
plt.subplot(2,1,1)
plt.title("Bode Plot of H(jw)")
plt.ylabel("|H(jw)| dB")
plt.semilogx(w, H_magDb)
plt.grid()

plt.subplot(2,1,2)
plt.ylabel("Phase in Deg")
plt.xlabel("rad/s")
plt.semilogx(w, H_phiDeg)
plt.grid()

plt.show()

    # Check phase with mutiple approches
H_phi2 = (np.pi/2) - np.arctan2( (w/(R*C)), (-1*(w**2) + (1/(L*C))) )
H_phiDeg2 = (180/np.pi) * H_phi2

    #Plot to make sure!
plt.figure(figsize=(10,7))
plt.figure(constrained_layout=True)
plt.subplot(2,1,1)
plt.title("np.arctan with Loop Adjustment VS. np.arctan2")
plt.ylabel("np.arctan (Deg)")
plt.semilogx(w, H_phiDeg)
plt.grid()

plt.subplot(2,1,2)
plt.ylabel("np.arctan2 (Deg)")
plt.semilogx(w, H_phiDeg2)
plt.xlabel("rad/s")
plt.grid()

plt.show()

#-------------------------------TASK 2, PART 1---------------#
    # Transfer function numerator and den.
num = [(1/(R*C)),0]
den = [1,1/(R*C),1/(L*C)]

    # Need a name change for omega array
w1 = w
sigW, sigBodeMag, sigBodePhi = sig.bode(([(1/(R*C)),0], [1,1/(R*C),1/(L*C)]), w=w1)

    # Start plot!
plt.figure(figsize=(10,7))
plt.figure(constrained_layout=True)
plt.subplot(2,1,1)
plt.title("scipy.signal.bode Bode Plot of H(s)")
plt.ylabel("dB")
plt.semilogx(w, sigBodeMag)
plt.grid()

plt.subplot(2,1,2)
plt.ylabel("deg")
plt.semilogx(w, sigBodePhi)
plt.xlabel("rad/s")
plt.grid()

plt.show()
#-------------------------------TASK 3, PART 1---------------#

    # Transfer funciton object, num and den are defined in previous task
H_s = con.TransferFunction(num,den)
    # Make the Bode plot
Mag, Phi, bodeW = con.bode(H_s, w, dB=True, Hz=True, deg=True, plot=True)

# This code was suposed to create a bodie plot from the output of con.bode
# that would better match the rest of the generated plots. This was abandoned
# in  persuit of finishing the lab in a reasonable time
"""
    # Adjusting the Phi from con.bode
plt.figure(figsize=(10,7))
plt.figure(constrained_layout=True)
plt.subplot(2,1,1)
plt.title("Bode Plot of H(s)")
plt.ylabel("Magnitude (dB)")
plt.semilogx(bodeW, Mag)
plt.grid()

plt.subplot(2,1,2)
plt.ylabel("Phase (deg)")
plt.xlabel("Frequency (Hz)")
plt.semilogx(bodeW, Phi)
plt.grid()

plt.show()
"""
#-------------------------------TASK 1, PART 2---------------#

    # Sampleing size
fs = 5e10
    #Define step size
steps = 1/fs

    #t for part 1
start = 0
stop = 1e-2
    #Define a range of t, with a stepsize of step
t = np.arange(start, stop, steps)

    # Make input signal array!
x_t = np.cos(2*np.pi*100*t) + np.cos(2*np.pi*3024*t) + np.sin(2*np.pi*5e4)

    # Move transfer function into z-domain
numZ, denZ = sig.bilinear(num,den,fs)

    # Pass signal through filter
y_t = sig.lfilter(numZ, denZ, x_t)

    # Do some ploting!
plt.figure(figsize=(10,7))
plt.figure(constrained_layout=True)
plt.subplot(2,1,1)
plt.title("x(t) Through Filter H(s)")
plt.ylabel("Filter Output y(t)")
plt.plot(t,y_t)
plt.grid()

plt.subplot(2,1,2)
plt.ylabel("Orginal Signal x(t)")
plt.xlabel("seconds")
plt.plot(t, x_t)
plt.grid()

plt.show()


