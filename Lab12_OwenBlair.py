################################################################
#
# Owen Blair
# ECE351-52
# Lab #12
# 12/9/2021
#
################################################################

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import scipy.fftpack as fft
import math as m
import pandas as pd

# The following package is not included with the Anaconda distribution
# and needed to be installed separately. The control package also has issues
# working on macs and a PC or a linux distribution is needed
import control as con

#-----------FUNCTIONS!!!!!!!----------------------------

"""
This is a faster version of matplot.pyplot.stem() plotting. This is from the
lab handout and is provided from the TA.

MODIFIED FROM LAB HANDOUT BY OWEN BLAIR 11/22/2021
    This addition of the return statement allows the programmer to set an axis
    from the return of this function. I was having issues with the code from
    the lab handout not working. The error I was getting was
    
    ValueError: Single argument to subplot must be a three-digit integer, not
    AxesSubplot(0.125,0.536818;0.775x0.343182)"
    
    I would also get random "list has no attribute 'min'" errors with:
    ax.set_ylim ([1.05 * y.min(), 1.05 * y.max()])
    
    After a bit of googleing, sometimes there are issues with the list.min()
    function. No issues yet with the min(list) function change ..... yet.....
"""
def make_stem(ax ,x,y,color='k',style='solid',label='',linewidths =1.5 ,** kwargs):
    ax.axhline(x[0],x[-1],0, color='r')
    ax.vlines(x, 0 ,y, color=color , linestyles=style , label=label , linewidths= linewidths)
    
    # This has been modified
    ax.set_ylim ([1.05 * min(y), 1.05 * max(y)])
    
    # This line has been added
    return ax


"""User defined fast fourier transform funnction.
INPUTS:
    X, a function array
    fs, frequency of sampleing rate
    
OUTPUTS:
    X_freq, frequency array coresponding to FFT
    X_mag, array of FFT magnatudes from fast fourier transform
    X_phi, array of FFT angles from fast fourier transform

NOTES:
    Any magnatude that is less than 1e-10 will also have a coresponding angle
    of zero. This serves to "clean up" the phase angle plot.
"""
def FFT(X, fs):
    
    # Length of input array
    n = len(X)
    
    # Preform fast fourier transorm
    X_fft = fft.fft(X)
        
    """
    Will not use shifted because the frequencies that are needed will be real
    and won't have a negative value!!!!!
    
    # shift zero frequency to center of the spectrium
    X_fft_shift = fft.fftshift(X_fft)
    """
    
    # Calculate frequnecies for output. fs is sampling frequency
    X_freq = np.arange(0, n) * fs / n
    
    # Calculate magnatude and phase
    X_mag = np.abs(X_fft)/n
    X_phi = np.angle(X_fft)
    
    # Clean up the phase array!
    for i in range(len(X_phi)):
        if ( X_mag[i] < 1e-10):
            X_phi[i] = 0
    
    # Return values!
    return X_freq, X_mag, X_phi


#---------------------------------------------------------------------


# Import signal
df = pd.read_csv('NoisySignal.csv')

t = df['0'].values
sensor_sig = df['1'].values

"""
Uncomment the follwoing to plot the input signal
"""
# Plott the signal!
plt.figure(figsize = (10, 7))
plt.plot(t, sensor_sig)
plt.grid()
plt.title("Input signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplatude (V)")
plt.show()



"""
This part uses the fast fourier transform to identify the noise and the
signal's frequencies and magnatudes (Modified from lab 9)
"""

    # Set sampling frequency
fs=1e6

lowFrq = []
lowMag = []

dataFrq = []
dataMag = []

highFrq = []
highMag = []

    # Run through signal with FFT
X_freq, X_mag, X_phi = FFT(sensor_sig, fs)

for i in range(len(X_freq)):
    
    if (X_freq[i] < 1.8e3):
        lowFrq.append(X_freq[i])
        lowMag.append(X_mag[i])
    
    if ((X_freq[i] <= 2e3) and (X_freq[i] >= 1.8e3)):
        dataFrq.append(X_freq[i])
        dataMag.append(X_mag[i])
        
    if (X_freq[i] > 2e3):
        highFrq.append(X_freq[i])
        highMag.append(X_mag[i])
    
    # Plot FFT stuff!
gridSize = (5,1)
fig = plt.figure(figsize = (10, 10), constrained_layout = True)

    # Magnatude of input signal
inputFFTMagAx = plt.subplot2grid(gridSize, (0,0))# Make axis object to modify
inputFFTMagAx = make_stem(inputFFTMagAx, X_freq, X_mag)
inputFFTMagAx.set_title("Input Signal Magnatudes")
inputFFTMagAx.set_ylabel("Magnatude (V)")
inputFFTMagAx.set_xlabel("Frequency (Hz)")
inputFFTMagAx.set_xlim(0, 9e5)
inputFFTMagAx.grid()

    # Low ( < 1.8 kHz) frequency zoom
lowFreqAx = plt.subplot2grid(gridSize, (1,0))
lowFreqAx = make_stem(lowFreqAx, lowFrq, lowMag)
lowFreqAx.set_title("Low Frequency Noise Magnatudes")
lowFreqAx.set_ylabel("Magnatude (V)")
lowFreqAx.set_xlabel("Frequency (Hz)")
#inputFFTMagAx.set_xlim(0, 1.8e3)
lowFreqAx.grid()

    # Data frequency (1.8 kHz < frq < 2.0 kHz)
dataFreqAx = plt.subplot2grid(gridSize, (2,0))
dataFreqAx = make_stem(dataFreqAx, dataFrq, dataMag)
dataFreqAx.set_title("Data Signal Magnatudes")
dataFreqAx.set_ylabel("Magnatude (V)")
dataFreqAx.set_xlabel("Frequency (Hz)")
#inputFFTMagAx.set_xlim(1.8e3, 2e3)
dataFreqAx.grid()

    # High frequency (between 2 kHz and 50 kHz) magnatudes
highFreqAx = plt.subplot2grid(gridSize, (3,0))
highFreqAx = make_stem(highFreqAx, highFrq, highMag)
highFreqAx.set_title("High Frequency Noise Magnatudes")
highFreqAx.set_ylabel("Magnatude (V)")
highFreqAx.set_xlabel("Frequency (Hz)")
inputFFTMagAx.set_xlim(2e3, 9e5)
highFreqAx.grid()

plt.show()


# FILTER!-------------------------------------------------------------------
    # Numerator and denomenator for transfer function H(s)
# The following was from a previous filter that was complacated but had a bode
# plot that sugested that it should work but still had a spike at 10e6 above
# 0.05 V

#filterNUM = [9e8, 0, 0]
#filterDEN = [18, 2.7e6, 2.34e10, 3.75e14, 1.5625e18]

"""
    The follwing is calculated using the series bandpass filter
    from ECE-212 (Karen's circuits class) and is used to calculate
    component values
"""
    # Fist number is in Hz but is multiplied by 2pi for rads/sec
bandwidth = 800 * (2*np.pi)
centerFrq = 1.9e3 * (2 * np.pi)

#R = 10
#L = R / bandwidth
#C = bandwidth / (centerFrq**2 * R)
#print(R, L, C)

R = 10
L = 1.989e-3
C = 3.527e-6

filterNUM = [0, R/L, 0]
filterDEN = [1, R/L, 1/(L*C)]


    # Transfer funciton object, num and den are defined above
H_s = con.TransferFunction(filterNUM, filterDEN)

# Define an omega for transfer function
    # Define step size
steps = 1

    # w for part 1
start = 0
stop = 9e5
    #Define a range of w, with a stepsize of step
w = np.arange(start, stop, steps)

# Make and show Bode plots!
    # Make the Bode plot
plt.figure(figsize = (10, 7))
plt.title("All Frequencies")
Mag, Phi, bodeW = con.bode(H_s, w * 2 * np.pi, dB=True, Hz=True, deg=True, plot=True)

plt.figure(figsize = (10, 7))
plt.title("Frequencies Below Data")
Mag, Phi, bodeW = con.bode(H_s, np.arange(1, 1.8e3, 10)* 2 * np.pi, dB=True, Hz=True, deg=True, plot=True)

plt.figure(figsize = (10,7))
Mag, Phi, bodeW = con.bode(H_s, np.arange(1.8e3, 2e3, 1)* 2 * np.pi, dB=True, Hz=True, deg=True, plot=True)

plt.figure(figsize = (10,7))
Mag, Phi, bodeW = con.bode(H_s, np.arange(2e3, 1e6, 10)* 2 * np.pi, dB=True, Hz=True, deg=True, plot=True)

    
# Filter the signal!
    # Move transfer function into z-domain
numZ, denZ = sig.bilinear(filterNUM, filterDEN,fs)

    # Pass signal through filter
filteredSignal = sig.lfilter(numZ, denZ, sensor_sig)

# Plott the filtered signal!
plt.figure(figsize = (10, 7))
plt.title("Filtered Sensor Signal")
plt.ylabel("Filtered Signal Amplatude")
plt.xlabel("Time (s)")
plt.plot(t,filteredSignal)
plt.grid()

plt.show()

ratioMult = 1

    # Do some ploting!
plt.figure(figsize=(20*ratioMult, 10*ratioMult))
plt.figure(constrained_layout=True)
plt.subplot(2,1,1)
plt.title("Filtered Vs. Unfiltered Signal Comparison")
plt.ylabel("Filtered Signal Amplatude")
plt.xlabel("Time (s)")
plt.plot(t,filteredSignal)
plt.grid()

plt.subplot(2,1,2)
plt.plot(t, sensor_sig)
plt.grid()
plt.title("Unfiltered Input Signal")
plt.xlabel("Time (s)")
plt.ylabel("Unfiltered Signal Amplatude")

plt.show()



#Put filtered signal through FFT-------------------------------------------
    # Run through signal with FFT
filteredFreq, filteredMag, filteredPhi = FFT(filteredSignal, fs)

        
"""
    The following uses the make stem function and subplot2grid functions
    to make a figure with multiple subplots. The variable gridSize controls
    how the subplots are arranged within the figure. The figsize variable
    controls the ratio of the figure size.
"""


# Ploting filtered signal through FFT
    # Plot FFT stuff!
gridSize = (4,1) # This is the size of the grid
fig = plt.figure(figsize = (12, 8), constrained_layout = True)


    # All frequency magnatudes
filteredFreqAx = plt.subplot2grid(gridSize, (0,0))
filteredFreqAx = make_stem(filteredFreqAx, filteredFreq, filteredMag)
filteredFreqAx.set_title("All Filtered Frequency Noise Magnatudes")
filteredFreqAx.set_ylabel("Magnatude (V)")
filteredFreqAx.set_xlabel("Frequency (Hz)")
filteredFreqAx.set_xlim(0, 9e5)
filteredFreqAx.grid()

    # Low ( < 1.8 kHz) frequency zoom
filteredLowFreqAx= plt.subplot2grid(gridSize, (1,0))
filteredLowFreqAx= make_stem(filteredLowFreqAx, filteredFreq, filteredMag)
filteredLowFreqAx.set_xlim(0,1.8e3)
filteredLowFreqAx.set_title("Filtered Low Frequency Noise Magnatudes")
filteredLowFreqAx.set_ylabel("Magnatude (V)")
filteredLowFreqAx.set_xlabel("Frequency (Hz)")
filteredLowFreqAx.grid()

    # Data frequency (1.8 kHz < frq < 2.0 kHz)
filteredDataFreqAx = plt.subplot2grid(gridSize, (2,0))
filteredDataFreqAx = make_stem(filteredDataFreqAx, filteredFreq, filteredMag)
filteredDataFreqAx.set_xlim(1.79e3, 2e3)
filteredDataFreqAx.set_title("Filtered Data Signal Magnatudes")
filteredDataFreqAx.set_ylabel("Magnatude (V)")
filteredDataFreqAx.set_xlabel("Frequency (Hz)")
filteredDataFreqAx.grid()

    # High frequency (above 2kHz) magnatudes
filteredHighFreqAx = plt.subplot2grid(gridSize, (3,0))
filteredHighFreqAx = make_stem(filteredHighFreqAx, filteredFreq, filteredMag)
filteredHighFreqAx.set_title("Filtered High Frequency Noise Magnatudes")
filteredHighFreqAx.set_ylabel("Magnatude (V)")
filteredHighFreqAx.set_xlabel("Frequency (Hz)")
filteredHighFreqAx.set_xlim(2.1e3, 9e5)
filteredHighFreqAx.grid()

plt.show()

