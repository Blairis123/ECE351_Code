################################################################
#
# Owen Blair
# ECE351-52
# Lab #9
# 10/28/2021
#
################################################################
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import scipy.fftpack as fft
import math as m

"""User defined fast fourier transform funnction.
INPUTS:
    X, a function array
    fs, frequency of sampleing rate
OUTPUTS:
    output, an array containig various information
    output[0], a fourier transform of the input function
    output[1], same as output[0] but zero frewuency is the center of spectrum
    output[2], frequency array coresponding to fourier output
    output[3], array of fourier magnatudes
    output[4], array of fourier angles
"""
def FFT(X, fs):
    
    #Length of input array
    n = len(X)
    
    #Preform fast fourier transorm
    X_fft = fft.fft(X)
        
    #shift zero frequency to center of the spectrium
    X_fft_shift = fft.fftshift(X_fft)
    
    # Calculate frequnecies for output. fa is sampling frequency
    X_freq = np.arange(-n/2, n/2) * fs / n
    #zm_freq = np.arrange(-n/4, n/4) * fs/(n/2)
    
    #Calculate magnatude and phase
    X_mag = np.abs(X_fft_shift)/n
    X_phi = np.angle(X_fft_shift)
    
    output = [X_fft, X_fft_shift, X_freq, X_mag, X_phi]    
    return output

"""CleanFFT
CleanFFT is like the user defined FFT functio, but if a given magnatude is
    less than 1e-10 the coresponding phase angle will be set to zero.
DEPENDENCIES:
    FFT(X, fs), Function that returns an array containing the fast fourier
    transform of an array X with a given sample rate fs.
INPUTS:
    X, an array to apply the fourier transform on
    fs, sampleing frequency for fourier transform
OUTPUTS:
    output, an array containing cleaned up arrays of X_mag, X_phi and freq
"""
def FFTClean(X,fs):
    XArry = FFT(X, fs)
    useableArry = [XArry[2], XArry[3], XArry[4]]
    for i in range(0, len(useableArry)-1):
        if (useableArry[1][i] <= 0.000000001):
            useableArry[2][i] = 0
    
    return useableArry

"""
Fourer transform!
"""
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
plot_fft is for ploting stuff!
"""
def plot_fft(title, x, X_mag, X_phi, freq, t, zmInt):
    
    #Calculate the zoomed in data for magnatude and frequency
    zm_mag = [];
    zm_mag_freq = [];
    for i in range(0, len(freq)-1):
        if ((freq[i]>=-zmInt) and (freq[i]<=zmInt)):
            zm_mag.append(X_mag[i])
            zm_mag_freq.append(freq[i])
            
    zm_phi = [];
    zm_phi_freq = [];
    for i in range(0, len(freq)-1):
        if ((freq[i]>=-zmInt) and (freq[i]<=zmInt)):
            zm_phi.append(X_phi[i])
            zm_phi_freq.append(freq[i])
    
    fig3 = plt.figure(constrained_layout=True)
    gs = fig3.add_gridspec(3, 2)
    f3_ax1 = fig3.add_subplot(gs[0, :])
    f3_ax1.set_title('User-Defined FFT of '+ title)
    f3_ax1.set_xlabel('time')
    f3_ax1.plot(t,x)
    plt.grid()

    
    f3_ax2 = fig3.add_subplot(gs[1, 0])
    f3_ax2.set_ylabel('Magnatude X(f)')
    f3_ax2.stem(freq, X_mag)
    plt.grid()
    
    f3_ax3 = fig3.add_subplot(gs[1, 1])
    f3_ax3.set_title('Better magnatude pic')
    f3_ax3.stem(zm_mag_freq, zm_mag)
    plt.grid()
    
    f3_ax4 = fig3.add_subplot(gs[2, 0])
    f3_ax4.set_ylabel('Angle of X(f)')
    f3_ax4.set_xlabel('f [Hz]')
    f3_ax4.stem(freq, X_phi)
    plt.grid()
    
    f3_ax5 = fig3.add_subplot(gs[2, 1])
    f3_ax5.set_title('Better angle of X(f)')
    f3_ax5.set_xlabel('f [Hz]')
    f3_ax5.stem(zm_phi_freq, zm_phi)
    plt.grid()

"""
Used for lab 8 plotting
"""
def plot_fft2(title, x, X_mag, X_phi, freq, t):
    
    
    fig3 = plt.figure(constrained_layout=True)
    gs = fig3.add_gridspec(3, 2)
    f3_ax1 = fig3.add_subplot(gs[0, :])
    f3_ax1.set_title('User-Defined FFT of '+ title)
    f3_ax1.set_xlabel('time')
    f3_ax1.plot(t,x)
    plt.grid()
    
    f3_ax2 = fig3.add_subplot(gs[1, :])
    f3_ax2.set_ylabel('Magnatude X(f)')
    f3_ax2.stem(freq, X_mag)
    plt.grid()
    
    f3_ax4 = fig3.add_subplot(gs[2, :])
    f3_ax4.set_ylabel('Angle of X(f)')
    f3_ax4.set_xlabel('f [Hz]')
    f3_ax4.stem(freq, X_phi)
    plt.grid()
#------------------END USER DEFINED FUNCTIONS!

    #Define step size
steps = 1e-2

    #t for part 1
start = 0
stop = 2
    #Define a range of t_pt1. Start @ 0 go to 20 (+a step) w/
    #a stepsize of step
t = np.arange(start, stop, steps)

# Sampling frquency for lab
fs = 100

# Task 1 input function, FFT, FFTClean
cos_2_pi = np.cos(2* np.pi * t)
FFTcos_2_pi = FFT(cos_2_pi, fs)
FFTCleanTask1 = FFTClean(cos_2_pi, fs)

# Task 2 input function, FFT, FFTClean
sin_2_pi_5 = 5 * np.sin(2 * np.pi * t)
FFTsin_2_pi_5 = FFT(sin_2_pi_5, fs)
FFTCleanTask2 = FFTClean(sin_2_pi_5, fs)

# Task 3  input function, FFT, FFTClean
task3Func = 2* np.cos((4*np.pi*t) - 2) + (np.sin((12*np.pi*t) + 3) )**2
FFT_task3Func = FFT(task3Func, fs)
FFTCleanTask3 = FFTClean(task3Func, fs)

#Fourier plot of the previous signal from lab 8
t2 = np.arange(0, 16, steps)
x_15 = xFourier(t2, 8, 15)

FFT_Lab8 = FFT(x_15, fs)

#Make the plots using the function!!!
plot_fft("Task 1 x(t)", cos_2_pi, FFTcos_2_pi[3], FFTcos_2_pi[4], FFTcos_2_pi[2], t, 2)
plot_fft("Task 2 x(t)", sin_2_pi_5, FFTsin_2_pi_5[3], FFTsin_2_pi_5[4], FFTsin_2_pi_5[2], t, 2)
plot_fft("Task 3 x(t)", task3Func, FFT_task3Func[3], FFT_task3Func[4], FFT_task3Func[2], t, 15)

# Plot the noise reduced versions of the functions
plot_fft("Task 1 x(t) Clean", cos_2_pi, FFTCleanTask1[1], FFTCleanTask1[2], FFTCleanTask1[0], t, 2)
plot_fft("Task 2 x(t) Clean", sin_2_pi_5, FFTCleanTask2[1], FFTCleanTask2[2], FFTCleanTask2[0], t, 2)
plot_fft("Task 3 x(t) Clean", task3Func, FFTCleanTask3[1], FFTCleanTask3[2], FFTCleanTask3[0], t, 15)

plot_fft("Signal from Lab 8", x_15, FFT_Lab8[1], FFT_Lab8[2], FFT_Lab8[0], t2, 1e-15)