#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 16:56:00 2019

@author: femap
"""

import numpy as np
import matplotlib.pyplot as plt
import math

# better with funcition
def fourierTransform(signal, num_samples, sample_rate):
    fft_output = np.fft.rfft(signal)
    magnitude_only = [np.sqrt(i.real**2 + i.imag**2)/len(fft_output) for i in fft_output]
    frequencies = [(i*1.0/num_samples)*sample_rate for i in range(num_samples//2+1)]
    return magnitude_only, frequencies

def findMax(mags, freqs, skipPoints=5):
    mags, freqs = mags[skipPoints::], freqs[skipPoints::]
    max_mag = max(mags)
    for index, mags in enumerate(mags):
        if max_mag == mags:
            max_index = index
            break
    return freqs[max_index]


def ave(signal):   
   ave=np.mean(signal)
   return ave


def rms(signal):
    #s = sum(signal)
    #average = s/len(signal)
    
    s2=0
    for i in signal:
        s2 += i**2
        #s2 += (i-average)**2
    rms=math.sqrt(s2/len(signal))
#    rms*= math.sqrt(2)
    #rms=math.sqrt(s2/len(signal))
    return rms 

def rm(signal):
    s = sum(signal)
    average = s/len(signal)
    s3=0
    for i in signal:
        s3 += (i-average)**2
    rm=math.sqrt(s3/len(signal))
#    rms*= math.sqrt(2)
    #rms=math.sqrt(s2/len(signal))
    return rm 

# import data
time, a = np.loadtxt("hull_forces.txt", skiprows=10000, delimiter="\t", usecols=(0,1), unpack=True)
time, b = np.loadtxt("hull_forces.txt", skiprows=10000, delimiter="\t", usecols=(0,2), unpack=True)
time, a1 = np.loadtxt("hull_forces.txt", skiprows=10000, delimiter="\t", usecols=(0,11), unpack=True)
time, b1 = np.loadtxt("hull_forces.txt", skiprows=10000, delimiter="\t", usecols=(0,12), unpack=True)
#Diameter = 0.0889,
#velocity =1, 
#pho =1000, 

PTC_d = a+b
PTC_l=(a1+b1)

plt.figure(figsize=(16,4))

ax1 = plt.subplot(1,2,1)
plt.plot(time, PTC_d)
ax1.plot(time, PTC_l)
#ax1.set_ylim(-0.002,0.002)
plt.grid( color = 'black',linestyle='--',linewidth = 0.5)

print ave(PTC_d)
print rms(PTC_l)
print ave(PTC_l)
print rm(PTC_l)

ax2 = plt.subplot(1,2,2)
sampleRate = 1/(time[1]-time[0])
mags, freqs = fourierTransform(PTC_d, len(PTC_d), sampleRate)
mags1, freqs1 = fourierTransform(PTC_l, len(PTC_l), sampleRate)
ax2.plot(freqs, mags)
ax2.plot(freqs1, mags1)
ax2.set_xlim(0,10)
ax2.set_ylim(0, 0.6)
maxFreq = findMax(mags, freqs)
#plt.plot([maxFreq, maxFreq], [-100, 100], linestyle="dashed", color="red")
plt.text(3.5, 0.035, "Freq={} Hz".format(round(maxFreq,3)),fontsize=16)
maxFreq1 = findMax(mags1, freqs1)
#plt.plot([maxFreq1, maxFreq1], [-100, 100], linestyle="dashed", color="green")
plt.text(3.5, 0.085, "Freq={} Hz".format(round(maxFreq1,3)),fontsize=16)
plt.grid( color = 'black',linestyle='-',linewidth = 0.5)

