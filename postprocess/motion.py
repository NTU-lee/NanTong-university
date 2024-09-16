#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import shapely.geometry as SG
import os
import pandas as pd
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

def rms(signal):
    #s = sum(signal)
    #average = s/len(signal)    
    s2=0
    for i in signal:
        s2 += i**2
        #s2 += (i-average)**2
    rms=math.sqrt(s2/len(signal))
    rms*= math.sqrt(2)
    #rms=math.sqrt(s2/len(signal))
    return rms

# import data
time, z = np.loadtxt("hull_motion.txt", skiprows=8000, delimiter="\t", usecols=(0,3), unpack=True)


Diameter = 0.0889,
velocity =0.5, 
pho =1000, 

z= z/ Diameter

fi = plt.figure(figsize=(14,2))
font={'family':'Times New Roman',
     'style':'italic',
    'weight':'normal',
      'color':'black',
      'size':18}
bwith = 2 #边框宽度设置为2

ax1 = plt.subplot(1,2,1)
ax1 = plt.gca()#获取边框
ax1.spines['bottom'].set_linewidth(bwith)
ax1.spines['left'].set_linewidth(bwith)
ax1.spines['top'].set_linewidth(bwith)
ax1.spines['right'].set_linewidth(bwith)
plt.grid( color = 'black',linestyle='-',linewidth = 0.2)
plt.plot(time, z, "-", c="b",linewidth=1, label= 'present')
#my_y_ticks = np.arange(-2,2.1,0.5)
#plt.yticks(my_y_ticks,family='Times New Roman',weight='normal',size=18)
#my_x_ticks = np.arange(15,61,5)
#plt.xticks(my_x_ticks,family='Times New Roman',weight='normal',size=18)
#plt.xlim(14,60)
#plt.ylim(-2.1, 2.1)
#plt.title("x/D=1.06",family='Times New Roman', style='italic',weight='normal',size=18)
plt.ylabel(r'$A*$', family='Times New Roman', style='italic',weight='normal',size=18)
#plt.savefig("smooth_vs_exp/1.06velocity_u.png", bbox_inches = 'tight', dpi=300)


ax2 = plt.subplot(1,2,2)
ax2 = plt.gca()#获取边框
ax2.spines['bottom'].set_linewidth(bwith)
ax2.spines['left'].set_linewidth(bwith)
ax2.spines['top'].set_linewidth(bwith)
ax2.spines['right'].set_linewidth(bwith)
plt.grid( color = 'black',linestyle='-',linewidth = 0.3)
sampleRate = 1/(time[1]-time[0])
mags, freqs = fourierTransform(z, len(z), sampleRate)
plt.plot(freqs, mags,c="b",linewidth=1, label= 'present')
my_y_ticks = np.arange(0,0.61,0.2)
plt.yticks(my_y_ticks,family='Times New Roman',weight='normal',size=18)
my_x_ticks = np.arange(0,6.1,1)
plt.xticks(my_x_ticks,family='Times New Roman',weight='normal',size=18)
plt.xlim(0,6)
plt.ylim(0, 0.6)
#plt.title("x/D=1.06",family='Times New Roman', style='italic',weight='normal',size=18)
#plt.ylabel(r'$\ \overline{u}/U_\infty$', family='Times New Roman', style='italic',weight='normal',size=18)
maxFreq = findMax(mags, freqs)
#plt.plot([maxFreq, maxFreq], [-10, 10], linestyle="dashed", color="red")
plt.text(1.5, 0.2, "Freq={} Hz".format(round(maxFreq,3)),fontsize=12)

plt.savefig("*.png", dpi=300)

print rms(z)

