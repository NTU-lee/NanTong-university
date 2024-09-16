
#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 16:56:00 2019

@author: femap
"""

import numpy as np
import matplotlib.pyplot as plt
import math


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

    #rms=math.sqrt(s2/len(signal))
    return rms


# import data
time, z = np.loadtxt("ppp.dat", skiprows=1, delimiter=",", usecols=(0, 1), unpack=True)
time, z1 = np.loadtxt("ppp.dat", skiprows=1, delimiter=",", usecols=(0, 1), unpack=True)
time, z2 = np.loadtxt("ppp.dat", skiprows=1, delimiter=",", usecols=(0, 2), unpack=True)
time, z3 = np.loadtxt("ppp.dat", skiprows=1, delimiter=",", usecols=(0, 3), unpack=True)
time, z4 = np.loadtxt("ppp.dat", skiprows=1, delimiter=",", usecols=(0, 4), unpack=True)
time, z5 = np.loadtxt("ppp.dat", skiprows=1, delimiter=",", usecols=(0, 5), unpack=True)
time, z6 = np.loadtxt("ppp.dat", skiprows=1, delimiter=",", usecols=(0, 6), unpack=True)
time, z7 = np.loadtxt("ppp.dat", skiprows=1, delimiter=",", usecols=(0, 7), unpack=True)
time, z8 = np.loadtxt("ppp.dat", skiprows=1, delimiter=",", usecols=(0, 8), unpack=True)
time, z9 = np.loadtxt("ppp.dat", skiprows=1, delimiter=",", usecols=(0, 9), unpack=True)
time, z10 = np.loadtxt("ppp.dat", skiprows=1, delimiter=",", usecols=(0, 10), unpack=True)
time, z11 = np.loadtxt("ppp.dat", skiprows=1, delimiter=",", usecols=(0, 11), unpack=True)
time, z12 = np.loadtxt("ppp.dat", skiprows=1, delimiter=",", usecols=(0, 12), unpack=True)
time, z13 = np.loadtxt("ppp.dat", skiprows=1, delimiter=",", usecols=(0, 13), unpack=True)
time, z14 = np.loadtxt("ppp.dat", skiprows=1, delimiter=",", usecols=(0, 14), unpack=True)
time, z15 = np.loadtxt("ppp.dat", skiprows=1, delimiter=",", usecols=(0, 15), unpack=True)
time, z16 = np.loadtxt("ppp.dat", skiprows=1, delimiter=",", usecols=(0, 16), unpack=True)
time, z17 = np.loadtxt("ppp.dat", skiprows=1, delimiter=",", usecols=(0, 17), unpack=True)
time, z18 = np.loadtxt("ppp.dat", skiprows=1, delimiter=",", usecols=(0, 18), unpack=True)
time, z19 = np.loadtxt("ppp.dat", skiprows=1, delimiter=",", usecols=(0, 19), unpack=True)
time, z20 = np.loadtxt("ppp.dat", skiprows=1, delimiter=",", usecols=(0, 20), unpack=True)
time, z21 = np.loadtxt("ppp.dat", skiprows=1, delimiter=",", usecols=(0, 21), unpack=True)
time, z22 = np.loadtxt("ppp.dat", skiprows=1, delimiter=",", usecols=(0, 22), unpack=True)
time, z23 = np.loadtxt("ppp.dat", skiprows=1, delimiter=",", usecols=(0, 23), unpack=True)
time, z24 = np.loadtxt("ppp.dat", skiprows=1, delimiter=",", usecols=(0, 24), unpack=True)
time, z25 = np.loadtxt("ppp.dat", skiprows=1, delimiter=",", usecols=(0, 25), unpack=True)
time, z26 = np.loadtxt("ppp.dat", skiprows=1, delimiter=",", usecols=(0, 26), unpack=True)
time, z27 = np.loadtxt("ppp.dat", skiprows=1, delimiter=",", usecols=(0, 27), unpack=True)
time, z28 = np.loadtxt("ppp.dat", skiprows=1, delimiter=",", usecols=(0, 28), unpack=True)
time, z29 = np.loadtxt("ppp.dat", skiprows=1, delimiter=",", usecols=(0, 29), unpack=True)
time, z30 = np.loadtxt("ppp.dat", skiprows=1, delimiter=",", usecols=(0, 30), unpack=True)
time, z31 = np.loadtxt("ppp.dat", skiprows=1, delimiter=",", usecols=(0, 31), unpack=True)

plt.figure(figsize=(12,3))

grid = plt.GridSpec(1,2, hspace=0)


ax1 = plt.subplot(1,2,1)

plt.plot(time, z, color="royalblue",)
plt.plot(time, z30, color="blue",)


plt.xlabel("Time(s)", fontsize=12)
plt.ylabel("A*", fontsize=12)
plt.title("")

ax2 = plt.subplot(1,2,2)

sampleRate = 1/(time[1]-time[0])

mags, freqs = fourierTransform(z, len(z), sampleRate)
ax2.plot(freqs, mags)
ax2.set_xlim(0,3)
ax2.set_ylim(0,0.6)


my_y_ticks = np.arange(0,0.61,0.2)
plt.yticks(my_y_ticks)

maxFreq = findMax(mags, freqs)
'''plt.plot([maxFreq, maxFreq], [-10, 10], linestyle="dashed", color="red")'''
plt.text(1.2, 0.28, "{}".format(round(maxFreq,3)),fontsize=12)

plt.xlabel("Frequency(Hz)", fontsize=12)
'''plt.ylabel("Amplitude", fontsize=12)'''
plt.plot(freqs, mags, color="royalblue",)

plt.title("")
plt.savefig("time.png", dpi=300)


rra1=sum((z-np.average(z))*( z1-np.average(z1)))
rrr1=np.sqrt(sum(z*z-np.average(z)* np.average(z))* sum(z1*z1-np.average(z1)* np.average(z1)))
r1=rra1/rrr1
print r1
rra2=sum((z-np.average(z))*( z2-np.average(z2)))
rrr2=np.sqrt(sum(z*z-np.average(z)* np.average(z))* sum(z2*z2-np.average(z2)* np.average(z2)))
r2=rra2/rrr2
print r2
rra3=sum((z-np.average(z))*( z3-np.average(z3)))
rrr3=np.sqrt(sum(z*z-np.average(z)* np.average(z))* sum(z3*z3-np.average(z3)* np.average(z3)))
r3=rra3/rrr3
print r3
rra4=sum((z-np.average(z))*( z4-np.average(z4)))
rrr4=np.sqrt(sum(z*z-np.average(z)* np.average(z))* sum(z4*z4-np.average(z4)* np.average(z4)))
r4=rra4/rrr4
print r4
rra5=sum((z-np.average(z))*( z5-np.average(z5)))
rrr5=np.sqrt(sum(z*z-np.average(z)* np.average(z))* sum(z5*z5-np.average(z5)* np.average(z5)))
r5=rra5/rrr5
print r5
rra6=sum((z-np.average(z))*( z6-np.average(z6)))
rrr6=np.sqrt(sum(z*z-np.average(z)* np.average(z))* sum(z6*z6-np.average(z6)* np.average(z6)))
r6=rra6/rrr6
print r6
r7=(sum((z-np.average(z))*( z7-np.average(z7))))/(np.sqrt(sum(z*z-np.square(np.average(z)))* sum(np.square(z7)-np.square(np.average(z7)))))
print r7
r8=(sum((z-np.average(z))*( z8-np.average(z8))))/(np.sqrt(sum(z*z-np.square(np.average(z)))* sum(np.square(z8)-np.square(np.average(z8)))))
print r8
r9=(sum((z-np.average(z))*( z9-np.average(z9))))/(np.sqrt(sum(z*z-np.square(np.average(z)))* sum(np.square(z9)-np.square(np.average(z9)))))
print r9
r10=(sum((z-np.average(z))*( z10-np.average(z10))))/(np.sqrt(sum(z*z-np.square(np.average(z)))* sum(np.square(z10)-np.square(np.average(z10)))))
print r10
r11=(sum((z-np.average(z))*( z11-np.average(z11))))/(np.sqrt(sum(z*z-np.square(np.average(z)))* sum(np.square(z11)-np.square(np.average(z11)))))
print r11
r12=(sum((z-np.average(z))*( z12-np.average(z12))))/(np.sqrt(sum(z*z-np.square(np.average(z)))* sum(np.square(z12)-np.square(np.average(z12)))))
print r12
r13=(sum((z-np.average(z))*( z13-np.average(z13))))/(np.sqrt(sum(z*z-np.square(np.average(z)))* sum(np.square(z13)-np.square(np.average(z13)))))
print r13
r14=(sum((z-np.average(z))*( z14-np.average(z14))))/(np.sqrt(sum(z*z-np.square(np.average(z)))* sum(np.square(z14)-np.square(np.average(z14)))))
print r14
r15=(sum((z-np.average(z))*( z15-np.average(z15))))/(np.sqrt(sum(z*z-np.square(np.average(z)))* sum(np.square(z15)-np.square(np.average(z15)))))
print r15
r16=(sum((z-np.average(z))*( z16-np.average(z16))))/(np.sqrt(sum(z*z-np.square(np.average(z)))* sum(np.square(z16)-np.square(np.average(z16)))))
print r16
r17=(sum((z-np.average(z))*( z17-np.average(z17))))/(np.sqrt(sum(z*z-np.square(np.average(z)))* sum(np.square(z17)-np.square(np.average(z17)))))
print r17
r18=(sum((z-np.average(z))*( z18-np.average(z18))))/(np.sqrt(sum(z*z-np.square(np.average(z)))* sum(np.square(z18)-np.square(np.average(z18)))))
print r18
r19=(sum((z-np.average(z))*( z19-np.average(z19))))/(np.sqrt(sum(z*z-np.square(np.average(z)))* sum(np.square(z19)-np.square(np.average(z19)))))
print r19
r20=(sum((z-np.average(z))*( z20-np.average(z20))))/(np.sqrt(sum(z*z-np.square(np.average(z)))* sum(np.square(z20)-np.square(np.average(z20)))))
print r20
r21=(sum((z-np.average(z))*( z21-np.average(z21))))/(np.sqrt(sum(z*z-np.square(np.average(z)))* sum(np.square(z21)-np.square(np.average(z21)))))
print r21
r22=(sum((z-np.average(z))*( z22-np.average(z22))))/(np.sqrt(sum(z*z-np.square(np.average(z)))* sum(np.square(z22)-np.square(np.average(z22)))))
print r22
r23=(sum((z-np.average(z))*( z23-np.average(z23))))/(np.sqrt(sum(z*z-np.square(np.average(z)))* sum(np.square(z23)-np.square(np.average(z23)))))
print r23
r24=(sum((z-np.average(z))*( z24-np.average(z24))))/(np.sqrt(sum(z*z-np.square(np.average(z)))* sum(np.square(z24)-np.square(np.average(z24)))))
print r24
r25=(sum((z-np.average(z))*( z25-np.average(z25))))/(np.sqrt(sum(z*z-np.square(np.average(z)))* sum(np.square(z25)-np.square(np.average(z25)))))
print r25
r26=(sum((z-np.average(z))*( z26-np.average(z26))))/(np.sqrt(sum(z*z-np.square(np.average(z)))* sum(np.square(z26)-np.square(np.average(z26)))))
print r26
r27=(sum((z-np.average(z))*( z27-np.average(z27))))/(np.sqrt(sum(z*z-np.square(np.average(z)))* sum(np.square(z27)-np.square(np.average(z27)))))
print r27
r28=(sum((z-np.average(z))*( z28-np.average(z28))))/(np.sqrt(sum(z*z-np.square(np.average(z)))* sum(np.square(z28)-np.square(np.average(z28)))))
print r28
r29=(sum((z-np.average(z))*( z29-np.average(z29))))/(np.sqrt(sum(z*z-np.square(np.average(z)))* sum(np.square(z29)-np.square(np.average(z29)))))
print r29
r30=(sum((z-np.average(z))*( z30-np.average(z30))))/(np.sqrt(sum(z*z-np.square(np.average(z)))* sum(np.square(z30)-np.square(np.average(z30)))))
print r30
r31=(sum((z-np.average(z))*( z31-np.average(z31))))/(np.sqrt(sum(z*z-np.square(np.average(z)))* sum(np.square(z31)-np.square(np.average(z31)))))
print r31

lift1 = z1+z2+z3+z4+z5+z6+z7+z8+z9+z10+z11+z12+z13+z14+z15+z16+z17+z18+z19+z20+z21+z22+z23+z24+z25+z26+z27+z28+z29+z30+z31
lift = rms(lift1)
print lift

plt.figure(figsize=(12,3))

ax0 = plt.subplot(1,1,1)

ax0.set_xlim(0,100)
plt.plot(time,z1,color="dodgerblue")
plt.plot(time,z31,color="dodgerblue")
plt.plot(time,lift1,color="dodgerblue")

