
#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 16:56:00 2019

@author: femap
"""

import numpy as np
import matplotlib.pyplot as plt
import math


# import data
data= np.loadtxt("out_data.dat", dtype=str, delimiter=",", unpack=False)

x1= data [1:32]


#time0, z0 = np.loadtxt("hull_motion.txt", skiprows=1, delimiter="\t", usecols=(0,3), unpack=True)
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

plt.figure(figsize=(12,9))

print x1

ax0 = plt.subplot(3,1,1)

ax0.set_xlim(54,60)
plt.plot(time,z,color="dodgerblue")


ax1 = plt.subplot(3,1,2)

ax1.set_xlim(1,81)

plt.plot(time,z1,color="dodgerblue")
plt.plot(time, z31, color="blue")

h = x1

ax2 = plt.subplot(3,1,3)

cm1 = plt.cm.get_cmap('jet')

#mrelab-k100-u15
#cs10 = plt.contourf(h,12,cmap=cm1)

cs10 = plt.contourf(h,levels=[-1.1,-0.6,-0.4,-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.4,0.6,1.1],
                 cmap=cm1)
#cs10 = plt.contourf(h,level=12, cmap=cm1)
plt.colorbar(cs10, extend='neither' )
plt.clim(-0.3,0.3)
major_ticks = np.arange(0, 100, 5)                                              
minor_ticks = np.arange(0, 101, 1)                                               

#ax2.set_xticks(major_ticks)                                                       
#ax2.set_xticks(minor_ticks, minor=True)                                           
#ax2.set_yticks(major_ticks)                                                       
ax2.set_xticks(minor_ticks, minor=True) 

ax2.set_xlim(1,101)
ax2.set_ylim(0,30)

plt.grid(c='k', which='minor', axis='x', ls='-', alpha=0.2)
plt.xticks([])
plt.yticks([])
#plt.rcParams['xtick.direction']='in'


ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['bottom'].set_visible(False)

            
plt.show

plt.savefig("u0.25.png", dpi=300)

