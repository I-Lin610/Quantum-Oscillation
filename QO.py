# This code is used to plot raw data of quantum oscillation, oscillation pattern : backgraound subtraction and crosssection area of Fermi surface from FFT. We could also fit the Dingle temp and fitting the effective mass of each pocket

# the packages installed
import numpy as np
import pandas as pd
import scipy 
from pylab import genfromtxt
from scipy.optimize import curve_fit
#from  matplotlib import pylab
import matplotlib.pyplot as plt
from matplotlib import gridspec
import math
from math import exp
from scipy.fftpack import fft, fftfreq
#import scipy.special as sp

# write the file for output of the plots/data

#Fmass = open ('0p0001.dat','ab')
#fig = plt.figure(figsize= (25,15))
plt.rc('legend',**{'fontsize':16})
plt.rc('xtick',**{'labelsize':16})
plt.rc('ytick',**{'labelsize':16})
plt.rc('axes',**{'labelsize':20})


# Effective mass plot& fitting

dataM =np.genfromtxt ('0p0001.dat', delimiter =',', skip_header =1)
Bf   = 8.4 # (6-18 T Dynacool)1atm
AmTB = 14.694/Bf


def funcMeff (x, a,m):
    return a*AmTB*m*x/(np.sinh(AmTB*m*x))

XNew = np.linspace(0.1, 8, 1000)
Xrr = (dataM[0:,-1])
Yr  = (dataM[0:, 1])
pr,pcr =curve_fit(funcMeff, Xrr,Yr)
Ya  = (dataM[0:, 3])
pa,pca =curve_fit(funcMeff, Xrr,Ya)
Yb  = (dataM[0:, 5])
pb,pcb =curve_fit(funcMeff, Xrr,Yb)

#Effective mess fitting
plt.subplot(324)
plt.xlabel(r'T (K)',fontsize=20)
plt.ylabel(r'FFT Amplitude (a.u.)',fontsize=20)
plt.plot (Xrr,Yr, 'g^', label =r'$m_{\gamma}$ = 0.309')#,fontsize =20)
plt.plot (Xrr,Ya, 'rs', label =r'$m_{\alpha}$ = 0.591')#,fontsize =20)
plt.plot (Xrr,Yb, 'bo', label =r'$m_{\beta}$ = 0.530')#,fontsize =20)
plt.plot ( XNew, funcMeff(XNew,*pr), 'C2')
plt.plot ( XNew, funcMeff(XNew,*pa), 'C1')
plt.plot ( XNew, funcMeff(XNew,*pb), 'C0')
lgd = plt.legend(bbox_to_anchor= (1,1),loc='upper right', fontsize =28)
plt.legend()
plt.legend(bbox_to_anchor= (1,1),loc='upper right')

#Fitting of berry phase & Dingle temperature

# preset of the initial fitting parameters and file at based temp
[File,me1,me2,me3,me4,T]=['R18.dat',0.30877376,0.52992297,0.5912156,0.0,1.8]
data1 = np.genfromtxt (File, delimiter =',', skip_header=31) 
Am  = 14.694
AmT = 14.694*T
[a1,Td1,Gamma1,B1] = [3.4165,19.7957,0.93539,42.5]
[a2,Td2,Gamma2,B2] = [0.3678,4.9233,0.87681,240.82]
[a3,Td3,Gamma3,B3] = [2.0263,6.2344,0.88241,257.7]
Y11  = data1 [:,6]*1000000
X11  = data1 [:,4]/10000

#background fitting
def func1 (x,a,b,c):
    return a*x*x+b*x+c
p1,pc1 = curve_fit(func1, X11,Y11)

#subtracte the background from the raw data and inverse the B-> 1/B
Y1 = Y11-func1(X11,*p1)
X1 = 1/X11
print (p1)

XD = np.array(X1[0:])
YD = np.array(Y1[0:])

#To getdingle temp and Berry phase, we fit the multiple band of LK formular with the suvtsritution of effective mass and Dingle temp.
def funcband (x, a2,Td2,a3,Td3,a1,Td1,Gamma2,Gamma3,Gamma1):
    return (a1*(np.exp(-Am*me1*Td1*x))*(AmT*me1*x)*np.cos(2.0*np.pi*(B1*x+Gamma1))/(np.sinh(AmT*me1*x))+a2*(np.exp(-Am*me2*Td2*x))*(AmT*me2*x)*np.cos(2.0*np.pi*(B2*x+Gamma2))/(np.sinh(AmT*me2*x))+a3*(np.exp(-Am*me3*Td3*x))*(AmT*me3*x)*np.cos(2.0*np.pi*(B3*x+Gamma3))/(np.sinh(AmT*me3*x))*np.sqrt(x))
pband, pcband = curve_fit(funcband, XD,YD)
Xnew =1/np.linspace(5.0, 14.01,5000)
print (pband)

#plot Fitting berry phase 
plt.subplot(326) #plot MR-H
plt.xlabel(r'$\frac{1}{B}$ ($\frac{1}{T}$)',fontsize=20)
plt.ylabel('$\Delta$ MR (m$\Omega$)',fontsize=20)
#plt.text(80, 38, r'(c)',fontsize = 20)
#plt.text(80, 23.8, r'(d)',fontsize = 20)
#plt.text(80, 8.8, r'(e)',fontsize = 20)
plt.plot (X1,Y1, 'b', label ='data')#,fontsize =20)
plt.plot ( Xnew, funcband(Xnew,*pband), 'r', label= 'Fitting')
lgd = plt.legend(bbox_to_anchor= (1,1),loc='upper right', fontsize =28)
plt.legend()
plt.legend(bbox_to_anchor= (1,1),loc='upper right')

# input of temp dependence of raw data file
for [AA, Lname]  in [['R18.dat','1.8 K'],['R25.dat','2.5 K'],['R3.dat','3.0 K'],['R35.dat','3.5 K'],['R4.dat','4.0 K'],['R45.dat','4.5 K'],['R5.dat','5.0 K'],['R55.dat','5.5 K']]:#, '1800MoTe21.dat', '1800MoTe24.dat']:
    data = np.genfromtxt (AA, delimiter =',', skip_header=31) 

#the unite of field is T, and microOhm of resistance
    X  = data[0:,4]/10000
    Y  = data[0:,6]*1000000

#fitting background  
    def func(x,a,b,c,d):
        return d*x**3+a*x*x+b*x+c
    popt, pcov =curve_fit(func,X,Y)

##    print (X[7:8])#test H for effective mass

#background subtraction
    Xh = 1/X
    Yh = Y-func(X,*popt)

#the delta x for FFT
    dt = Xh[10]-Xh[11]
    N =len(X[:])

    Yf = fft(Yh[0:])
    Xf = fftfreq(N, dt)

# To determine the effective mass : write the data of the intensity of the FFT frequency

#    print (Xf[3:4],100.0/N*np.abs(Yf[22:23] ))
#    TEST = np.array([-Xf[22:23], 1000.0/N*np.abs(Yf[22:23]),-Xf[25:26], 1000.0/N*np.abs(Yf[25:26])  ])
# write I(peak)
##    TEST = np.array([-Xf[3:4], 1000.0/N*np.abs(Yf[3:4]), -Xf[22:23], 1000.0/N*np.abs(Yf[22:23]),-Xf[25:26], 1000.0/N*np.abs(Yf[25:26])  ])
##    TEST = TEST.T
##    np.savetxt(Fmass, TEST, newline = '\n', delimiter = ',')
#    plt.plot(Xf, 1000.0/N*np.abs(Yf))#, xlabel='frequency (T)', ylabel ='abs')

#plot MR-H( raw data)
#    fig = plt.figure()
    plt.subplot(221) #plot MR-H
    plt.xlabel ('B (T)', fontsize = 20)
#    plt.ylim (0.02,0.045)
    plt.xlim (6,14)
    plt.ylabel('MR ( m $\Omega$ )', fontsize = 20)
    plt.plot (X, Y/2.987313,label=Lname)#, fontsize = 20)
    lgd = plt.legend(bbox_to_anchor= (1,0.55),loc='upper right', fontsize =20)
    plt.legend()
    plt.legend(bbox_to_anchor= (1,0.55),loc='upper right')
#    plt.subplot (321) #plot MR-H
#    plt.plot (X, Y,label=Lname)


#plot MR-1/H ( quantum oscillation of 1/B- quantum oscillation signal)
    plt.subplot (223) #plot MR-1/H
    plt.xlabel(r'$\frac{1}{B}$ ($\frac{1}{T}$)',fontsize=20)
    plt.ylabel ('$\Delta$ MR ( m $\Omega$ )', fontsize = 20)
#    plt.text (50,10,'(b)')
    plt.xlim (0.07,0.16)
#    plt.text(80, 8.8, r'(a)',fontsize = 20)
#    plt.text(80, 38, r'(b)',fontsize = 20)
    plt.plot (Xh[:],Yh[:]) 
#    lgd = plt.legend(bbox_to_anchor= (1,1),loc='upper right', fontsize =20)
    plt.legend()
#    plt.legend(bbox_to_anchor= (1,1),loc='upper right')
    plt.draw()


#crosssection area of Fermi surface from FFT
    plt.subplot (322) 
    plt.xlim (0,1200)
    plt.xlabel ('frequency (T)',fontsize = 20)
    plt.ylabel ('FFT amplitude (a.u.)', fontsize = 20)
    plt.text(-1350, 100, r'(a)',fontsize = 20)
    plt.text(-1350, -130, r'(b)',fontsize = 20)
    plt.text(80, 100, r'(c)',fontsize = 20)
    plt.text(60, -54.,r'(d)', fontsize = 20)
    plt.text(80, -195, r'(e)', fontsize = 20)    
    plt.plot(Xf, 1000.0/N*np.abs(Yf),label=Lname)#, xlabel='frequency (T)', ylabel ='abs')
    lgd = plt.legend(bbox_to_anchor= (1,1),loc='upper right', fontsize =20)
    plt.legend()
    plt.legend(bbox_to_anchor= (1,1),loc='upper right')
#fig.savefig('p1t.pdf')
plt.draw()
plt.show()

#fig.savefig('p1t.pdf')
#fig=plt.gcf()
#Fmass.close()
