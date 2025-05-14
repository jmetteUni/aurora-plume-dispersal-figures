#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 17:03:22 2024

@author: jmette@uni.bremen.de
"""

import os
working_dir = '/home/jonathan/Dokumente/Masterarbeit/python/thesis/'
os.chdir(working_dir)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.signal as sg  #Package for signal analysis
import scipy.ndimage as si #Another package for signal analysis
from scipy import stats    #Used for 2D binned statistics
import xarray as xr
import analytic_wavelet
from scipy import signal, ndimage
from datetime import timedelta
import spectrum

rcm_path = '/home/jonathan/Dokumente/Masterarbeit/MooringData/RCM/'
fig_path = '/home/jonathan/Dokumente/Masterarbeit/Thesis_Latex_Project/figures/4-results/4-2-time/'


#%%#################### Print Figures Flag #####################################
print_fig = True
#%% import data
rcm474_path = rcm_path+"rcm474"
rcm506_path = rcm_path+"rcm506"
rcm512_path = rcm_path+"rcm512"
rcm569_path = rcm_path+"rcm569"

rcm474 = pd.read_csv(rcm474_path,names=['datetime','u','v'])
rcm506 = pd.read_csv(rcm506_path,names=['datetime','u','v'])
rcm512 = pd.read_csv(rcm512_path,names=['datetime','u','v'])
rcm569 = pd.read_csv(rcm569_path,names=['datetime','u','v'])

time_vec = pd.date_range(pd.to_datetime(rcm569.datetime.iloc[0]),pd.to_datetime(rcm569.datetime.iloc[-1]),freq='h')
rcm569.datetime = time_vec

for data in rcm474,rcm506,rcm512,rcm569:
    data['datetime'] = pd.to_datetime(data['datetime'],format='mixed')
    data['cv'] = data['u'] + 1j*data['v']
#rcm474_joel.drop(rcm474_joel.tail(80).index,inplace=True) #remove deployment and recovery
#rcm474_joel.drop(rcm474_joel.head(80).index,inplace=True)
#%% statistical analyis
plt.figure()
plt.title('474')
rcm474['u'].plot.hist()

plt.figure()
plt.title('506')
rcm506['u'].plot.hist()

plt.figure()
plt.title('512')
rcm512['u'].plot.hist()

plt.figure()
plt.title('569')
rcm569['u'].plot.hist()

data = rcm474
data.loc[data['u'] > 0,'u'].count()/data['u'].count()

#%% timeseries

def plot_complex_velocity(data,filt=False,rot=False):
    """
    Line plot of the u and v parts of a complex velocity.

    Args:
        date: Date as a datetime object
        cv: Complex velocity u+iv in cm/s, with time in *rows*

    Returns:
        Nothing
    """
    cv = data.cv
    comp1_label = 'u'
    comp2_label = 'v'
    if rot == True:
        cv = data.cvr
        comp1_label = 'Downstream'
        comp2_label = 'Crosstream'
    if filt == True:
        window = sg.windows.hann(24)      #24-point hanning window
        window = window / np.sum(window)  #normalized to sum to unity
        cv =  si.convolve(cv, window, mode="mirror")  #filtered version of velocity at deepest depth

    date = data.datetime
    cv = np.array(cv)
    plt.figure(figsize=(10,2.5))
    ax=plt.gca()
    ax.plot(date,cv.real, linewidth=1,label=comp1_label)
    ax.plot(date,cv.imag, linewidth=1,label=comp2_label)
    ax.autoscale(enable=True, tight=True)
    plt.ylabel('Velocity in cm s$^{-1}$')
    ax.axhline(0, linestyle=":", color="grey", linewidth=0.5)
    ax.legend()
    plt.tight_layout()

plot_complex_velocity(rcm474)
if print_fig == True:
    plt.savefig(fig_path+'timeseries_rcm474.pdf')
plot_complex_velocity(rcm506)
if print_fig == True:
    plt.savefig(fig_path+'timeseries_rcm506.pdf')
plot_complex_velocity(rcm512)
if print_fig == True:
    plt.savefig(fig_path+'timeseries_rcm512.pdf')
plot_complex_velocity(rcm569)
if print_fig == True:
    plt.savefig(fig_path+'timeseries_rcm569.pdf')

#%% filtered timeseries

fig, axs = plt.subplots(4)
plot_complex_velocity(rcm474,filt=True)
if print_fig == True:
    plt.savefig(fig_path+'timeseries_filt_rcm474.pdf')
plot_complex_velocity(rcm506,filt=True)
if print_fig == True:
    plt.savefig(fig_path+'timeseries_filt_rcm506.pdf')
plot_complex_velocity(rcm512,filt=True)
if print_fig == True:
    plt.savefig(fig_path+'timeseries_filt_rcm512.pdf')
plot_complex_velocity(rcm569,filt=True)
if print_fig == True:
    plt.savefig(fig_path+'timeseries_filt_rcm569.pdf')

#%% progressive vector diagram

import matplotlib.ticker as mticker
def provec(dt,rcm,var,device,color):
    """
    Compute and plot the progressive vector diagram.

    Args:
        dt: Sampling interval in seconds
        cv: Complex velocity u+iv in cm/s, with time in *rows*
        ticks: Tickmark locations in km (optional); defaults to every 1000 km

    Returns:
        Nothing
    """
    cv=rcm[var]
    t = rcm.datetime
    t_hours = np.arange(0,len(t))
    cx=np.array(np.cumsum(cv, axis=0)*dt/100/1000)  #complex displacement x+iy in kilometers

    ax=plt.gca()
    #displ = ax.scatter(np.array(cx).real,np.array(cx).imag,c=np.array(t_hours),s=0.5,cmap='hsv_r') #use for seasons as colorcoding
    displ = ax.plot(np.array(cx).real,np.array(cx).imag,color=color) #use to plot without seasons

    ax.set_aspect('equal')  #this sets the data aspect ratio to 1:1   <--------------
    ax.text(cx.real[-1]-19,cx.imag[-1]+5,device)

    plt.xlabel('Displacement East - West in km')
    plt.ylabel('Displacement North - South in km')

    #plt.xticks(ticks), plt.yticks(ticks)    #set x-tick and y-tick locations
    #ax.autoscale(enable=True, tight=True)   # tighten axes limits around the data

    return(displ)

#generates monthly tick labels

#dt=(rcm_datetime[1]-rcm_datetime[0])
dt=60*60
ticks_months = np.arange(155,8171,750)
ticks_labels = ['Aug','Sep','Oct','Nov','Dec','Jan','Feb','Mar','Apr','Jun','Jul']
labels= ['474: 3572m','506: 3366m','512: 3215m']

fig, ax = plt.subplots(1, 1,figsize=(6,4))
provec(dt,rcm512,'cv','512','blue')
provec(dt,rcm506,'cv','506','orange')
provec(dt,rcm569,'cv','569','olive')
displ = provec(dt,rcm474,'cv','474','red')
#plt.colorbar(displ,fraction=0.03, pad=0.04,ticks=ticks_months,format=mticker.FixedFormatter(ticks_labels),extend='both') #month coloring

plt.scatter(0,0,marker='+',color='black',s=50)
plt.legend(['512 at 3064 m','506 at 3215 m','569 at 3366 m','474 at 3572 m'])
plt.tight_layout()
plt.show()
if print_fig == True:
    plt.savefig(fig_path+'prog_vector.png',dpi=300)


#%% scatter with mean +auqadopp
aqua_u_mean = -0.06024621842173035
aqua_v_mean = 0.06431983493416378

plt.figure(figsize=(8,8))

#plt.scatter(rcm512['u'].mean(),rcm512['v'].mean(),marker='x')
plt.arrow(0,0,rcm512['u'].mean(),rcm512['v'].mean(),edgecolor='None',facecolor='blue',width=0.05)
print('u:'+str(rcm512['u'].mean())+' v:'+str(rcm512['v'].mean()))

#plt.scatter(rcm506['u'].mean(),rcm506['v'].mean(),marker='x')
plt.arrow(0,0,rcm506['u'].mean(),rcm506['v'].mean(),edgecolor='None',facecolor='orange',width=0.05)
print('u:'+str(rcm506['u'].mean())+' v:'+str(rcm506['v'].mean()))

#plt.scatter(-0.045172413793103446*100,0.025551724137931043*100,color='magenta',marker='x')
plt.arrow(0,0,aqua_u_mean*100,aqua_v_mean*100,edgecolor='None',facecolor='magenta',width=0.05)
print('u:'+str(aqua_u_mean*100)+' v:'+str(aqua_v_mean*100))


#plt.scatter(rcm569['u'].mean(),rcm569['v'].mean(),marker='x')
plt.arrow(0,0,rcm569['u'].mean(),rcm569['v'].mean(),edgecolor='None',facecolor='olive',width=0.05)
print('u:'+str(rcm569['u'].mean())+' v:'+str(rcm569['v'].mean()))

#plt.scatter(rcm474['u'].mean(),rcm474['v'].mean(),marker='x')
plt.arrow(0,0,rcm474['u'].mean(),rcm474['v'].mean(),edgecolor='None',facecolor='red',width=0.05)
print('u:'+str(rcm474['u'].mean())+' v:'+str(rcm474['v'].mean()))

#plot origin for orientation and mean of aquadopp level
#plt.scatter(0,0,marker='x',color='black',label='Origin')

plt.xlabel('Mean $u$ in cm/s')
plt.ylabel('Mean $v$ in cm/s')
plt.gca().set_aspect('equal')
plt.grid()
plt.legend(['512 at 3064m','506 at 3215 m','Aquadopp at 3220m','569 at 3366 m','474 at 3572 m','0,0'])
plt.tight_layout()
plt.show()
if print_fig == True:
    plt.savefig(fig_path+'aqua_moor_means.png',dpi=300)

#%% rotated currents

#rotation angle
print(np.mean(rcm569.cv))

phi512 = np.angle(np.mean(rcm512.cv))
phi=np.angle(np.mean(rcm512.cv));
print(np.mean(rcm512.cv)*np.exp(1j*-phi)) #mean cv after rotation

for i in rcm474,rcm506,rcm512,rcm569: #rotate all cv's'
    i['cvr'] = i.cv*np.exp(1j*phi)

#%% rotated progressive vector diagram
fig, ax = plt.subplots(1, 1,figsize=(8,8))
provec(dt,rcm474,'cvr','474')
provec(dt,rcm512,'cvr','569')
provec(dt,rcm506,'cvr','506')
displ = provec(dt,rcm569,'cvr','512')
plt.scatter(0,0,marker='x',color='black',s=50)
plt.legend(['474 at 3572 m','569 at 3366 m','506 at 3215 m', '512 at 3064m'])
plt.show()
if print_fig == True:
    plt.savefig(fig_path+'prog_vector_rot.pdf')

#%% filtered rotated timeseries

plot_complex_velocity(rcm474,filt=True,rot=True)
if print_fig == True:
    plt.savefig(fig_path+'timeseries_filtrot_rcm474.pdf')
plot_complex_velocity(rcm506,filt=True,rot=True)
if print_fig == True:
    plt.savefig(fig_path+'timeseries_filtrot_rcm506.pdf')
plot_complex_velocity(rcm512,filt=True,rot=True)
if print_fig == True:
    plt.savefig(fig_path+'timeseries_filtrot_rcm512.pdf')
plot_complex_velocity(rcm569,filt=True,rot=True)
if print_fig == True:
    plt.savefig(fig_path+'timeseries_filtrot_rcm569.pdf')

#%% var ellipse
def plot_varellipse(data,label):
    def plot_twodstat(xbins,ybins,x,y,z=False,statistic="count",tickstep=False,axlines=(0,0),cmap = 'nipy_spectral',colorbar=True,meandot=True,meanline=False,axisequal=False):
        """
        Compute and plot two a dimensional statistic.

        Args:
            xbins: Array of bin edges for x-bins
            ybins: Array of bin edges for y-bins
            x: Array of x-values to be binned
            y: Array of y-values to be binned; same size as x

        Optional Args:
            z: Array of z-values for which statistic is be formed; same size as x
            statistic: "count", "log10count", "mean", "median", or "std";
                defaults to "count", in which case the z argument is not needed
            tickstep: X- and y-axis tick step, a length 2 tuple; defaults to auto
            axlines: Axis origin locations for horizontal and vertical lines,
                a length 2 tuple, defaults to (0,0), lines omitted if False
            cmap: Colormap, defaults to Spectral_r
            colorbar: Plots a colorbar, defaults to True
            meandot: Plots a dot at the mean value, defaults to true
            meanline: Plots a line from the origin to the mean value, defaults to false
            axisequal: Sets plot aspect ratio to equal, defaults to false

        Returns:
            im: Image handle
            tuple of mean x,y coordinates


        The computation of the statistic is handled by stats.binned_statistic_2d.

        Note also that z may be complex valued, in which case we define the standard
        deviation the square root of <(z - <z>)(z - <z>)^*> = <|z|^2> - |<z>|^2,
        which will be real-valued and non-negative.
        """

        #plot just one twodhist
        if statistic=="count":
            q = stats.binned_statistic_2d(x, y, None, bins=[xbins, ybins], statistic="count").statistic
            q[q==0]=np.nan  #swap zero values for NaNs, so they don't appear with a color
            clabel='Histogram'
        elif statistic=="log10count":
            q = stats.binned_statistic_2d(x, y, None, bins=[xbins, ybins], statistic="count").statistic
            q[q==0]=np.nan  #swap zero values for NaNs, so they don't appear with a color
            q=np.log10(q)
            clabel='Log10 Histogram'
        elif statistic=="mean":
            q = stats.binned_statistic_2d(x, y, z, bins=[xbins, ybins], statistic="mean").statistic
            clabel='Mean'
        elif statistic=="median":
            q = stats.binned_statistic_2d(x, y, z, bins=[xbins, ybins], statistic="median").statistic
            clabel='Median'
        elif statistic=="std":
            if np.all(np.isreal(z)): #real-valued case
                 q  = stats.binned_statistic_2d(x, y, z, bins=[xbins, ybins], statistic="std").statistic
            else:  #complex-valued case
                qr  = stats.binned_statistic_2d(x, y, z.real, bins=[xbins, ybins], statistic="std").statistic
                qi  = stats.binned_statistic_2d(x, y, z.imag, bins=[xbins, ybins], statistic="std").statistic
                q = np.sqrt(qr**2+qi**2)
            clabel='Standard Deviation'

        ax=plt.gca()

        im=ax.pcolormesh(xbins, ybins, np.transpose(q), cmap=cmap, shading="flat",vmin=0,vmax=2.4) #add for fixed colorbar
        if colorbar:
            cb=fig.colorbar(im, ax=ax)
            cb.set_label(clabel)

        if axisequal:
            ax.set_aspect("equal")

        if not(not(axlines)):
            ax.axhline(axlines[0], linestyle=":", color="k")
            ax.axvline(axlines[1], linestyle=":", color="k")

        if meanline:
            #plt.arrow(0,0,np.mean(x),np.mean(y),width=0.8,length_includes_head=False,facecolor="k",edgecolor="w")
            ax.plot([0,np.mean(x)],[0,np.mean(y)],color="w",linewidth=4.5)
            ax.plot([0,np.mean(x)],[0,np.mean(y)],color="k",linewidth=3)

        if meandot:
            ax.plot(np.mean(x),np.mean(y), "wo", markerfacecolor="k", markersize=8)

        plt.xlim([min(xbins), max(xbins)]),plt.ylim([min(ybins), max(ybins)])

        if not(tickstep==False):
            plt.xticks(np.arange(min(xbins), max(xbins), tickstep[0]))  # set x-label locations
            plt.yticks(np.arange(min(ybins), max(ybins), tickstep[1]))  # set x-label locations

        return im, (np.mean(x),np.mean(y))

    def variance_ellipse(u,v):
        """
        Compute parameters of the variance ellipse.

        Args:
            u: 1-D array of eastward velocities (or real part of complex variable)
            v: 1-D array of northward velocities (or imaginary part of complex variable)

        Returns:
            a: semi-major axis
            b: semi-minor axis
            theta: orientation angle counterclockwise from x axis, in radians
        """

        #compute terms in the covariance matrix
        cuu=np.mean(np.multiply(u-np.mean(u),u-np.mean(u)))
        cvv=np.mean(np.multiply(v-np.mean(v),v-np.mean(v)))
        cuv=np.mean(np.multiply(u-np.mean(u),v-np.mean(v)))

        detc=np.real(cuu*cvv-cuv**2) #determinant of covariance matrix
        trc=cuu+cvv #trace of covariance matrix

        a=np.sqrt(trc/2+np.sqrt(trc**2-4*detc)/2)#semi-major axis
        b=np.sqrt(trc/2-np.sqrt(trc**2-4*detc)/2)#semi-minor axis
        theta=np.arctan2(2*cuv,cuu-cvv)/2#orientation angle

        return a,b,theta

    def plot_ellipse(a,b,theta,center=(0,0),color=(0.5,0.5,0.5),linewidth=3,outline="None",aspect=1,transform=False):
        """
        Plot an ellipse.

        Args:
            a: Ellipse semi-major axis
            b: Ellipse semi-minor axis
            theta: Ellipse orientation angle in radians

        Optional Args:
            center: The ellipse center, a complex number or 2-tuple; defaults to (0,0)
            color: Ellipse line color, a 3-tuple; defaults to (0.5,0.5,0.5)
            linewidth: Ellipse line width; defaults to 3
            outline: Color for an outline around the ellipse; defaults to "w"
            aspect: Aspect ratio by which to adjust the ellipse; defaults to one
            transform: Coordinate transform to apply for use with Cartopy; defaults to none

        Returns:
            h: Handle to ellipse object
            z: Array of complex numbers describing the ellipse periphery
        """
        phi=np.arange(0,2*np.pi+.1,.1)

        if isinstance(center,complex):
            x=center.real
            y=center.imag
        else:
            x=center[0];
            y=center[1];

        z=np.exp(1j*theta)*(a*np.cos(phi)-1j*b*np.sin(phi))
        ax = plt.gca()

        if not(transform):
            ax.plot(np.real(z)+x,(1/aspect)*np.imag(z)+y,linewidth=linewidth*1.5,color=outline)
            h=ax.plot(np.real(z)+x,(1/aspect)*np.imag(z)+y,linewidth=linewidth,color=color)
        else:
            ax.plot(np.real(z)+x,(1/aspect)*np.imag(z)+y,linewidth=linewidth*1.5,color=outline,transform=transform)
            h=ax.plot(np.real(z)+x,(1/aspect)*np.imag(z)+y,linewidth=linewidth,color=color,transform=transform)

        return h,z

    rcm = data
    rcm_name = label
    #same plot as above using 1x1 cm/s bins
    fig,ax = plt.subplots(1,1)
    bins = np.arange(-10, 10, 0.5)
    im, mean = plot_twodstat(bins,bins,np.array(rcm.cv).real,np.array(rcm.cv).imag, statistic="log10count",meanline=False,axisequal=True)
    plt.xlabel('u in cm/s'),plt.ylabel('v in cm/s')
    #plt.title('Two-Dimensional Histogram of Velocity, '+rcm_name);
    #form ellipse parameters for rotated velocity at deepest depth
    a,b,theta = variance_ellipse(np.array(rcm.cv).real,np.array(rcm.cv).imag)
    plot_ellipse(a,b,theta,center=np.mean(rcm.cv));

    print('Mean u-velocity: '+str(mean[0]))
    print('Mean v-velocity: '+str(mean[1]))
    print('Var ellipse a: '+str(a))
    print('Var ellipse b: '+str(b))
    print('Var ellipse Theta: '+str(theta))

plot_varellipse(rcm474, '474')
if print_fig == True:
    plt.savefig(fig_path+'varell_rcm474.pdf')
plot_varellipse(rcm569, '569')
if print_fig == True:
    plt.savefig(fig_path+'varell_rcm569.pdf')
plot_varellipse(rcm506, '506')
if print_fig == True:
    plt.savefig(fig_path+'varell_rcm506.pdf')
plot_varellipse(rcm512, '512')
if print_fig == True:
    plt.savefig(fig_path+'varell_rcm512.pdf')

#%% Wavelet transform

#define functions to return the Coriolis and tidal frequencies
def corfreq(lat):
    """
    The Coriolis frequency in rad / day at a given latitude.

    Args:
        lat: Latitude in degree

    Returns:
        The Coriolis frequecy at latitude lat
    """
    omega=7.2921159e-5;
    return 2*np.sin(lat*2*np.pi/360)*omega*(3600)*24;

def tidefreq():
    """
    Eight major tidal frequencies in rad / day.  See Gill (1982) page 335.

    Args:
        None

    Returns:
        An array of eight frequencies
    """
    return 24*2*np.pi/np.array([327.85,25.8194,24.0659,23.9344,12.6584,12.4206,12.0000,11.9673])

rcm = rcm474
dt = 1/24 #Sample rate in days
num = (rcm.datetime.iloc[-1] - rcm.datetime.iloc[0]).days
num = np.linspace(0,num,len(rcm.datetime))
cv=np.array(rcm.cv) #complex-valued velocity at 4th depth
phi = np.angle(np.mean(cv))      #Find angle of downstream direction
cv = cv*np.exp(-phi*1j)  #rotated velocity

gamma = 3
beta = 2

morse = analytic_wavelet.GeneralizedMorseWavelet(gamma, beta)  # create morse wavelet object (class instance)
fs = morse.log_spaced_frequencies(num_timepoints=len(cv))  # use object to construct frequency space
psi, psif = morse.make_wavelet(len(cv), fs)  # use object to create a morse wavelet in time and frequency space

wu = analytic_wavelet.analytic_wavelet_transform(cv.real, psif, False)
wv = analytic_wavelet.analytic_wavelet_transform(cv.imag, psif, False)

window = signal.windows.hann(24)  # construct Hanning window
cv_smoothed = signal.convolve(cv, window, mode="same") / np.sum(window)  # smooth by convolving speed with window

fig, ax = plt.subplots(3, 1, figsize=(13, 8.5), sharex=True)

analytic_wavelet.time_series_plot(ax[0], num, cv_smoothed.real)  # internal function for plotting time series
analytic_wavelet.time_series_plot(ax[0], num, cv_smoothed.imag)  # internal function for plotting time series

cmap = plt.colormaps["Spectral_r"]  # choose colormap
#cmap = matplotlib.colormaps.get_cmap("Spectral_r")  # choose colormap
# internal function for constructing contourf plot. Takes plt.contourf **kwargs
c = analytic_wavelet.wavelet_contourf(ax[1], num, dt/(fs/2/np.pi), wu,
                                      levels=50,cmap=cmap, vmin=0, vmax=15)
c = analytic_wavelet.wavelet_contourf(ax[2], num, dt/(fs/2/np.pi), wv,
                                      levels=50,cmap=cmap, vmin=0, vmax=15)

for n in (1,2):
    ax[n].set_ylabel("Period in days", fontsize=14)
    ax[n].invert_yaxis()
    ax[n].set_yscale("log")
    #plot lines at the semidiurnal and tidal frequencies
    ax[n].axhline(2*np.pi/tidefreq()[5], linestyle="solid", color="gray")  #tidefreq()[5] is M2 semidiurnal
    ax[n].axhline(2*np.pi/tidefreq()[3], linestyle="solid", color="red")  #tidefreq()[5] is M2 semidiurnal
    ax[n].hlines(2*np.pi/corfreq(82.897767),ax[n].get_xlim()[0],ax[n].get_xlim()[1],linestyle=":", color="black")

    #plot edge effect lines
    L=dt*2*np.sqrt(2)*np.sqrt(gamma*beta)/fs;
    #ax[n].plot(num[0]-numo+L/2,dt*2*np.pi/fs,color="white")
    #ax[n].plot(num[-1]-numo-L/2,dt*2*np.pi/fs,color="white")
    ax[n].plot(num[0]+L/2,dt*2*np.pi/fs,color="white")
    ax[n].plot(num[-1]-L/2,dt*2*np.pi/fs,color="white")

#fig.colorbar(c, ax=ax[1])
#ax[0].set_title("Alongstream and Cross-Stream Wavelet Transforms", fontsize=14)
ax[0].legend(['u Velocity','v Velocity'])
ax[1].text(0,90,'u Transform')
ax[2].text(0,140,'v Transform')
plt.xlabel('Days from '+str(rcm.datetime.iloc[0]))

fig.tight_layout()

#%% spectral analysis

