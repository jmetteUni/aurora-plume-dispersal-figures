#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 10:53:26 2024

@author: jmette@uni-bremen.de
"""
import os
working_dir = '/path/to/working/directory/'
os.chdir(working_dir)

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from hyvent.io import read_from_csv
from hyvent.misc import keys_to_data, calc_vel_from_posi

#%%
fig_path = '/path/to/output/directory/figures/4-results/4-2-time/'

dat_path = '/path/to/Aquadopp/028-0101.dat'
hdr_path = '/path/to/Aquadopp/028-0101.hdr'
cnv_path = '/path/to/CTD_data_processing/csv_files_noprocessing/'

aurora_stations = ['022_01', '026_01', '028_01', '033_01', '036_01', '041_01', '054_01', '055_01']

#%%#################### Print Figures Flag #####################################
print_fig = True

#%% read cnv

profile_data = read_from_csv(cnv_path, 'cnv')
profile_data = keys_to_data(profile_data, 'cnv')
profile_data = pd.concat(profile_data.values(),ignore_index=True)
profile_data = profile_data[profile_data['Station'].isin(aurora_stations)]

#%% depth control plot and select station 028 for stationary aquadopp measurement

control_plot = True
if control_plot == True:
    station_list = [d for _, d in profile_data.groupby(['Station'])]
    for station in station_list:
        plt.figure()
        plt.plot(station.index,station['DEPTH'])
        plt.xlabel('Index')
        plt.ylabel('Depth in m')
        plt.title(station['Station'].iloc[0])
        plt.show()

profile_aqua = profile_data[profile_data['Station']=='028_01']
# stationary from index 60300 -> 69448
profile_aqua_level = profile_aqua[(profile_aqua.index>60300) & (profile_aqua.index<69448)]

control_plot = True
if control_plot == True:
    plt.figure()
    plt.plot(profile_aqua.index,profile_aqua['DEPTH'])
    plt.xlabel('Index')
    plt.ylabel('Depth in m')
    plt.title(station['Station'].iloc[0])
    plt.show()

#%% read quadopp station 028

def read_auquadopp(dat_path, hdr_path):
    #read and format header file
    aqua_hdr = pd.read_csv(hdr_path,header=None,skiprows=118,nrows=27)[0]
    aqua_hdr = aqua_hdr.str.split()
    aqua_hdr = aqua_hdr.apply(lambda x: x[1:])
    aqua_hdr = aqua_hdr.apply(lambda x: ''.join(x))

    #read aquadopp data file
    aqua = pd.read_csv(dat_path, sep='\s+',header=None)
    aqua = aqua.rename(index=str, columns=aqua_hdr)
    aqua.index = aqua.index.astype(int)

    #make column with complex velocites
    aqua['cv']=np.array(aqua['Velocity(Beam1|X|East)(m/s)']+1j*aqua['Velocity(Beam2|Y|North)(m/s)'])
    aqua = aqua.rename(columns={'Velocity(Beam1|X|East)(m/s)': 'u', 'Velocity(Beam2|Y|North)(m/s)': 'v'})
    aqua['phi'] = np.angle(aqua['cv'],deg=True)
    aqua['phi'] = np.mod(aqua['phi'],360)
    aqua['r'] = np.abs(aqua['cv'])

    #format date and time columns to pandas datetime
    aqua_datetime = aqua[['Year','Month(1-12)','Day(1-31)','Hour(0-23)','Minute(0-59)','Second(0-59)']]
    aqua_datetime = aqua_datetime.rename(columns = {'Month(1-12)':'Month','Day(1-31)':'Day','Hour(0-23)':'Hour','Minute(0-59)':'Minute','Second(0-59)':'Second'})
    aqua['datetime'] = pd.to_datetime(aqua_datetime)
    aqua = aqua.drop(columns=['Year','Month(1-12)','Day(1-31)','Hour(0-23)','Minute(0-59)','Second(0-59)'])

    return aqua

aqua = read_auquadopp(dat_path, hdr_path)

posi = profile_aqua[['datetime','CTD_lon','CTD_lat']]
aqua = pd.merge(aqua, posi, how='left', on='datetime')
aqua = aqua.drop_duplicates()

#%% calc distance and velocity

aqua = aqua.dropna(subset=['CTD_lon','CTD_lat'])

aqua = calc_vel_from_posi(aqua)
aqua['cv_corr'] = aqua['cv'] - aqua['cv_mov']
aqua['u_corr'] = aqua['u'] - aqua['u_mov']
aqua['v_corr'] = aqua['v'] - aqua['v_mov']
aqua['phi_corr'] = np.angle(aqua['cv'],deg=True)
aqua['phi_corr'] = np.mod(aqua['phi'],360)
aqua['r_corr'] = np.abs(aqua['cv'])

print(aqua.u.mean())
print(aqua.u_mov.mean())
print(aqua.u_corr.mean())

print(aqua.v.mean())
print(aqua.v_mov.mean())
print(aqua.v_corr.mean())



#%%
plt.plot(aqua['CTD_lon'],aqua['CTD_lat'])

#select depth plateau
aqua_level = aqua[(aqua.index>49) & (aqua.index<79)]

print(aqua_level.u.mean())
print(aqua_level.u_mov.mean())
print(aqua_level.u_corr.mean())

print(aqua_level.v.mean())
print(aqua_level.v_mov.mean())
print(aqua_level.v_corr.mean())


plt.figure()
plt.plot(aqua_level.datetime,aqua_level['Pressure(m)'])
plt.xlabel('Datetime')
plt.ylabel('Depth in m')
plt.show()
#%%
#plt.quiver(aqua_vel[8],aqua_vel[9],angles='xy')

def plot_aquadopp(data):
    plt.figure()
    mean = np.sqrt(np.mean(data['u_corr'])**2+np.mean(data['v_corr']))**2
    plt.plot(data['u_corr'],data['v_corr'],'x',color='m')
    plt.plot(0,0,'x',color='black',label='Origin')
    plt.plot(np.mean(data['u_corr']),np.mean(data['v_corr']),'o',color='m',label='Mean: '+str(round(mean,2)))
    print('u:'+str(np.mean(data['u_corr']))+'v:'+str(np.mean(data['v_corr'])))
    plt.xlabel('u in m s$^{-1}$')
    plt.ylabel('v in m s$^{-1}$')
    # print(np.mean(aqua_vel[8]))
    # print(np.mean(aqua_vel[9]))
    # print(np.sqrt(np.mean(aqua_vel[8])**2+np.mean(aqua_vel[9]))**2)
    plt.legend()
    plt.show()

def plot_movement(data):
    plt.figure()
    plt.plot(data['Heading(degrees)'], label='Heading')
    plt.plot(data['Pitch(degrees)'], label='Pitch')
    plt.plot(data['Roll(degrees)'], label='Roll')
    plt.ylabel('Orientation in Degree')
    plt.legend()
    plt.show()

def plot_displ(x,y):
    df = pd.DataFrame()
    x = x.dropna()
    y = y.dropna()
    df['u_displ'] = (x.diff() * 111320 * np.cos(y * np.pi/180)).cumsum()
    df['v_displ'] = (y.diff() * 111320).cumsum()

    plt.figure()
    plt.plot(df['u_displ'],df['v_displ'])
    plt.xlabel('Displacement East-West in m')
    plt.ylabel('Displacement North - South in m')
    #plt.title()
    plt.show()

#plot horizontal velocity measurements
plot_aquadopp(aqua)
plot_aquadopp(aqua_level)

#plot Heading,Pitch Roll
plot_movement(aqua)
plot_movement(aqua_level)

#plot displacement
plot_displ(profile_aqua['CTD_lon'], profile_aqua['CTD_lat'])
plot_displ(profile_aqua_level['CTD_lon'], profile_aqua_level['CTD_lat'])

if print_fig == True:
    plt.savefig(fig_path+'aquadopp_scatter.pdf')

#%% time series

def plot_complex_velocity(uv_data,datetime,filt=False,rot=False):
    """
    Line plot of the real and imaginary parts of a complex velocity.

    Args:
        date: Date as a datetime object
        cv: Complex velocity u+iv in cm/s, with time in *rows*

    Returns:
        Nothing
    """
    cv = np.array(uv_data)
    comp1_label = 'u'
    comp2_label = 'v'
    # if rot == True:
    #     cv = data.cvr
    #     comp1_label = 'Downstream'
    #     comp2_label = 'Crosstream'
    # if filt == True:
    #     window = sg.windows.hann(24)      #24-point hanning window
    #     window = window / np.sum(window)  #normalized to sum to unity
    #     cv =  si.convolve(cv, window, mode="mirror")  #filtered version of velocity at deepest depth

    date = datetime
    cv = np.array(cv)
    plt.figure(figsize=(10,2.5))
    ax=plt.gca()
    ax.plot(date,cv.real*100, linewidth=1,linestyle='--',color='tab:blue')    #convert to cm/s
    ax.plot(date,cv.imag*100, linewidth=1,linestyle='--',color='tab:orange')
    ax.scatter(date,cv.real*100, label=comp1_label,color='tab:blue')    #convert to cm/s
    ax.scatter(date,cv.imag*100, label=comp2_label,color='tab:orange')

    import matplotlib.dates as mdates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

    ax.autoscale(enable=True)
    plt.ylabel('Velocity in cm s$^{-1}$')
    ax.axhline(0, linestyle=":", color="grey", linewidth=0.5)
    ax.legend()

#plot_complex_velocity(aqua['cv'],aqua['datetime'])
plot_complex_velocity(aqua_level['cv_corr'],aqua_level['datetime'])
plt.tight_layout()
if print_fig == True:
    plt.savefig(fig_path+'aqua_level_timeseries_corr.pdf')

#plt.plot(aqua['Pressure(m)'])
#plt.plot(aqua_level['datetime'],aqua_level['Pressure(m)'])

#%% lot aquadopp prog vector plots
def provec(dt,data):
    """
    Compute and plot the progressive vector diagram.

    Args:
        dt: Sampling interval in seconds
        cv: Complex velocity u+iv in cm/s, with time in *rows*
        ticks: Tickmark locations in km (optional); defaults to every 1000 km

    Returns:
        Nothing
    """
    cv=np.array(data['u']+1j*data['v'])
    t = data.datetime
    t_hours = np.arange(0,len(t))
    cx=np.array(np.cumsum(cv, axis=0)*dt/1000)  #complex displacement x+iy in kilometers

    ax=plt.gca()
    #displ = ax.scatter(np.array(cx).real,np.array(cx).imag,c=np.array(t_hours),s=0.5,cmap='hsv_r') #use for seasons as colorcoding
    displ = ax.plot(np.array(cx).real,np.array(cx).imag) #use to plot without seasons

    ax.set_aspect('equal')  #this sets the data aspect ratio to 1:1   <--------------
    #ax.text(cx.real[-1]-19,cx.imag[-1]+5,device)

    plt.xlabel('Displacement East - West in m')
    plt.ylabel('Displacement North - Sout in m')

    #plt.xticks(ticks), plt.yticks(ticks)    #set x-tick and y-tick locations
    #ax.autoscale(enable=True, tight=True)   # tighten axes limits around the data

    return(displ)

provec(60*5, aqua_level)
plt.scatter(0,0,marker='x',color='black',s=50)
plt.scatter(0,0,marker='x',color='black',s=50)

#%% plot u,v, vs depth
plt.figure()
plt.scatter(aqua['u'],aqua['DEPTH'])
plt.scatter(aqua['v'],aqua['DEPTH'])
plt.plot(aqua['u'],aqua['DEPTH'],linewidth=0.4,label='u')
plt.plot(aqua['v'],aqua['DEPTH'],linewidth=0.4,label='v')
plt.gca().invert_yaxis()
plt.legend()

plt.figure()
plt.scatter(aqua['u_corr'],aqua['DEPTH'])
plt.scatter(aqua['v_corr'],aqua['DEPTH'])
plt.plot(aqua['u_corr'],aqua['DEPTH'],linewidth=0.4,label='u_corr')
plt.plot(aqua['v_corr'],aqua['DEPTH'],linewidth=0.4,label='v_corr')
plt.scatter(aqua['u'],aqua['DEPTH'],color='blue',alpha=0.2)
plt.scatter(aqua['v'],aqua['DEPTH'],color='orange',alpha=0.2)
plt.plot(aqua['u'],aqua['DEPTH'],linewidth=0.4,label='u',color='blue',alpha=0.2)
plt.plot(aqua['v'],aqua['DEPTH'],linewidth=0.4,label='v',color='orange',alpha=0.2)
plt.gca().invert_yaxis()
plt.legend()

fig,ax1 = plt.subplots()
ax2 = ax1.twiny()
ax1.scatter(aqua['phi'],aqua['DEPTH'])
ax2.scatter(aqua['r'],aqua['DEPTH'],color='orange')
ax1.plot(aqua['phi'],aqua['DEPTH'],linewidth=0.4)
ax2.plot(aqua['r'],aqua['DEPTH'],linewidth=0.4,color='orange')
plt.gca().invert_yaxis()
ax1.set_xlabel('phi')
ax2.set_xlabel('r')
ax1.xaxis.label.set_color('blue')
ax2.xaxis.label.set_color('orange')

fig,ax1 = plt.subplots()
ax2 = ax1.twiny()
ax1.scatter(aqua['phi_corr'],aqua['Pressure(m)'])
ax2.scatter(aqua['r_corr'],aqua['Pressure(m)'],color='orange')
ax1.plot(aqua['phi_corr'],aqua['Pressure(m)'],linewidth=0.4)
ax2.plot(aqua['r_corr'],aqua['Pressure(m)'],linewidth=0.4,color='orange')
plt.gca().invert_yaxis()
ax1.set_xlabel('phi_corr')
ax2.set_xlabel('r_corr')
ax1.xaxis.label.set_color('blue')
ax2.xaxis.label.set_color('orange')

#%% separate casts
from hyvent.processing import sep_casts
aqua = aqua.rename(columns={'Pressure(m)':'DEPTH'})

cast_list = sep_casts(aqua, window_size=10)
aqua = cast_list[-1]

for cast in cast_list:
    plt.plot(cast['DEPTH'])



#%% look for suspicious effects in signal strength? and vertical velocity
plt.figure()
plt.plot(aqua_level['Amplitude(Beam1)(counts)'])
plt.plot(aqua_level['Amplitude(Beam2)(counts)'])
plt.plot(aqua_level['Amplitude(Beam3)(counts)'])

plt.figure()
plt.plot(aqua_level['Velocity(Beam3|Z|Up)(m/s)'])
