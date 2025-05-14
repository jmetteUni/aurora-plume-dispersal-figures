#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  4 12:13:38 2025

@author: jmette@uni-bremen.de
"""

import xarray as xa
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.ticker as ticker
import numpy as np
import os
import cartopy.crs as ccrs
#from roms_tools import Grid, ROMSOutput

import os
working_dir = '/home/jmette/roms_project/data_backup/'
os.chdir(working_dir)
run_dir = working_dir+'aurora-0-he_1month/'
bhf_path = '/home/jmette/roms_project/aurora-0-he/Data/grid-M256L256_bhflux_180MW_20220726T00.nc'
bpf_path = '/home/jmette/roms_project/aurora-0-he/Data/grid-M256L256_bpflux_20220726T00.nc'
grd_path = '/home/jmette/roms_project/aurora-0-he/Data/grid-M256L256.nc'


fig_path = run_dir+'figures/'

bhf = xa.open_dataset(bhf_path,engine='scipy')
bpf = xa.open_dataset(bpf_path,engine='scipy')
grd = xa.open_dataset(grd_path,engine='scipy')

avg = xa.open_dataset(run_dir+'roms_avg.nc',engine='scipy')
avg = avg.chunk({'ocean_time': 1})

vent_loc = (-6.255333333333334, 82.89716666666666, 3000)
mooring_loc = (-6.2507166666666665, 82.89776666666667)

list_of_vars = list(avg.keys())


def calc_z(ds):
    #with mean sea surface
    zeta = ds.zeta.mean()
    Zo_rho = (ds.hc * ds.s_rho + ds.Cs_r * ds.h) / (ds.hc + ds.h)
    z_rho = zeta + (zeta+ ds.h) * Zo_rho
    ds.coords['z_rho'] = z_rho.transpose()
    #ds.coords['z_rho'] = -ds.z_rho

    return ds

def rot_vel(u,v,grd_angle):
    r = abs(u+1j*v)
    phi = np.angle(u+1j*v)

    phi = phi + grd_angle

    u_rot = r * np.cos(phi)
    v_rot = r * np.sin(phi)

    return (u_rot,v_rot)

#avg.coords["z_rho"] = z_rho.transpose()  # needing transpose seems to be an xarray bug
#test = avg.set_index(s_rho='z_rho')
avg = calc_z(avg)
avg.coords['h_neg'] = -avg.h


#grid rotation
#grd.angle
#i,j index, see https://www.myroms.org/wiki/Grid_Generation

# u(XI,ETA)=u(LON,LAT)*cos(angle(i,j))+v(LON,LAT)*sin(angle(i,j))
# v(XI,ETA)=u(LON,LAT)*sin(angle(i,j))-v(LON,LAT)*cos(angle(i,j))

# or

# u(XI,ETA)=u(LON,LAT)*cos(angle(i,j))+v(LON,LAT)*sin(angle(i,j))
# v(XI,ETA)=-u(LON,LAT)*sin(angle(i,j))+v(LON,LAT)*cos(angle(i,j))


#%% print fig

print_fig = True
#%% get source coords

avg.isel(ocean_time=0,eta_rho=0,lon_rho=0).dye_01.idxmin()
#%% plot bottom forcing

bpf.isel(ocean_time=0).dye_01_bflux.plot()
#bpf.where(bpf==bpf.min(), drop=True).dye_01_bflux.plot()

#%% plot bathymetry

plt.figure(figsize=(6,4))

avg.coords['h_neg'] = -avg.h
avg.h_neg.plot.contourf(x='lon_rho',y='lat_rho',cmap='viridis',levels=40,cbar_kwargs={'label': "Depth in m"})
avg.h_neg.plot.contour(x='lon_rho',y='lat_rho',colors='black',alpha=0.3,linewidths=0.5,levels=40)
plt.scatter(vent_loc[0],vent_loc[1],edgecolors='red',c='red',marker='*',s=400,label='Aurora Vent Site')

plt.xlabel('Longitude')
plt.ylabel('Latitude')

plt.tight_layout()
if print_fig == True:
    plt.savefig(fig_path+'domain_bathy.png', dpi=300)
plt.show()

#print domain corners
print(avg.lon_rho.min().values)
print(avg.lon_rho.max().values)
print(avg.lat_rho.min().values)
print(avg.lat_rho.max().values)

#%% timeseries

source = avg.isel(ocean_time=0,s_rho=0)
source.dye_01.max()

#%% section
#https://docs.xarray.dev/en/stable/examples/ROMS_ocean_model.html

section = avg.isel(xi_rho = 125, ocean_time=200)

section.dye_01.plot(x='lon_rho', y='z_rho')


#%% vertical profile at one coor

profile = avg.isel(xi_rho=120,eta_rho=130).resample(ocean_time='1D').mean()
profile = calc_z(profile)

timesteps = (np.linspace(0,len(profile.ocean_time)-1,31)).astype(int)

plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.inferno(np.linspace(0,1,len(timesteps))))

fig, ax = plt.subplots(figsize=(7,6))
for i,step in enumerate(timesteps):
    step_label = profile.ocean_time[i].values.astype(str).removesuffix('T00:00:00.000000000')
    profile.isel(ocean_time=i).dye_01.plot(y='z_rho', ax=ax)

# colorbar
normalize = mcolors.Normalize(vmin=0, vmax=len(timesteps)-1)
sm = plt.cm.ScalarMappable(cmap='inferno', norm=normalize)
sm.set_array(np.arange(len(timesteps)))

cbar = plt.colorbar(sm, ax=ax)
#cbar.set_label('Time step')
cbar.set_ticks(np.linspace(0, len(timesteps)-1, 5))

tick_labels = [profile.ocean_time[int(tick)].values.astype(str).removesuffix('T00:00:00.000000000') for tick in cbar.get_ticks()]
cbar.set_ticklabels(tick_labels)

ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{abs(x):.0f}' if x!= 0 else '0'))

ax.set_ylabel('Depth in m')
ax.set_xlabel('Tracer concentration in kg m$^{-3}$')
ax.set_title(None)

if print_fig == True:
    plt.savefig(fig_path+'depth_dye_full.png', dpi=300)
plt.show()

#%% potential temperature

fig, ax = plt.subplots(figsize=(7,6))
for i,step in enumerate(timesteps):
    step_label = profile.ocean_time[i].values.astype(str).removesuffix('T00:00:00.000000000')
    profile.isel(ocean_time=i).temp.plot(y='z_rho', ax=ax)

# colorbar
normalize = mcolors.Normalize(vmin=0, vmax=len(timesteps)-1)
sm = plt.cm.ScalarMappable(cmap='inferno', norm=normalize)
sm.set_array(np.arange(len(timesteps)))

cbar = plt.colorbar(sm, ax=ax)
#cbar.set_label('Time step')
cbar.set_ticks(np.linspace(0, len(timesteps)-1, 5))

tick_labels = [profile.ocean_time[int(tick)].values.astype(str).removesuffix('T00:00:00.000000000') for tick in cbar.get_ticks()]
cbar.set_ticklabels(tick_labels)

ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{abs(x):.0f}' if x!= 0 else '0'))

ax.set_ylabel('Depth in m')
ax.set_xlabel(r'$\Theta$ in $^{\circ}$C')
ax.set_title(None)

if print_fig == True:
    plt.savefig(fig_path+'depth_potemp_full.png', dpi=300)
plt.show()


#%% density anomaly

fig, ax = plt.subplots(figsize=(7,6))
for i,step in enumerate(timesteps):
    step_label = profile.ocean_time[i].values.astype(str).removesuffix('T00:00:00.000000000')
    profile.isel(ocean_time=i).rho.plot(y='z_rho', ax=ax)

# colorbar
normalize = mcolors.Normalize(vmin=0, vmax=len(timesteps)-1)
sm = plt.cm.ScalarMappable(cmap='inferno', norm=normalize)
sm.set_array(np.arange(len(timesteps)))

cbar = plt.colorbar(sm, ax=ax)
#cbar.set_label('Time step')
cbar.set_ticks(np.linspace(0, len(timesteps)-1, 5))

tick_labels = [profile.ocean_time[int(tick)].values.astype(str).removesuffix('T00:00:00.000000000') for tick in cbar.get_ticks()]
cbar.set_ticklabels(tick_labels)

ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{abs(x):.0f}' if x!= 0 else '0'))

ax.set_ylabel('Depth in m')
ax.set_xlabel(r'$\Theta$ in $^{\circ}$C')
ax.set_title(None)

if print_fig == True:
    plt.savefig(fig_path+'depth_potemp_full.png', dpi=300)
plt.show()


#%% concentration per time

profile_depth = avg.isel(xi_rho=120,eta_rho=130).resample(ocean_time='1h').mean()
profile_depth = calc_z(profile_depth)

profile_depth['dye_01'] = profile_depth.dye_01.where(profile_depth.z_rho<-2800,np.nan)

plt.figure(figsize=(8,3))

profile_depth.dye_01.plot(x='ocean_time',y='z_rho',cmap='Oranges',cbar_kwargs={'label':'Tracer concentration in kg m$^{-3}$'})
plt.ylim(profile.z_rho.min(),-2800)

plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{abs(x):.0f}' if x!= 0 else '0'))

plt.ylabel('Depth in m')
plt.xlabel(None)
plt.title(None)
plt.tight_layout()
if print_fig == True:
    plt.savefig(fig_path+'dye_per_time.png', dpi=300)
plt.show()

# profile_depth['temp'] = profile_depth.temp.where(profile_depth.z_rho<-2800,np.nan)

# plt.figure(figsize=(8,3))

# profile_depth.temp.plot(x='ocean_time',y='z_rho',cmap='Reds',cbar_kwargs={'label':r'$\Theta$ in $^{\circ}$C'})
# plt.ylim(profile.z_rho.min(),-2800)

# plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{abs(x):.0f}' if x!= 0 else '0'))

# plt.ylabel('Depth in m')
# plt.xlabel(None)
# plt.title(None)
# plt.tight_layout()
# if print_fig == True:
#     plt.savefig(fig_path+'temp_per_time.png', dpi=300)
# plt.show()


#%% horizontal
#boundaries of observation plots
lonW = -6.45
lonE = -6.175
latS = 82.875
latN = 82.92

date_list = ['2022-07-26','2022-07-28','2022-07-29','2022-07-31','2022-08-02','2022-08-25']

contour = avg.resample(ocean_time='1D').mean()

#contour['dye_01'] = contour.dye_01.where((avg.z_rho>-3800) & (avg.z_rho<-2500),np.nan)
#slice for plume depth
contour = contour.sel(s_rho=slice(-0.9,-0.55)).sum(dim='s_rho')

vmin = contour.dye_01.min().values
#vmax = contour.dye_01.max().values
vmax = 1000

for i, step in enumerate(contour.ocean_time):
    step_label = contour.ocean_time[i].values.astype(str).rstrip('T00:00:00.000000000')
    if step_label in date_list:
        if step_label in ['2022-07-28','2022-07-31','2022-08-25']:
            cbar_kwargs = {'label':'Tracer concentration in kg m$^{-3}$'}
            add_colorbar = True
            plt.figure(figsize=(5,4))
        else:
            cbar_kwargs = {'label':'Tracer concentration in kg m$^{-3}$'}
            add_colorbar = True
            plt.figure(figsize=(5,4))
        contour.isel(ocean_time=i).dye_01.plot.pcolormesh(x='lon_rho',y='lat_rho',cmap='Oranges',vmin=vmin,vmax=vmax,cbar_kwargs=cbar_kwargs,add_colorbar=add_colorbar)
        contourlines = contour.isel(ocean_time=i).dye_01.plot.contour(x='lon_rho',y='lat_rho',colors='black',alpha=0.3,levels=5)
        plt.clabel(contourlines, inline=True, fontsize=10, fmt='%d', colors = 'black')

        #plot observation reference
        plt.scatter(lonW,latN,marker='+',color='grey')
        plt.scatter(lonW,latS,marker='+',color='grey')
        plt.scatter(lonE,latN,marker='+',color='grey')
        plt.scatter(lonE,latS,marker='+',color='grey')

        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title(None)

        plt.tight_layout()
        if print_fig == True:
            if step_label in date_list:
                plt.savefig(fig_path+step_label+'_2D_dye.png', dpi=300)
        plt.show()

#bathymetry overlay
#contourlines = avg.h_neg.plot.contour(x='lon_rho',y='lat_rho',colors='black',alpha=0.5,linewidths=0.5,levels=40)
#plt.clabel(contourlines, inline=True, fontsize=6, fmt='%d', colors = 'black')

#%% currents timeseries
#rcm474 at 3572 :s_rho=-0.783333

currents = avg.sel(s_rho=-0.7833333,method='nearest')

plt.figure(figsize=(8,3))

u = currents.u.isel(xi_u=120,eta_u=130)#*100     #in cm/s
v = currents.v.isel(xi_v=120,eta_v=130)#*100
angle = grd.angle.isel(xi_rho=120,eta_rho=130)

#rotating
u_lonlat,v_lonlat = rot_vel(u,v,angle.values)

u_lonlat = u_lonlat*100     #in cm/s
v_lonlat = v_lonlat*100

u_lonlat.plot(color='tab:blue',label='u')
v_lonlat.plot(color='tab:orange',label='v')

plt.legend()
plt.ylabel('Velocity in cm s$^{-1}$')
plt.xlabel(None)
plt.title(None)
plt.tight_layout()
if print_fig == True:
    plt.savefig(fig_path+'timeseries_modell_at474.png',dpi=300)
plt.show()

#%% progressive vector

#rcm512 at 3064 :s_rho=-0.65
#rcm506 at 3215 :s_rho=-0.683333
#rcm569 at 3366 :s_rho=-0.75
#rcm474 at 3572 :s_rho=-0.783333

currents512 = avg.sel(s_rho=-0.65,method='nearest')
currents506 = avg.sel(s_rho=-0.683333,method='nearest')
currents569 = avg.sel(s_rho=-0.75,method='nearest')
currents474 = avg.sel(s_rho=-0.7833333,method='nearest')

plt.figure(figsize=(7,6))
dt=4

angle = grd.angle.isel(xi_rho=120,eta_rho=130).values

u, v = rot_vel(currents512.u.isel(xi_u=120,eta_u=130).values, currents512.v.isel(xi_v=120,eta_v=130).values, angle)

u_cum = np.cumsum(u)*dt/1000
v_cum = np.cumsum(v)*dt/1000
plt.plot(u_cum,v_cum,color='blue',label='512 at 3064 m')

u, v = rot_vel(currents506.u.isel(xi_u=120,eta_u=130).values, currents506.v.isel(xi_v=120,eta_v=130).values, angle)

u_cum = np.cumsum(u)*dt/1000
v_cum = np.cumsum(v)*dt/1000
plt.plot(u_cum,v_cum,color='orange',label='506 at 3215 m')

u, v = rot_vel(currents569.u.isel(xi_u=120,eta_u=130).values, currents569.v.isel(xi_v=120,eta_v=130).values, angle)

u_cum = np.cumsum(u)*dt/1000
v_cum = np.cumsum(v)*dt/1000
plt.plot(u_cum,v_cum,color='olive',label='569 at 3366 m')

u, v = rot_vel(currents474.u.isel(xi_u=120,eta_u=130).values, currents474.v.isel(xi_v=120,eta_v=130).values, angle)

u_cum = np.cumsum(u)*dt/1000
v_cum = np.cumsum(v)*dt/1000
plt.plot(u_cum,v_cum,color='red',label='474 at 3572 m')

plt.scatter(0,0,marker='+',color='black',s=100)

plt.xlim(-0.15,0.15)

plt.gca().set_aspect('equal')
plt.xlabel('Displacement East - West in km')
plt.ylabel('Displacement North - South in km')
plt.legend(['512 at \n 3064 m','506 at \n 3215 m','569 at \n 3366 m','474 at \n 3572 m'],ncol=1)
plt.tight_layout()
if print_fig == True:
    plt.savefig(fig_path+'prog_vector_modell_at474.png',dpi=300)
plt.show()


#%%

section = avg.isel(xi_rho = 120, ocean_time=100)

section.dye_01.plot(x='lat_rho', y='z_rho',cmap='Oranges')



#%%

