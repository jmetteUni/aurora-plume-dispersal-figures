#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 11:40:23 2025

@author: jmette@uni-bremen.de
"""

import os
working_dir = '/home/jonathan/Dokumente/Masterarbeit/python/thesis'
os.chdir(working_dir)

from lat_lon_parser import parse

from hyvent.io import (read_from_csv,
                       read_gebco)
from hyvent.plotting_maps import *
from hyvent.plotting import *
from hyvent.misc import keys_to_data, dist, add_castno, get_var
from hyvent.processing import *
from hyvent.quality_control import qc_IQR

import pandas as pd
import gsw
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

#%% set paths
cnv_path = '/home/jonathan/Dokumente/SHK Maren/PS137/CTD_data_processing/csv_files_noprocessing/'
btl_path = '/home/jonathan/Dokumente/SHK Maren/PS137/CTD_data_processing/csv_files_noprocessing/'
mapr_path = '/home/jonathan/Dokumente/SHK Maren/PS137/MAPR/csv_files/'
gebcopath = '/home/jonathan/Dokumente/SHK Maren/PS137/PS137_AuroraVent_25m_bilinear_WGS84.nc'     # path for bathymetry files
gebcopath_hillshade = '/home/jonathan/Dokumente/SHK Maren/PS137/PS137_AuroraVent_25m_bilinear_WGS84_hillshade.nc'
fig_path = '/home/jonathan/Dokumente/Masterarbeit/Thesis_Latex_Project/figures/5-disc/'

lonW = -6.45
lonE = -6.175
latS = 82.875
latN = 82.92
vent_loc = (parse("6° 15.32'W"),parse("82° 53.83'N"),3900)
mooring_loc =  (parse("6° 15.043'W"),parse("82° 53.866'N"))
aurora_stations = ['022_01', '026_01', '028_01', '033_01', '036_01', '041_01', '054_01', '055_01']

#%% import data
btl_data = read_from_csv(btl_path, 'btl')
profile_data = read_from_csv(cnv_path, 'cnv')
mapr_data = read_from_csv(mapr_path, 'mapr')
bathy = read_gebco(gebcopath,lonE,lonW,latS,latN)

profile_data = keys_to_data(profile_data, 'cnv')
btl_data = keys_to_data(btl_data, 'btl')
mapr_data = keys_to_data(mapr_data, 'mapr')

#remove level part from 018 and 028 (no reindexing, maybe needed?) and stores it separetely -> Todo also MAPR
sta018 = profile_data['PS137_018_01']
sta028 = profile_data['PS137_028_01']
sta018_level = sta018.loc[(sta018['datetime'] > '2023-07-01 02:20:00') & (sta018['datetime'] < '2023-07-01 04:03:00')]
sta028_level = sta028.loc[(sta028['datetime'] > '2023-07-06 00:39:00') & (sta028['datetime'] < '2023-07-06 03:11:00')]
sta018 = pd.concat([sta018[sta018['datetime'] <= '2023-07-01 02:20:00'],sta018[sta018['datetime'] >= '2023-07-01 04:03:00']])
sta028 = pd.concat([sta028[sta028['datetime'] <= '2023-07-06 00:39:00'],sta028[sta028['datetime'] >= '2023-07-06 03:11:00']])

profile_data['PS137_018_01'] = sta018
profile_data['PS137_028_01'] = sta028
profile_data['PS137_018_le'] = sta018_level
profile_data['PS137_028_le'] = sta028_level

#subset mapr data in this period for analyis
mapr018_72 = mapr_data['PS137_CTD18-1_72']
mapr018_74 = mapr_data['PS137_CTD18-1_74']
mapr028_73 = mapr_data['PS137_CTD28-1_73']
mapr018_72_level = mapr018_72.loc[(mapr018_72['datetime'] > '2023-07-01 02:20:00') & (mapr018_72['datetime'] < '2023-07-01 04:03:00')]
mapr018_74_level = mapr018_74.loc[(mapr018_74['datetime'] > '2023-07-01 02:20:00') & (mapr018_74['datetime'] < '2023-07-01 04:03:00')]
mapr028_73_level = mapr028_73.loc[(mapr028_73['datetime'] > '2023-07-06 00:39:00') & (mapr028_73['datetime'] < '2023-07-06 03:11:00')]

profile_data = keys_to_data(profile_data, 'cnv')
btl_data = keys_to_data(btl_data, 'btl')
mapr_data = keys_to_data(mapr_data, 'mapr')

profile_data = pd.concat(profile_data.values(),ignore_index=True)
btl_data = pd.concat(btl_data.values(),ignore_index=True)
mapr_data = pd.concat(mapr_data.values(),ignore_index=True)

#%% processing

profile_data = derive_CTD(profile_data)
mapr_data = mapr_data.rename(columns={'Press(dB)':'PRES','Temp(°C)':'TEMP','Depth_corr(m)':'DEPTH'})
mapr_data = mapr_data[['datetime','PRES','TEMP','DEPTH','Neph(volts)','Dship_lon','Dship_lat','CTD_lon','CTD_lat','dORP','Cruise','Station','SN','Operation']]
mapr_data = derive_mapr(mapr_data, profile_data, aurora_stations)

#%% unify profile and mapr

mapr_formerge = mapr_data[mapr_data['Operation']=='CTD']
profile_mapr = pd.concat((profile_data,mapr_formerge),axis=0)
profile_mapr['datetime'] = pd.to_datetime(profile_mapr['datetime'])

#%% sort stations
profile_background = profile_data[profile_data['Station']=='018_01']
btl_background = btl_data[btl_data['Station']=='018_01']
mapr_background = mapr_data[mapr_data['Station']=='018_01']

profile_explo = profile_data[profile_data['Station']=='049_01']
mapr_explo = mapr_data[mapr_data['Station']=='049_01']
btl_explo = btl_data[mapr_data['Station']=='049_01']

profile_data = profile_data[profile_data['Station'].isin(aurora_stations)]
btl_data = btl_data[btl_data['Station'].isin(aurora_stations)]
mapr_data = mapr_data[mapr_data['Station'].isin(aurora_stations)]

#%% calc deltas for profile
min_dep = 2000
max_dep = 4600

control_plot =False

dens_var = 'Sigma3'
fit_cast_no = {'022_01':0,'026_01':0,'028_01':0,'033_01':18,'036_01':11,'041_01':1,'054_01':0,'055_01':1}
dens_cut_arr = {'022_01':41.98335,'026_01':41.98335,'028_01':41.98262,'033_01':41.98355,'036_01':41.98355,'041_01':41.98350,'054_01':41.98259,'055_01':41.98273}
data_list = [d for _, d in profile_data.groupby(['Station'])]
for i, station in enumerate(data_list):
    station_name = station['Station'].iloc[0]
    dens_cut = dens_cut_arr[station_name]
    fit_cast = get_fit_cast(station, dens_var, min_dep, window_size=1000, control_plot=control_plot)[fit_cast_no[station_name]]
    data_list[i] = calc_delta_densfit(station, dens_var, dens_cut, 2500, fit_cast, fit_order=3, control_plot=control_plot)
profile_delta_potemperature_dens = pd.concat(data_list)

var = 'potemperature'
data_list = [d for _, d in profile_data.groupby(['Station'])]
for i, station in enumerate(data_list):
    data_list[i] = calc_delta_by_bgfit(station, profile_background, var, min_dep, max_dep, fit='poly',param=(10),control_plot=control_plot)
profile_delta_potemperature_bg = pd.concat(data_list)

var = 'Sigma3'
data_list = [d for _, d in profile_data.groupby(['Station'])]
for i, station in enumerate(data_list):
    data_list[i] = calc_delta_by_bgfit(station, profile_background, var, min_dep, max_dep, fit='poly',param=(10),control_plot=control_plot)
profile_delta_sigma3 = pd.concat(data_list)

var = 'PSAL'
data_list = [d for _, d in profile_data.groupby(['Station'])]
for i, station in enumerate(data_list):
    data_list[i] = calc_delta_by_bgfit(station, profile_background, var, min_dep, max_dep, fit='poly',param=(10),control_plot=control_plot)
profile_delta_psal = pd.concat(data_list)

var = 'delta3He'
data_list = [d for _, d in btl_data.groupby(['Station'])]
for i, station in enumerate(data_list):
    data_list[i] = calc_helium_delta(station, btl_background, var, 990, max_dep,control_plot=control_plot)
btl_delta_he = pd.concat(data_list)

#%% cal deltas for mapr
dens_var = 'Sigma3'
fit_cast_no = {'026_01;73':0,'028_01;73':8,'033_01;72':18,'033_01;73':19,'033_01;74':19,'036_01;72':20,'036_01;73':10,'036_01;74':10,'041_01;74':1,'054_01;73':1,'054_01;74':1,'055_01;73':0}
dens_cut_arr = {'026_01;73':45,'028_01;73':41.98363,'033_01;72':45,'033_01;73':45,'033_01;74':45,'036_01;72':41.98437,'036_01;73':45,'036_01;74':45,'041_01;74':41.98431,'054_01;73':45,'054_01;74':45,'055_01;73':45}
data_list = [d for _, d in mapr_data.groupby(['Station','SN'])]
for i, station in enumerate(data_list):
    station_name = station['Station'].iloc[0]+';'+station['SN'].iloc[0]
    dens_cut = dens_cut_arr[station_name]
    fit_cast = get_fit_cast(station, dens_var, min_dep, window_size=400, control_plot=control_plot)[fit_cast_no[station_name]]
    data_list[i] = calc_delta_densfit(station, dens_var, dens_cut, 2500, fit_cast, fit_order=3, control_plot=control_plot)
mapr_delta_potemperature_dens = pd.concat(data_list)

#delta potemperature by bg
# var = 'potemperature'
# data_list = [d for _, d in mapr_data.groupby(['Station','SN'])]
# for i, station in enumerate(data_list):
#     data_list[i] = calc_delta_by_bgfit(station, mapr_background, var, min_dep, max_dep, fit='poly',param=(10),control_plot=control_plot)
# mapr_delta_potemperature_bg = pd.concat(data_list)
# mapr_delta_potemperature_bg = mapr_delta_potemperature_bg[(mapr_delta_potemperature_bg['SN']=='74') | (mapr_delta_potemperature_bg['SN']=='72')]

var = 'Sigma3'
data_list = [d for _, d in mapr_data.groupby(['Station','SN'])]
for i, station in enumerate(data_list):
    data_list[i] = calc_delta_by_bgfit(station, mapr_background, var, min_dep, max_dep, fit='poly',param=(10),control_plot=control_plot)
mapr_delta_sigma3 = pd.concat(data_list)
mapr_delta_sigma3 = mapr_delta_sigma3[(mapr_delta_sigma3['SN']=='74') | (mapr_delta_sigma3['SN']=='72')]

# var = 'Neph_outl(volts)'
# data_list = [d for _, d in mapr_data.groupby(['Station','SN'])]
# for i, station in enumerate(data_list):
#     data_list[i] = calc_delta_by_bgfit(station, mapr_background, var, min_dep, max_dep, fit='poly', param=(10),control_plot=control_plot)
# mapr_delta_turb = pd.concat(data_list)

#Turb deltas by steation mean for a smaller range
var = 'Neph_smoo(volts)'
data_list = [d for _, d in mapr_data.groupby(['Station','SN'])]
for i, station in enumerate(data_list):
    data_list[i] = calc_turb_delta(station, var, (1000,2700), (5500,6500))
mapr_delta_turb = pd.concat(data_list)


#%%#################### Print Figures Flag #####################################
print_fig = True
#%% compare ctd sensors


data_list = [d for _, d in profile_data.groupby(['Station'])]
plt.figure(figsize=(5, 4))
for i in data_list:
    i = i[i['DEPTH']>50]
    plt.scatter(i['TEMP'],i['TEMP2'],s=1,label=i['Station'].iloc[0],color='red')
    #plt.title(i['Station'].iloc[0])
#plt.legend()
plt.plot([0, 1], [0, 1], transform=plt.gca().transAxes,color='black')
plt.xlabel('Temperature (primary) in $^{\circ}$C')
plt.ylabel('Temperature in $^{\circ}$C')
plt.tight_layout()
plt.show()
if print_fig == True:
    plt.savefig(fig_path+'temp_compare.png',dpi=300)

#cond
plt.figure(figsize=(5, 4))
plt.scatter(profile_data['CNDC'][profile_data['DEPTH']>50],profile_data['CNDC2'][profile_data['DEPTH']>50],s=1,color='cyan')
plt.plot([0, 1], [0, 1], transform=plt.gca().transAxes,color='black')
plt.xlabel('Conductivity (primary) in S/m')
plt.ylabel('Conductivity in S/m')
plt.tight_layout()
plt.show()
if print_fig == True:
    plt.savefig(fig_path+'cndc_compare.png',dpi=300)


#%% separate MAPR baselines

data_list = [d for _, d in mapr_data.groupby(['SN'])]
plt.figure(figsize=(5, 4))
for device in data_list:
    plt.plot(device['Neph_smoo(volts)'],device['DEPTH'],label=device['SN'].iloc[0])
    #print(device['SN'].iloc[0]+': '+str(device['Neph_smoo(volts)'].mean()))
plt.legend()
plt.gca().invert_yaxis()
plt.xlabel(get_var('Neph_smoo(volts)')[0])
plt.ylabel(get_var('DEPTH')[0])
plt.tight_layout()
plt.show()
if print_fig == True:
    plt.savefig(fig_path+'mapr_baselines.pdf')

#%% MAPR background level

depth_plot(mapr_data, 'Neph_smoo(volts)','DEPTH', 200, background=mapr_explo)


#%% mean CTD salinity profile for MAPR

plt.figure(figsize=(5, 4))
data = profile_data[profile_data['DEPTH']>2000]
data_list = [d for _, d in data.groupby(['Station'])]
for station in data_list:
    plt.plot(station['PSAL'],station['DEPTH'],alpha=0.8)
    #print(station['Station'].iloc[0])

sal_mean = calc_mean_profile(data, 'PSAL', aurora_stations)
#plt.plot(sal_mean['PSAL_mean'],sal_mean['DEPTH'])
plt.gca().invert_yaxis()

#%% ORP drift

# station_list = [d for _, d in profile_data.groupby(['Station'])]
# for station in station_list:
station = profile_data[profile_data['Station']=='033_01']
station = station[station['DEPTH']>1000]
fig, ax = plt.subplots(figsize=(8, 3))
# fig.set_figwidth(12)
# fig.set_figheight(5)
ax.plot(station['datetime'],station['upoly0'],color='black',label='$E_h$ in mv')
ax.set_ylabel('$E_h$ in mv')
#ax.yaxis.label.set_color('black')

twin1 = ax.twinx()
twin1.plot(station['datetime'],station['dORP'],color=get_var('dORP')[1],label=get_var('dORP')[0],alpha=0.6)
twin1.set_ylabel(get_var('dORP')[0])
#twin1.yaxis.label.set_color('green')
twin1.axhline(y=0,color='grey')

ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
#ax.xaxis.set_major_formatter(majorFmt)
# major locator
xloc=mdates.MinuteLocator(byminute = [0,30,60])
ax.xaxis.set_major_locator(xloc)
ax.set_xlabel('Time')

fig.legend(bbox_to_anchor=(1,0.3), bbox_transform=ax.transAxes)

fig.set_tight_layout(True)
plt.show()
if print_fig == True:
    plt.savefig(fig_path+'orp_drift.pdf')


#%% sta018, sta028 spikes

#sta018 = sta018[sta018['DEPTH']>2000]
# fig, ax = plt.subplots()
# ax.plot(sta018_level['TEMP'],sta018_level['DEPTH'])
# #plt.plot(sta018_level['potemperature'],sta018_level['potemperature'])
# ax.plot(sta018['TEMP'],sta018['DEPTH'])

# # twin1 = ax.twiny()
# # twin1.plot(sta018_level['potemperature'],sta018_level['DEPTH'])
# # twin1.plot(sta018['potemperature'],sta018['DEPTH'])

# plt.gca().invert_yaxis()
# plt.show()

sta018_level['TEMP'].max()-sta018_level['TEMP'].min()

plt.figure(figsize=(5, 4))
plt.plot(sta018_level['datetime'],sta018_level['TEMP'], label='CTD (primary)')
plt.plot(sta018_level['datetime'],sta018_level['TEMP2'], label='CTD')
plt.plot(mapr018_72_level['datetime'],mapr018_72_level['Temp(°C)'], label='MAPR 72')
plt.plot(mapr018_74_level['datetime'],mapr018_74_level['Temp(°C)'], label='MAPR 74',color='m')

plt.ylim(mapr018_74_level['Temp(°C)'].min()-0.004,sta018_level['TEMP'].max()+0.006)
plt.ylabel(get_var('TEMP')[0])
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
plt.xlabel('Time')
plt.legend(loc='upper left')
plt.show()
plt.tight_layout()
if print_fig == True:
    plt.savefig(fig_path+'sta018_level_temp.pdf')


#sta028 = sta028[sta028['DEPTH']>2000]
# fig, ax = plt.subplots()
# ax.plot(sta028_level['TEMP'],sta028_level['DEPTH'])
# ax.plot(sta028_level['TEMP2'],sta028_level['DEPTH'])

# #plt.plot(sta028_level['potemperature'],sta028_level['potemperature'])
# ax.plot(sta028['TEMP'],sta028['DEPTH'])


# # twin1 = ax.twiny()
# # twin1.plot(sta028_level['potemperature'],sta028_level['DEPTH'])
# # twin1.plot(sta028['potemperature'],sta028['DEPTH'])

# plt.gca().invert_yaxis()
# plt.show()
sta028_level['TEMP'].max()-sta028_level['TEMP'].min()

plt.figure(figsize=(5, 4))
plt.plot(sta028_level['datetime'],sta028_level['TEMP'], label='CTD (primary)')
plt.plot(sta028_level['datetime'],sta028_level['TEMP2'], label='CTD')
plt.plot(mapr028_73_level['datetime'],mapr028_73_level['Temp(°C)'], label='MAPR 73')

plt.ylabel(get_var('TEMP')[0])
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
plt.xlabel('Time')
plt.legend(loc='upper left')
plt.show()
plt.tight_layout()
if print_fig == True:
    plt.savefig(fig_path+'sta028_level_temp.pdf')

#%% theta sigma fit for 018
min_dep = 2000
max_dep = 4600

control_plot =False

dens_var = 'Sigma3'
fit_cast_no = {'018_01':3}
dens_cut_arr = {'018_01':42}

station = profile_background
station_name = station['Station'].iloc[0]
dens_cut = dens_cut_arr[station_name]
fit_cast = get_fit_cast(station, dens_var, min_dep, window_size=1000, control_plot=True)[fit_cast_no[station_name]]
data_list[i] = calc_delta_densfit(station, dens_var, dens_cut, 2500, fit_cast, fit_order=3, control_plot=True)

#%% bg vs dens_fit

sta033_bg = profile_delta_potemperature_bg[profile_delta_potemperature_bg['Station']=='033_01']
sta033_dens = profile_delta_potemperature_dens[profile_delta_potemperature_dens['Station']=='033_01']
sta033_bg = sta033_bg[sta033_bg['DEPTH']>2500]
sta033_dens = sta033_dens[sta033_dens['DEPTH']>2500]

plt.figure()
plt.plot(sta033_dens['Delta_potemperature'],sta033_dens['DEPTH'],label='$\Delta$$\Theta$ by in-station method')
plt.plot(sta033_bg['Delta_potemperature'],sta033_bg['DEPTH'],label='$\Delta$$\Theta$ by background method')

plt.gca().invert_yaxis()
plt.xlabel(get_var('Delta_potemperature')[0])
plt.ylabel(get_var('DEPTH')[0])
plt.legend()
plt.show()
plt.tight_layout()
if print_fig == True:
    plt.savefig(fig_path+'bg_dens_method_comparison.pdf')

#%% 018 helium and turb

var = 'Neph_smoo(volts)'
data_list = [d for _, d in mapr_background.groupby(['Station','SN'])]
for i, station in enumerate(data_list):
    data_list[i] = calc_turb_delta(station, var, (1000,2700), (5500,6500))
mapr_bg_delta_turb = pd.concat(data_list)

#mapr_bg_delta_turb = mapr_bg_delta_turb[mapr_bg_delta_turb['DEPTH']>]

fig, ax = plt.subplots(figsize=(5, 4))

ax.plot(mapr_bg_delta_turb['Delta_Neph_smoo(volts)'], mapr_bg_delta_turb['DEPTH'], color=get_var('Neph_smoo(volts)')[1],label=get_var('Delta_Neph_smoo(volts)')[0])
start, end = ax.get_xlim()
ax.xaxis.set_ticks(np.round(np.arange(start, end,0.003),decimals=3))
ax.set_xlabel(get_var('Delta_Neph_smoo(volts)')[0])
#ax.xaxis.label.set_color(get_var('Neph_smoo(volts)')[1])
ax.set_ylabel(get_var('DEPTH')[0])

twin1 = ax.twiny()
twin1.scatter(btl_background['delta3He'][btl_background['DepSM_mean']>500],btl_background['DepSM_mean'][btl_background['DepSM_mean']>500], color=get_var('delta3He')[1],label=get_var('delta3He')[0])
twin1.set_xlabel(get_var('delta3He')[0])
#twin1.xaxis.label.set_color(get_var('delta3He')[1])

fig.legend(bbox_to_anchor=(1,1), bbox_transform=ax.transAxes)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
if print_fig == True:
    plt.savefig(fig_path+'turb_he_comparison.pdf')

#%% heat flux with maximum

ref_station = profile_data[profile_data['Station']=='055_01']

zplume = {'theta_ctd':3400,'dORP_ctd':3400,'he':3100,'theta_mapr':3400,'dORP_mapr':3200,'turb_mapr':3200}
zmax = dict()
for key in zplume:
    zmax[key] = vent_loc[2]-zplume[key]
heat_flux = dict()
nsquared = dict()
for key in zmax:
    heat_flux[key], nsquared[key] = calc_flux(zmax[key], vent_loc[2], ref_station)
    heat_flux[key] = round(heat_flux[key]*10**(-6))
    nsquared[key] = round(nsquared[key],10)

zmax
heat_flux
nsquared


#%% calc distance

profile_data = add_castno(profile_data)
profile_delta_potemperature_dens = add_castno(profile_delta_potemperature_dens)
mapr_data = add_castno(mapr_data)
mapr_delta_potemperature_dens = add_castno(mapr_delta_potemperature_dens)
mapr_delta_turb = add_castno(mapr_delta_turb)

# T endmember:
#     St037 NUI44-IGT2 126 °C +-10°C, 651.89115%
#     St050 NUI45-IGT1 370 °C +-10°C, 685.52245%; 684.63171% double sample

potemperature_vent_bg = profile_data['potemperature'][profile_data['DEPTH'].between(3890,3910)].mean()

dist_potemperature = profile_delta_potemperature_dens[profile_delta_potemperature_dens['DEPTH']>2500]
#dist_potemperature['Dist'] = dist_potemperature.apply(lambda x: dist(x['CTD_lon'], x['CTD_lat'], vent_loc[0], vent_loc[1], x['DEPTH'], vent_loc[2]), axis=1)
dist_potemperature['Dist'] = dist_potemperature.apply(lambda x: dist(x['CTD_lon'], x['CTD_lat'], vent_loc[0], vent_loc[1]), axis=1)

dist_dORP = profile_data[profile_data['DEPTH']>2500]
#dist_dORP['Dist'] = dist_dORP.apply(lambda x: dist(x['CTD_lon'], x['CTD_lat'], vent_loc[0], vent_loc[1], x['DEPTH'], vent_loc[2]), axis=1)
dist_dORP['Dist'] = dist_dORP.apply(lambda x: dist(x['CTD_lon'], x['CTD_lat'], vent_loc[0], vent_loc[1]), axis=1)

dist_he = btl_data[btl_data['DepSM_mean']>2500]
#dist_he['Dist'] = dist_he.apply(lambda x: dist(x['CTD_lon'], x['CTD_lat'], vent_loc[0], vent_loc[1], x['DepSM_mean'], vent_loc[2]), axis=1)
dist_he['Dist'] = dist_he.apply(lambda x: dist(x['CTD_lon'], x['CTD_lat'], vent_loc[0], vent_loc[1]), axis=1)

dist_potemperature_mapr = mapr_delta_potemperature_dens[mapr_delta_potemperature_dens['DEPTH']>2500]
#dist_potemperature_mapr['Dist'] = dist_potemperature_mapr.apply(lambda x: dist(x['CTD_lon'], x['CTD_lat'], vent_loc[0], vent_loc[1], x['DEPTH'], vent_loc[2]), axis=1)
dist_potemperature_mapr['Dist'] = dist_potemperature_mapr.apply(lambda x: dist(x['CTD_lon'], x['CTD_lat'], vent_loc[0], vent_loc[1]), axis=1)

dist_dORP_mapr = mapr_data[mapr_data['DEPTH']>2500]
#dist_dORP_mapr['Dist'] = dist_dORP_mapr.apply(lambda x: dist(x['CTD_lon'], x['CTD_lat'], vent_loc[0], vent_loc[1], x['DEPTH'], vent_loc[2]), axis=1)
dist_dORP_mapr['Dist'] = dist_dORP_mapr.apply(lambda x: dist(x['CTD_lon'], x['CTD_lat'], vent_loc[0], vent_loc[1]), axis=1)

dist_turb_mapr = mapr_delta_turb[mapr_delta_turb['DEPTH']>2500]
#dist_turb_mapr['Dist'] = dist_turb_mapr.apply(lambda x: dist(x['CTD_lon'], x['CTD_lat'], vent_loc[0], vent_loc[1], x['DEPTH'], vent_loc[2]), axis=1)
dist_turb_mapr['Dist'] = dist_turb_mapr.apply(lambda x: dist(x['CTD_lon'], x['CTD_lat'], vent_loc[0], vent_loc[1]), axis=1)

#%% he ps86 distance

ps86_mat_path = '/home/jonathan/Dokumente/Masterarbeit/CTD_data_collection/PS86/ps86allctd.mat'
import scipy.io
ps86_ctd = scipy.io.loadmat(ps86_mat_path)

#btl data
btl_time = ps86_ctd['btim'].squeeze()
btl_delta3he = ps86_ctd['bd3he'].squeeze()
ps86_btl = pd.DataFrame(data=[btl_time,btl_delta3he]).T
ps86_btl = ps86_btl.rename(columns={ps86_btl.columns[0]:'Date/Time',ps86_btl.columns[1]:'delta3He'})
ps86_btl['Date/Time'] = pd.to_datetime(ps86_btl['Date/Time']-719529,unit='d').round('s')

depth = ps86_ctd['dpt']
lon = ps86_ctd['lon']
lat = ps86_ctd['lat']
station = ps86_ctd['stn']
mattime = ps86_ctd['tim']

nstation = np.arange(0,17,1)
ps86_ctd_dict = dict()
for n in nstation:
    stn = station[0,n]
    ps86_ctd_dict[stn] = pd.DataFrame(columns=['Depth_m','Longitude','Latitude','Event','Date/Time'])
    ps86_ctd_dict[stn]['Depth_m'] = depth[:,n]
    ps86_ctd_dict[stn]['Longitude'] = lon[:,n]
    ps86_ctd_dict[stn]['Latitude'] = lat[:,n]
    ps86_ctd_dict[stn]['Event'] = 'PS86/0'+str(stn)
    ps86_ctd_dict[stn]['Date/Time'] = mattime[:,n]

ps86_ctd = pd.concat(ps86_ctd_dict.values(),ignore_index=True,axis=0)
ps86_ctd['Date/Time'] = pd.to_datetime(ps86_ctd['Date/Time']-719529,unit='d').round('s')
ps86_ctd.dropna(subset=['Date/Time'],inplace=True)

ps86_he = pd.merge_asof(ps86_ctd, ps86_btl, by='Date/Time', direction='nearest')
ps86_he = ps86_he.dropna(subset=['delta3He'])
ps86_he = ps86_he[ps86_he['Latitude']<82.95]

dist_ps86_he = ps86_he[ps86_he['Depth_m']>2500]
dist_ps86_he['Dist'] = dist_ps86_he.apply(lambda x: dist(x['Longitude'], x['Latitude'], vent_loc[0], vent_loc[1]), axis=1)
#%%  helium per distance both cruises

fig, ax = plt.subplots(figsize=(5, 4))

ax.scatter(dist_he['Dist'],dist_he['delta3He'], marker='o', color = get_var('delta3He')[1],label='PS137')
ax.scatter(dist_ps86_he['Dist'],dist_ps86_he['delta3He'], marker='o', color = 'black',label='PS86')
ax.set_ylabel(get_var('delta3He')[0])

twin1 = ax.twinx()
twin1.scatter(dist_he['Dist'],dist_he['delta3He']/np.mean([651.89115, 685.52245, 684.63171])*100, marker='s', color = get_var('delta3He')[1],s=5)
twin1.scatter(dist_ps86_he['Dist'],dist_ps86_he['delta3He']/np.mean([651.89115, 685.52245, 684.63171])*100, marker='s', color = 'black',s=5)
twin1.set_ylabel('Dilution in %')

ax.set_xlabel('Vent Distance in m')
fig.legend(bbox_to_anchor=(1,1), bbox_transform=ax.transAxes)
plt.tight_layout()
plt.show()

if print_fig == True:
    plt.savefig(fig_path+'he_ps86_compare.png',dpi=300)

#%% calculate depth/distance variability in maximum anomalies

# stations = ['033_01','041_01','054_01']

# var = 'Delta_potemperature'
# station = '033_01'
#data = [profile_delta_potemperature_dens[profile_delta_potemperature_dens['Station'].isin(stations)
#try to dicuss variability in temp anomalies

potemp_anom_dep = pd.DataFrame()
dorp_anom_dep = pd.DataFrame()
he_anom_dep = pd.DataFrame()

data_list = [d for _, d in dist_potemperature.groupby(['Cast'])]
for cast in data_list:
    cast = cast[cast['DEPTH']>2500]
    cast['Dist_castmean'] = cast['Dist'].mean()
    potemp_anom_dep = pd.concat([potemp_anom_dep,cast[cast['Delta_potemperature']==cast['Delta_potemperature'].max()]])

data_list = [d for _, d in dist_dORP.groupby(['Cast'])]
for cast in data_list:
    cast = cast[cast['DEPTH']>2500]
    cast['Dist_castmean'] = cast['Dist'].mean()
    dorp_anom_dep = pd.concat([dorp_anom_dep,cast[cast['dORP']==cast['dORP'].min()]])

data_list = [d for _, d in dist_he.groupby(['Station'])]
for station in data_list:
    station = station[station['DepSM_mean']>2500]
    station['Dist_castmean'] = station['Dist'].mean()
    he_anom_dep = pd.concat([he_anom_dep,station[station['delta3He']==station['delta3He'].max()]])

potemp_anom_dep_mapr = pd.DataFrame()
dorp_anom_dep_mapr = pd.DataFrame()
turb_anom_dep_mapr = pd.DataFrame()

data_list = [d for _, d in dist_potemperature_mapr.groupby(['Cast'])]
for cast in data_list:
    cast = cast[cast['DEPTH']>2500]
    cast['Dist_castmean'] = cast['Dist'].mean()
    potemp_anom_dep_mapr = pd.concat([potemp_anom_dep_mapr,cast[cast['Delta_potemperature']==cast['Delta_potemperature'].max()]])

data_list = [d for _, d in dist_dORP_mapr.groupby(['Cast'])]
for cast in data_list:
    cast = cast[cast['DEPTH']>2500]
    cast['Dist_castmean'] = cast['Dist'].mean()
    dorp_anom_dep_mapr = pd.concat([dorp_anom_dep_mapr,cast[cast['dORP']==cast['dORP'].min()]])

data_list = [d for _, d in dist_turb_mapr.groupby(['Station'])]
for station in data_list:
    station = station[station['DEPTH']>2500]
    station['Dist_castmean'] = station['Dist'].mean()
    turb_anom_dep_mapr = pd.concat([turb_anom_dep_mapr,station[station['Delta_Neph_smoo(volts)']==station['Delta_Neph_smoo(volts)'].max()]])

#%%
#Define the bin size
bin_size = 50  # adjust this value to change the size of the bins

# Split the dataframes into bins
bins = pd.cut(dist_potemperature['Dist'], bins=np.arange(0, dist_potemperature['Dist'].max() + bin_size, bin_size))

# Group the data by the bins
grouped_potemp = dist_potemperature.groupby(bins)

# Apply the same operations to each group
potemp_anom_dep = pd.DataFrame()
for group_name, group_data in grouped_potemp:
    group_data = group_data[group_data['DEPTH'] > 2500]
    group_data['Dist_castmean'] = group_data['Dist'].mean()
    potemp_anom_dep = pd.concat([potemp_anom_dep, group_data[group_data['Delta_potemperature'] == group_data['Delta_potemperature'].max()]])

# Repeat the same process for the other dataframes
bins = pd.cut(dist_dORP['Dist'], bins=np.arange(0, dist_dORP['Dist'].max() + bin_size, bin_size))
grouped_dorp = dist_dORP.groupby(bins)
dorp_anom_dep = pd.DataFrame()
for group_name, group_data in grouped_dorp:
    group_data = group_data[group_data['DEPTH'] > 2500]
    group_data['Dist_castmean'] = group_data['Dist'].mean()
    dorp_anom_dep = pd.concat([dorp_anom_dep, group_data[group_data['dORP'] == group_data['dORP'].min()]])

bins = pd.cut(dist_he['Dist'], bins=np.arange(0, dist_he['Dist'].max() + bin_size, bin_size))
grouped_he = dist_he.groupby(bins)
he_anom_dep = pd.DataFrame()
for group_name, group_data in grouped_he:
    group_data = group_data[group_data['DepSM_mean'] > 2500]
    group_data['Dist_castmean'] = group_data['Dist'].mean()
    he_anom_dep = pd.concat([he_anom_dep, group_data[group_data['delta3He'] == group_data['delta3He'].max()]])

# Repeat the same process for the mapr dataframes
bins = pd.cut(dist_potemperature_mapr['Dist'], bins=np.arange(0, dist_potemperature_mapr['Dist'].max() + bin_size, bin_size))
grouped_potemp_mapr = dist_potemperature_mapr.groupby(bins)
potemp_anom_dep_mapr = pd.DataFrame()
for group_name, group_data in grouped_potemp_mapr:
    group_data = group_data[group_data['DEPTH'] > 2500]
    group_data['Dist_castmean'] = group_data['Dist'].mean()
    potemp_anom_dep_mapr = pd.concat([potemp_anom_dep_mapr, group_data[group_data['Delta_potemperature'] == group_data['Delta_potemperature'].max()]])

bins = pd.cut(dist_dORP_mapr['Dist'], bins=np.arange(0, dist_dORP_mapr['Dist'].max() + bin_size, bin_size))
grouped_dorp_mapr = dist_dORP_mapr.groupby(bins)
dorp_anom_dep_mapr = pd.DataFrame()
for group_name, group_data in grouped_dorp_mapr:
    group_data = group_data[group_data['DEPTH'] > 2500]
    group_data['Dist_castmean'] = group_data['Dist'].mean()
    dorp_anom_dep_mapr = pd.concat([dorp_anom_dep_mapr, group_data[group_data['dORP'] == group_data['dORP'].min()]])

bins = pd.cut(dist_turb_mapr['Dist'], bins=np.arange(0, dist_turb_mapr['Dist'].max() + bin_size, bin_size))
grouped_turb_mapr = dist_turb_mapr.groupby(bins)
turb_anom_dep_mapr = pd.DataFrame()
for group_name, group_data in grouped_turb_mapr:
    group_data = group_data[group_data['DEPTH'] > 2500]
    group_data['Dist_castmean'] = group_data['Dist'].mean()
    turb_anom_dep_mapr = pd.concat([turb_anom_dep_mapr, group_data[group_data['Delta_Neph_smoo(volts)'] == group_data['Delta_Neph_smoo(volts)'].max()]])

#%% max anomaly histogram
from matplotlib.ticker import MultipleLocator

plt.figure(figsize=(5, 4))
data = [potemp_anom_dep['DEPTH'],dorp_anom_dep['DEPTH']]
colors = [get_var('Delta_potemperature')[1],get_var('dORP')[1]]
labels = [get_var('Delta_potemperature')[0],get_var('dORP')[0]]

plt.hist(data,bins=14,range=(2600,4000), color=colors,label=labels)

plt.xlabel('Depth in m')
plt.ylabel('Count of maximum anomalies')
plt.legend(loc='upper left')
plt.gca().xaxis.set_minor_locator(MultipleLocator(100))

plt.tight_layout()
plt.show()
if print_fig == True:
    plt.savefig(fig_path+'profile_max_anomaly_depth_hist.png',dpi=300)

plt.figure(figsize=(5, 4))

data = [potemp_anom_dep_mapr['DEPTH'],dorp_anom_dep_mapr['DEPTH'],turb_anom_dep_mapr['DEPTH']]
colors = [get_var('Delta_potemperature')[1],get_var('dORP')[1],get_var('Delta_Neph_smoo(volts)')[1]]
labels = [get_var('Delta_potemperature')[0],get_var('dORP')[0],get_var('Delta_Neph_smoo(volts)')[0]]

plt.hist(data,bins=14,range=(2600,4000), color=colors,label=labels)

plt.xlabel('Depth in m')
plt.ylabel('Count of maximum anomalies')
plt.legend(loc='upper right')
plt.gca().xaxis.set_minor_locator(MultipleLocator(100))

plt.tight_layout()
plt.show()
if print_fig == True:
    plt.savefig(fig_path+'mapr_max_anomaly_depth_hist.png',dpi=300)

#%% plot depth/distance variability

fig, ax = plt.subplots(figsize=(10,4))

dorp = ax.scatter(dorp_anom_dep['Dist'],dorp_anom_dep['DEPTH'],c=dorp_anom_dep['dORP'],cmap=get_var('dORP')[2],edgecolor='grey',s=100,marker='s',label='Maximum '+get_var('dORP')[0]) #color=get_var('dORP')[1]
potemp = ax.scatter(potemp_anom_dep['Dist'],potemp_anom_dep['DEPTH'],c=potemp_anom_dep['Delta_potemperature'],cmap=get_var('Delta_potemperature')[2],edgecolor='grey',s=100,marker='d',label='Maximum '+get_var('Delta_potemperature')[0]) #color=get_var('Delta_potemperature')[1]
he = ax.scatter(dist_he['Dist'],dist_he['DepSM_mean'],c=dist_he['delta3He'],cmap=get_var('delta3He')[2],edgecolor='grey',s=100,label='Maximum '+get_var('delta3He')[0]) #color=get_var('delta3He')[1]
#he = ax.scatter(he_anom_dep['Dist_castmean'],he_anom_dep['DepSM_mean'],c=he_anom_dep['delta3He'],cmap=get_var('delta3He')[2],edgecolor='grey',s=100,label='Maximum '+get_var('delta3He')[0]) #color=get_var('delta3He')[1]

plt.colorbar(dorp,label=get_var('dORP')[0],pad=-0.02)
plt.colorbar(potemp,label=get_var('Delta_potemperature')[0],pad=-0.04)
plt.colorbar(he,label=get_var('delta3He')[0],pad=0.02)

ax.legend()

ax.set_ylabel(get_var('DEPTH')[0])
ax.set_xlabel('Vent Distance in m')
ax.invert_yaxis()

plt.tight_layout()
plt.show()
if print_fig == True:
    plt.savefig(fig_path+'max_anomaly_depth_per_cast.png',dpi=300)

#%% temp dilution

fig, ax = plt.subplots()

ax.scatter(0,370,label='Vent sample',color='black')
ax.scatter(dist_potemperature['Dist'],dist_potemperature['Delta_potemperature'], marker='x', color = get_var('Delta_potemperature')[1],label='CTD samples')
#plt.plot([0,603.557407851],[np.mean([651.89115, 685.52245, 684.63171]),8.9728765],linestyle='--',color = get_var('Delta_potemperature')[1],label='Linear dilution curve')

#dilution factor
twin1 = ax.twinx()
twin1.scatter(dist_potemperature['Dist'],dist_potemperature['Delta_potemperature']/370*100, marker='s', color = 'black',label='Dilution')
twin1.set_ylabel('Dilution in %')

ax.set_xlabel('Vent Distance in m')

fig.legend(bbox_to_anchor=(1,1), bbox_transform=ax.transAxes)

plt.legend()
plt.tight_layout()
plt.show()
if print_fig == True:
    plt.savefig(fig_path+'Delta_potemperature_dilution.png',dpi=300)


#%%cross corelation

he_potemperature = pd.merge(btl_data,profile_delta_potemperature_dens[['datetime','potemperature']],how='left',on='datetime')
he_dorp = pd.merge(btl_data,profile_delta_potemperature_dens[['datetime','dORP']])

plt.figure()
plt.scatter(he_potemperature['delta3He'],he_potemperature['potemperature'],color='black')
plt.xlabel(get_var('delta3He')[0])
plt.ylabel(get_var('Delta_potemperature')[0])
#plt.scatter(he_dorp['delta3He'],he_dorp['dORP'])

plt.tight_layout()
plt.show()
if print_fig == True:
    plt.savefig(fig_path+'cross_he_potemperature.png',dpi=300)

#%% detail station
import xarray as xa
da_bathy = xa.open_dataset(gebcopath)

sta041 = profile_delta_potemperature_dens[profile_delta_potemperature_dens['Station']=='033_01']

#plot3Ddistr(sta041, 'Delta_potemperature', 2500, bathy, vent_loc)
plot_contourf(sta041, 'Delta_potemperature', 'CTD_lat',2500, da_bathy, vent_loc)
if print_fig == True:
    plt.savefig(fig_path+'cross_interp_033.png',dpi=300)