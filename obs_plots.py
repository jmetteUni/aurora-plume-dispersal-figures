#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 14:58:50 2024

@author: jmette@uni.bremen.de
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

import numpy as np
import pandas as pd
import gsw
import matplotlib.pyplot as plt
import matplotlib.offsetbox as offsetbox

#%% set paths
cnv_path = '/home/jonathan/Dokumente/SHK Maren/PS137/CTD_data_processing/csv_files_noprocessing/'
btl_path = '/home/jonathan/Dokumente/SHK Maren/PS137/CTD_data_processing/csv_files_noprocessing/'
mapr_path = '/home/jonathan/Dokumente/SHK Maren/PS137/MAPR/csv_files/'
gebcopath = '/home/jonathan/Dokumente/SHK Maren/PS137/PS137_AuroraVent_25m_bilinear_WGS84.nc'     # path for bathymetry files
gebcopath_hillshade = '/home/jonathan/Dokumente/SHK Maren/PS137/PS137_AuroraVent_25m_bilinear_WGS84_hillshade.nc'
fig_path = '/home/jonathan/Dokumente/Masterarbeit/Thesis_Latex_Project/figures/4-results/4-1-spatial/'

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
mapr_data = mapr_data.resample('S').interpolate()

#%% unify profile and mapr

mapr_formerge = mapr_data[mapr_data['Operation']=='CTD']
profile_mapr = pd.concat((profile_data,mapr_formerge),axis=0)
profile_mapr['datetime'] = pd.to_datetime(profile_mapr['datetime'])

#%% sort stations
profile_background = profile_data[profile_data['Station']=='018_01']
btl_background = btl_data[btl_data['Station']=='018_01']
mapr_background = mapr_data[mapr_data['Station']=='018_01']

sta018_level = profile_data[profile_data['Station']=='018_le']
sta028_level = profile_data[profile_data['Station']=='028_le']

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
fit_cast_no = {'026_01;73':0,'028_01;73':8,'033_01;72':18,'033_01;73':20,'033_01;74':19,'036_01;72':20,'036_01;73':10,'036_01;74':10,'041_01;74':1,'054_01;73':1,'054_01;74':1,'055_01;73':0}
dens_cut_arr = {'026_01;73':45,'028_01;73':41.98363,'033_01;72':45,'033_01;73':45,'033_01;74':45,'036_01;72':41.98437,'036_01;73':45,'036_01;74':45,'041_01;74':41.98431,'054_01;73':45,'054_01;74':45,'055_01;73':45}
data_list = [d for _, d in mapr_data.groupby(['Station','SN'])]
for i, station in enumerate(data_list):
    station_name = station['Station'].iloc[0]+';'+station['SN'].iloc[0]
    dens_cut = dens_cut_arr[station_name]
    fit_cast = get_fit_cast(station, dens_var, min_dep, window_size=400, control_plot=control_plot)[fit_cast_no[station_name]]
    data_list[i] = calc_delta_densfit(station, dens_var, dens_cut, 2500, fit_cast, fit_order=3, control_plot=control_plot)
mapr_delta_potemperature_dens = pd.concat(data_list)

#delta potemperature by bg
var = 'potemperature'
data_list = [d for _, d in mapr_data.groupby(['Station','SN'])]
for i, station in enumerate(data_list):
    data_list[i] = calc_delta_by_bgfit(station, mapr_background, var, min_dep, max_dep, fit='poly',param=(10),control_plot=control_plot)
mapr_delta_potemperature_bg = pd.concat(data_list)
mapr_delta_potemperature_bg = mapr_delta_potemperature_bg[(mapr_delta_potemperature_bg['SN']=='74') | (mapr_delta_potemperature_bg['SN']=='72')]

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

#%% export csv for odv
# odv_out_path = '/home/jonathan/Dokumente/Masterarbeit/python/csv_for_odv/'

# profile_delta_potemperature_dens.to_csv(odv_out_path+'profile_delta_potemperature_dens.csv',index=False)


#%%#################### Print Figures Flag #####################################
print_fig = True
#%% ctd track plot

plot_track(profile_data, btl_data, vent_loc=mooring_loc, bathy=bathy)

#%% time profile plots

time_plot(profile_data, '036_01', 1000)
if print_fig == True:
    plt.savefig(fig_path+'timeplot_036_01.pdf')

#%% plot ts diagram

#all data
# plot_ts(profile_data, 'DEPTH', 2500, 5500, 3000, vent_loc[0], vent_loc[1])
# if print_fig == True:
#     plt.savefig(fig_path+'ts.pdf',dpi=200)

#by station
plot_ts(profile_data, 'Station', 2500, 5500, 3000, vent_loc[0], vent_loc[1])
if print_fig == True:
    plt.savefig(fig_path+'ts_bystation.png',dpi=300)

plt.show()


#by cast        #no sense, too many casts
# plot_ts(profile_data, 'Cast', 2500, 5500, 3000, vent_loc[0], vent_loc[1])
# if print_fig == True:
#     plt.savefig(fig_path+'ts_bycast.pdf',dpi=200)

#by radius
profile_data['Dist_vent'] = profile_data.apply(lambda x: dist(x['CTD_lon'], x['CTD_lat'], vent_loc[0], vent_loc[1], x['DEPTH'], vent_loc[2]), axis=1)
plot_ts(profile_data, 'Dist_vent', 2500, 5500, 3000, vent_loc[0], vent_loc[1])
if print_fig == True:
    plt.savefig(fig_path+'ts_distvent.png',dpi=300)

#%% plot TS Zoom

# plt.xlim(34.92396,34.92640)
# plt.ylim(-0.8892,-0.8745)

ts_zoom = profile_data[(34.92396<profile_data['PSAL']) & (profile_data['PSAL']<34.92640)]
ts_zoom = ts_zoom[(-0.8892<ts_zoom['potemperature']) & (ts_zoom['potemperature']<-0.8745)]

plot_ts_zoom(ts_zoom, 'Station', 2500, 5500, 3000, vent_loc[0], vent_loc[1])
plt.xlim(34.9238,34.92640)
plt.ylim(-0.8895,-0.8745)
if print_fig == True:
    plt.savefig(fig_path+'ts_bystation_zoom.png',dpi=300)

#by radius
ts_zoom['Dist_vent'] =ts_zoom.apply(lambda x: dist(x['CTD_lon'], x['CTD_lat'], vent_loc[0], vent_loc[1],x['DEPTH'], vent_loc[2] ), axis=1)
plot_ts_zoom(ts_zoom, 'Dist_vent', 2500, 5500, 3000, vent_loc[0], vent_loc[1])
plt.xlim(34.9238,34.92640)
plt.ylim(-0.8895,-0.8745)
if print_fig == True:
    plt.savefig(fig_path+'ts_distvent_zoom.png',dpi=300)


#%% CTD & MAPR depth profile plots

cutoff_depth = 2000

depth_plot(profile_data, 'potemperature','DEPTH', cutoff_depth, background=profile_background)
if print_fig == True:
    plt.savefig(fig_path+'profile_potemp_depth_bg.pdf',dpi=300)

depth_plot(profile_data, 'dORP','DEPTH', cutoff_depth, background=profile_background)
if print_fig == True:
    plt.savefig(fig_path+'profile_dORP_depth_bg.pdf',dpi=300)

depth_plot(profile_data, 'PSAL','DEPTH', cutoff_depth, background=profile_background)
if print_fig == True:
    plt.savefig(fig_path+'profile_psal_depth_bg.pdf',dpi=300)

depth_plot(profile_data, 'Sigma3','DEPTH', cutoff_depth, background=profile_background)
if print_fig == True:
    plt.savefig(fig_path+'profile_dens_depth_bg.pdf',dpi=300)

depth_plot(btl_data, 'delta3He', 'DepSM_mean', 990, background=btl_background)
if print_fig == True:
    plt.savefig(fig_path+'profile_he_depth_bg.pdf',dpi=300)

depth_plot(mapr_data,'potemperature','DEPTH', cutoff_depth, background=mapr_background)
if print_fig == True:
    plt.savefig(fig_path+'mapr_potemp_depth_bg.pdf',dpi=300)

depth_plot(mapr_data, 'dORP','DEPTH', cutoff_depth, background=mapr_background)
if print_fig == True:
    plt.savefig(fig_path+'mapr_dORP_depth_bg.pdf',dpi=300)

depth_plot(mapr_data, 'Sigma3','DEPTH', cutoff_depth, background=mapr_background)
if print_fig == True:
    plt.savefig(fig_path+'mapr_dens_depth_bg.pdf',dpi=300)

depth_plot(mapr_data, 'Neph_smoo(volts)','DEPTH', 1000, background=mapr_background)
if print_fig == True:
    plt.savefig(fig_path+'mapr_turbsmoo_depth_bg.pdf',dpi=300)

#%% CTD depth anomaly plots
cutoff_depth = 2500

depth_plot(profile_delta_potemperature_dens, 'Delta_potemperature', 'DEPTH', cutoff_depth)
if print_fig == True:
    plt.savefig(fig_path+'profile_delta_potemp_depth_dens.pdf',dpi=300)

depth_plot(profile_delta_potemperature_bg, 'Delta_potemperature', 'DEPTH', cutoff_depth)
if print_fig == True:
    plt.savefig(fig_path+'profile_delta_potemp_depth_bg.pdf',dpi=300)

depth_plot(profile_delta_sigma3, 'Delta_Sigma3', 'DEPTH', cutoff_depth)
if print_fig == True:
    plt.savefig(fig_path+'profile_delta_dens_depth_bg.pdf',dpi=300)

depth_plot(profile_delta_psal, 'Delta_PSAL', 'DEPTH', cutoff_depth)
if print_fig == True:
    plt.savefig(fig_path+'profile_delta_psal_depth_bg.pdf',dpi=300)

depth_plot(btl_delta_he, 'Delta_delta3He', 'DEPTH', 990)
if print_fig == True:
    plt.savefig(fig_path+'profile_delta_he_depth_bg.pdf',dpi=300)

# mapr depth anomaly plots
depth_plot(mapr_delta_potemperature_dens, 'Delta_potemperature', 'DEPTH', cutoff_depth)
if print_fig == True:
    plt.savefig(fig_path+'mapr_delta_potemp_depth_dens.pdf',dpi=300)

depth_plot(mapr_delta_potemperature_bg, 'Delta_potemperature', 'DEPTH', cutoff_depth)
if print_fig == True:
    plt.savefig(fig_path+'mapr_delta_potemp_depth_bg.pdf',dpi=300)

depth_plot(mapr_delta_sigma3, 'Delta_Sigma3', 'DEPTH', cutoff_depth)
if print_fig == True:
    plt.savefig(fig_path+'mapr_delta_dens_depth_bg.pdf',dpi=300)

# depth_plot(mapr_delta_turb, 'Delta_Neph_outl(volts)', 'DEPTH', cutoff_depth)
# if print_fig == True:
#     plt.savefig(fig_path+'mapr_delta_turb_depth_bg.pdf',dpi=300)

depth_plot(mapr_delta_turb, 'Delta_Neph_smoo(volts)', 'DEPTH', cutoff_depth)
if print_fig == True:
    plt.savefig(fig_path+'mapr_delta_turbsmoo_depth_stamean.pdf',dpi=300)

#%% calculating heat flux
ref_station = profile_data[profile_data['Station']=='055_01']

zplume = {'theta_ctd':2900,'dORP_ctd':2800,'he':2700,'theta_mapr':2900,'dORP_mapr':2900,'turb_mapr':3000}
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

#%% delta anomaly 2D
min_dep = 2500
max_dep = 4600

plot_2Dpercast(profile_delta_potemperature_dens, 'Delta_potemperature', min_dep, max_dep, window_size=1000, vent_loc=vent_loc, bathy=bathy)
if print_fig == True:
    plt.savefig(fig_path+'profile_delta_potemperature_binned2D_dens.pdf',dpi=300)

plot_2Dpercast(profile_delta_potemperature_bg, 'Delta_potemperature', min_dep, max_dep, window_size=1000, vent_loc=vent_loc, bathy=bathy)
if print_fig == True:
    plt.savefig(fig_path+'profile_delta_potemperature_binned2D_bg.pdf',dpi=300)

plot_2Dpercast(profile_data, 'dORP', min_dep, max_dep, window_size=1000, vent_loc=vent_loc, bathy=bathy)
if print_fig == True:
    plt.savefig(fig_path+'profile_delta_dORP_binned2D.pdf',dpi=300)

plot_var_in_2D(btl_data, 'delta3He', min_dep, max_dep, 1, vent_loc=vent_loc,bathy=bathy)
if print_fig == True:
    plt.savefig(fig_path+'profile_he_2D.pdf',dpi=300)

#%% mapr8
plot_2Dpercast(mapr_delta_potemperature_dens, 'Delta_potemperature', min_dep, max_dep, window_size=400, vent_loc=vent_loc, bathy=bathy)
if print_fig == True:
    plt.savefig(fig_path+'mapr_delta_potemperature_binned2D_dens.pdf',dpi=300)

plot_2Dpercast(mapr_delta_potemperature_bg, 'Delta_potemperature', min_dep, max_dep, window_size=400, vent_loc=vent_loc, bathy=bathy)
if print_fig == True:
    plt.savefig(fig_path+'mapr_delta_potemperature_binned2D_bg.pdf',dpi=300)

plot_2Dpercast(mapr_data, 'dORP', min_dep, max_dep, window_size=400, vent_loc=vent_loc, bathy=bathy)
if print_fig == True:
    plt.savefig(fig_path+'mapr_delta_dORP_binned2D.pdf',dpi=300)

plot_2Dpercast(mapr_delta_turb, 'Delta_Neph_smoo(volts)', min_dep, max_dep, window_size=400, vent_loc=vent_loc, bathy=bathy)
if print_fig == True:
    plt.savefig(fig_path+'mapr_delta_turb_binned2D.pdf',dpi=300)


#%% calc distance

# T endmember:
#     St037 NUI44-IGT2 126 °C +-10°C, 651.89115%
#     St050 NUI45-IGT1 370 °C +-10°C, 685.52245%; 684.63171% double sample

potemperature_vent_bg = profile_data['potemperature'][profile_data['DEPTH'].between(3890,3910)].mean()


#first line for 3D distance
#second line for geodesic

dist_potemperature = profile_delta_potemperature_dens[profile_delta_potemperature_dens['DEPTH']>2900]
#dist_potemperature['Dist'] = dist_potemperature.apply(lambda x: dist(x['CTD_lon'], x['CTD_lat'], vent_loc[0], vent_loc[1], x['DEPTH'], vent_loc[2]), axis=1)
dist_potemperature['Dist'] = dist_potemperature.apply(lambda x: dist(x['CTD_lon'], x['CTD_lat'], vent_loc[0], vent_loc[1]), axis=1)


dist_dORP = profile_data[profile_data['DEPTH']>2900]
#dist_dORP['Dist'] = dist_dORP.apply(lambda x: dist(x['CTD_lon'], x['CTD_lat'], vent_loc[0], vent_loc[1], x['DEPTH'], vent_loc[2]), axis=1)
dist_dORP['Dist'] = dist_dORP.apply(lambda x: dist(x['CTD_lon'], x['CTD_lat'], vent_loc[0], vent_loc[1]), axis=1)


dist_he = btl_data[btl_data['DepSM_mean']>2900]
#dist_he['Dist'] = dist_he.apply(lambda x: dist(x['CTD_lon'], x['CTD_lat'], vent_loc[0], vent_loc[1], x['DepSM_mean'], vent_loc[2]), axis=1)
dist_he['Dist'] = dist_he.apply(lambda x: dist(x['CTD_lon'], x['CTD_lat'], vent_loc[0], vent_loc[1]), axis=1)

dist_potemperature_mapr = mapr_delta_potemperature_dens[mapr_delta_potemperature_dens['DEPTH']>2900]
#dist_potemperature_mapr['Dist'] = dist_potemperature_mapr.apply(lambda x: dist(x['CTD_lon'], x['CTD_lat'], vent_loc[0], vent_loc[1], x['DEPTH'], vent_loc[2]), axis=1)
dist_potemperature_mapr['Dist'] = dist_potemperature_mapr.apply(lambda x: dist(x['CTD_lon'], x['CTD_lat'], vent_loc[0], vent_loc[1]), axis=1)


dist_dORP_mapr = mapr_data[mapr_data['DEPTH']>2900]
#dist_dORP_mapr['Dist'] = dist_dORP_mapr.apply(lambda x: dist(x['CTD_lon'], x['CTD_lat'], vent_loc[0], vent_loc[1], x['DEPTH'], vent_loc[2]), axis=1)
dist_dORP_mapr['Dist'] = dist_dORP_mapr.apply(lambda x: dist(x['CTD_lon'], x['CTD_lat'], vent_loc[0], vent_loc[1]), axis=1)


dist_turb_mapr = mapr_delta_turb[mapr_delta_turb['DEPTH']>2900]
#dist_turb_mapr['Dist'] = dist_turb_mapr.apply(lambda x: dist(x['CTD_lon'], x['CTD_lat'], vent_loc[0], vent_loc[1], x['DEPTH'], vent_loc[2]), axis=1)
dist_turb_mapr['Dist'] = dist_turb_mapr.apply(lambda x: dist(x['CTD_lon'], x['CTD_lat'], vent_loc[0], vent_loc[1]), axis=1)


#%% plot profile distance
zoom = False

from scipy.signal import savgol_filter
import matplotlib.offsetbox as offsetbox
avg_no = 20

fig, ax = plt.subplots(figsize=(10,3))

ax.axhline(0,linestyle='--',color='black',label='Background level')

ax.scatter(dist_potemperature['Dist'],dist_potemperature['Delta_potemperature'], marker='.', s=5, color = get_var('Delta_potemperature')[1],label=get_var('Delta_potemperature')[0])
#ax.scatter(dist_potemperature['Dist'],savgol_filter(dist_potemperature['Delta_potemperature'],avg_no,3), marker='.', s=5, color = get_var('Delta_potemperature')[1],label=get_var('Delta_potemperature')[0])

#ax.axhline(0,linestyle='--', color = get_var('Delta_potemperature')[1])
ax.set_ylabel(get_var('Delta_potemperature')[0])
ax.set_ylim(dist_potemperature['Delta_potemperature'].min(),dist_potemperature['Delta_potemperature'].max()+0.0035)

twin1 = ax.twinx()

twin1.scatter(dist_dORP['Dist'],dist_dORP['dORP'], marker='.', s=5, color = get_var('dORP')[1],alpha=1,label=get_var('dORP')[0])
#twin1.scatter(dist_dORP['Dist'],savgol_filter(dist_dORP['dORP'],avg_no,3), marker='.', s=5, color = get_var('dORP')[1],alpha=1,label=get_var('dORP')[0])

#twin1.axhline(0,linestyle='--', color = get_var('dORP')[1])
twin1.set_ylabel(get_var('dORP')[0])
twin1.invert_yaxis()
twin1.set_ylim(dist_dORP['dORP'].max(),dist_dORP['dORP'].min()-0.07)

twin2 = ax.twinx()
twin2.scatter(dist_he['Dist'],dist_he['delta3He'], marker='o',edgecolors = 'black', color = get_var('delta3He')[1],label=get_var('delta3He')[0],s=60)
#twin2.axhline(7.146,linestyle='--', color = get_var('delta3He')[1])
twin2.set_ylabel(get_var('delta3He')[0])
twin2.spines.right.set_position(("axes", 1.11))
twin2.set_ylim(dist_he['delta3He'].min(),dist_he['delta3He'].max()+0.8)

ax.axvline(x=801,linestyle='dashed',color='black',alpha=0.5)

ax.locator_params(axis='x', nbins=20)
ax.set_xlabel('Vent Distance in m')

#use this for normal legend
#lgnd = fig.legend(bbox_to_anchor=(1,1), bbox_transform=ax.transAxes,ncol=2)

#use this for the additional legend from png file
# Load the PNG file
img = offsetbox.OffsetImage(plt.imread(fig_path+'profile_perdistance_legend.png'), zoom=0.5)
# Create and ad annotation box
ab = offsetbox.AnnotationBbox(img, (0.83, 0.89), xycoords='axes fraction',boxcoords='axes fraction',pad=0.0,frameon=False)
ax.add_artist(ab)

plt.tight_layout()
if print_fig == True:
    #plt.savefig(fig_path+'profile_perdistance.pdf')
    plt.savefig(fig_path+'profile_perdistance.png',dpi=300)

ax.set_xlim(0,800)

if print_fig == True:
    #plt.savefig(fig_path+'profile_perdistance.pdf')
    plt.savefig(fig_path+'profile_perdistance_zoom.png',dpi=300)

plt.show()


#%% plot mapr distance
fig, ax = plt.subplots(figsize=(10,3))

ax.axhline(0,linestyle='--',color='black',label='Background level')

ax.scatter(dist_potemperature_mapr['Dist'],dist_potemperature_mapr['Delta_potemperature'].rolling(1).mean(), marker='.', s=5, color = get_var('Delta_potemperature')[1],label=get_var('Delta_potemperature')[0])
#ax.axhline(0,linestyle='--', color = get_var('Delta_potemperature')[1])
ax.set_ylabel(get_var('Delta_potemperature')[0])
ax.set_ylim(dist_potemperature_mapr['Delta_potemperature'].min(),dist_potemperature_mapr['Delta_potemperature'].max()+0.0045)

twin1 = ax.twinx()
twin1.scatter(dist_dORP_mapr['Dist'],dist_dORP_mapr['dORP'].rolling(1).mean(), marker='.', s=5, color = get_var('dORP')[1],alpha=1,label=get_var('dORP')[0])
#twin1.axhline(0,linestyle='--', color = get_var('dORP')[1])
twin1.set_ylabel(get_var('dORP')[0])
twin1.invert_yaxis()
twin1.set_ylim(dist_dORP_mapr['dORP'].max()-0.028,dist_dORP_mapr['dORP'].min())

twin2 = ax.twinx()
twin2.scatter(dist_turb_mapr['Dist'],dist_turb_mapr['Delta_Neph_smoo(volts)'].rolling(1).mean(), marker='.', s=5, color = get_var('Delta_Neph_smoo(volts)')[1],alpha=0.2,label=get_var('Delta_Neph_smoo(volts)')[0])
#twin2.axhline(0,linestyle='--', color = get_var('Delta_Neph_smoo(volts)')[1])
twin2.set_ylabel(get_var('Delta_Neph_smoo(volts)')[0])
twin2.spines.right.set_position(("axes", 1.11))

ax.axvline(x=801,linestyle='dashed',color='black',alpha=0.5)

ax.locator_params(axis='x', nbins=20)
ax.set_xlabel('Vent Distance in m')

#use this for normal legend
#lgnd = fig.legend(bbox_to_anchor=(1,1), bbox_transform=ax.transAxes,ncol=2)

#use this for the additional legend from png file
# Load the PNG file
img = offsetbox.OffsetImage(plt.imread(fig_path+'mapr_perdistance_legend.png'), zoom=0.5)
# Create and ad annotation box
ab = offsetbox.AnnotationBbox(img, (0.81, 0.89), xycoords='axes fraction',boxcoords='axes fraction',pad=0.0,frameon=False)
ax.add_artist(ab)

plt.tight_layout()
if print_fig == True:
    #plt.savefig(fig_path+'profile_perdistance.pdf')
    plt.savefig(fig_path+'mapr_perdistance.png',dpi=300)

ax.set_xlim(0,800)

if print_fig == True:
    #plt.savefig(fig_path+'profile_perdistance.pdf')
    plt.savefig(fig_path+'mapr_perdistance_zoom.png',dpi=300)

plt.show()

#%% plot background station

station_plot(profile_background,btl_background,200)
if print_fig == True:
    plt.savefig(fig_path+'profile_background_depth.png',dpi=300)



