#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 10:33:00 2024

@author: jmette@uni-bremen.de
"""

import os
working_dir = '/home/jonathan/Dokumente/Masterarbeit/python/thesis/'
os.chdir(working_dir)

import xarray as xa
from lat_lon_parser import parse
import matplotlib.pyplot as plt
import cartopy.crs as ccrs


#%% print fig

print_fig = True

#%% plotting ridge crosssection

fig_path = '/home/jonathan/Dokumente/Masterarbeit/Thesis_Latex_Project/figures/2-background/'

fp = r'/home/jonathan/Dokumente/SHK Maren/PS137/Manuscript_Joel/GEBCO_28_Oct_2024_b50a4f017cb4/gebco_2024_n90.0_s70.0_w-30.0_e30.0.tif'
dem = xa.open_dataset(fp)

vent_loc = (parse("6° 15.32'W"),parse("82° 53.83'N"))

cross = dem.sel(y=vent_loc[1], method='nearest')
x = cross.x.to_numpy()
depth = cross.elevation.to_numpy()[0]

plt.figure(figsize=(8, 3))
plt.plot(x, depth,color='black')
plt.scatter(vent_loc[0],-3850,marker='*',s=100,color='red',label='Aurora Vent Site')
plt.xlabel("Longitude")
plt.ylabel("Depth in m")
plt.xlim((-10,-2))
plt.ylim((depth.min(),0))
plt.legend()
plt.tight_layout()
if print_fig == True:
    plt.savefig(fig_path+'ridge-crosssection.png',dpi=300)
plt.show()

#%%plotting ridge crosssection with current meters

fig_path = '/home/jonathan/Dokumente/Masterarbeit/Thesis_Latex_Project/figures/2-background/'

fp = r'/home/jonathan/Dokumente/SHK Maren/PS137/Manuscript_Joel/GEBCO_28_Oct_2024_b50a4f017cb4/gebco_2024_n90.0_s70.0_w-30.0_e30.0.tif'
dem = xa.open_dataset(fp)

vent_loc = (parse("6° 15.32'W"),parse("82° 53.83'N"))

cross = dem.sel(y=vent_loc[1], method='nearest')
x = cross.x.to_numpy()
depth = cross.elevation.to_numpy()[0]

plt.figure(figsize=(8, 3))
plt.plot(x, depth,color='black')
plt.scatter(vent_loc[0],-3850,marker='*',s=100,color='red',label='Aurora Vent Site')

#mooring
plt.hlines(y=-3064,xmin=-6.4,xmax=-6.1,label='512 at 3064m',colors='blue')
plt.hlines(y=-3215,xmin=-6.4,xmax=-6.1,label='506 at 3215 m',colors='orange')
#aquadopp
plt.hlines(y=-3220,xmin=-6.4,xmax=-6.1,label='Aquadopp at 3220 m',colors='magenta')
#mmoring
plt.hlines(y=-3366,xmin=-6.4,xmax=-6.1,label='569 at 3366 m',colors='olive')
plt.hlines(y=-3572,xmin=-6.4,xmax=-6.1,label='474 at 3572 m',colors='red')

plt.vlines(vent_loc[0],ymin=-3850,ymax=0,color='black',linestyle=':')

plt.xlabel("Longitude")
plt.ylabel("Depth in m")
plt.xlim((-10,-2))
plt.ylim((depth.min(),0))
plt.legend()
plt.tight_layout()
plt.show()
if print_fig == True:
    plt.savefig('/home/jonathan/Dokumente/Masterarbeit/Thesis_Latex_Project/figures/3-methods/'+'mooring_aqua_ridge-crosssection.pdf',dpi=200)

#%% plot large overview map

fig_path = '/home/jonathan/Dokumente/Masterarbeit/Thesis_Latex_Project/figures/3-methods/'
fp = r'/home/jonathan/Dokumente/SHK Maren/PS137/Manuscript_Joel/GEBCO_28_Oct_2024_b50a4f017cb4/gebco_2024_n90.0_s70.0_w-30.0_e30.0.tif'
coll_path = '/home/jonathan/Dokumente/Masterarbeit/CTD_data_collection/all_ctd_20240719.csv'

mooring_loc =  (parse("6° 15.043'W"),parse("82° 53.866'N"))
vent_loc = (parse("6° 15.32'W"),parse("82° 53.83'N"))

import matplotlib.pyplot as plt
import xarray as xa

dem = xa.open_dataset(fp)
dem = dem.elevation

dem_low = dem.coarsen(x=5).mean().coarsen(y=5).mean()

# coll = pd.read_csv(coll_path)
# coll = coll[coll['Event'].isin(['PS137_022_01', 'PS137_026_01', 'PS137_028_01', 'PS137_033_01','PS137_036_01' 'PS137_041_01', 'PS137_054_01', 'PS137_055_01'])]
# data_list = [d for _, d in coll.groupby(['Event'])]

plt.figure(figsize=(7,5))

#central_longitude=-6,false_easting=-30,cutoff=70,standard_parallels = (50, 50)
ax = plt.axes(projection=ccrs.LambertConformal(central_longitude=-6))
ax.coastlines()
ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)

bathy = dem_low.plot(ax=ax, transform=ccrs.PlateCarree(), cmap='viridis', vmax=0, add_colorbar=False)
contourlines = ax.contour(dem_low.x,dem_low.y,-dem_low.to_numpy().squeeze(),levels=20, colors='black',linestyles='solid',linewidths=0.5,alpha=0.3,transform=ccrs.PlateCarree())
cbar = plt.colorbar(bathy, ax=ax, fraction=0.036, pad=0.1)
# cbar = bathy.colorbar()
cbar.set_label('Depth in m')
# cbar.ax.set_anchor((1.05, 0.5))

ax.set_extent([15,-20,81,85]) #for thesis

# for station in data_list:
#     plt.scatter(station['Longitude'].iloc[0],station['Latitude'].iloc[0],transform = ccrs.PlateCarree(),marker='x',color='black')
plt.scatter(	-5.81366,82.6144,transform = ccrs.PlateCarree(),marker='*',color='black',s=200)  #marker for station 018
#plt.scatter(mooring_loc[0],mooring_loc[1],transform = ccrs.PlateCarree(),color='red',marker='*',s=100,label='Mooring')
plt.scatter(vent_loc[0],vent_loc[1],transform = ccrs.PlateCarree(),color='red',marker='*',s=300)

#plt.legend(['Station 018-01','Aurora Vent Site, Mooring'])

plt.title(None)
plt.tight_layout()
plt.show()
# if print_fig == True:
#     plt.savefig(fig_path+'map_overview.png', dpi=300)

#%% plot large overview map for model domain

fig_path = '/home/jonathan/Dokumente/Masterarbeit/Thesis_Latex_Project/figures/3-methods/'
fp = r'/home/jonathan/Dokumente/SHK Maren/PS137/Manuscript_Joel/GEBCO_28_Oct_2024_b50a4f017cb4/gebco_2024_n90.0_s70.0_w-30.0_e30.0.tif'
coll_path = '/home/jonathan/Dokumente/Masterarbeit/CTD_data_collection/all_ctd_20240719.csv'

mooring_loc =  (parse("6° 15.043'W"),parse("82° 53.866'N"))
vent_loc = (parse("6° 15.32'W"),parse("82° 53.83'N"))

import matplotlib.pyplot as plt
import xarray as xa
from matplotlib.patches import Rectangle

dem = xa.open_dataset(fp)
dem = dem.elevation

dem_low = dem.coarsen(x=5).mean().coarsen(y=5).mean()

plt.figure(figsize=(7,5))

#central_longitude=-6,false_easting=-30,cutoff=70,standard_parallels = (50, 50)
ax = plt.axes(projection=ccrs.LambertConformal(central_longitude=-6))
ax.coastlines()
ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)

bathy = dem_low.plot(ax=ax, transform=ccrs.PlateCarree(), cmap='viridis', vmax=0, add_colorbar=False)
contourlines = ax.contour(dem_low.x,dem_low.y,-dem_low.to_numpy().squeeze(),levels=20, colors='black',linestyles='solid',linewidths=0.5,alpha=0.3,transform=ccrs.PlateCarree())
cbar = plt.colorbar(bathy, ax=ax, fraction=0.036, pad=0.1)
# cbar = bathy.colorbar()
cbar.set_label('Depth in m')
# cbar.ax.set_anchor((1.05, 0.5))

ax.set_extent([15,-20,81,85]) #for thesis

#plot model domain
lon_min = -7.0584416461002135
lon_max = -5.511549697642964
lat_min = 82.80289162465036
lat_max = 82.98648016125792
ax.add_patch(Rectangle((lon_max,lat_min),(lon_min-lon_max),(lat_max-lat_min),facecolor = 'red',transform=ccrs.PlateCarree()))

plt.title(None)
plt.tight_layout()
plt.show()
# if print_fig == True:
#     plt.savefig(fig_path+'map_model_domain.png', dpi=300)

#%% plot large overview map for model domain discussion

fig_path = '/home/jonathan/Dokumente/Masterarbeit/Thesis_Latex_Project/figures/3-methods/'
fp = r'/home/jonathan/Dokumente/SHK Maren/PS137/Manuscript_Joel/GEBCO_28_Oct_2024_b50a4f017cb4/gebco_2024_n90.0_s70.0_w-30.0_e30.0.tif'
coll_path = '/home/jonathan/Dokumente/Masterarbeit/CTD_data_collection/all_ctd_20240719.csv'

mooring_loc =  (parse("6° 15.043'W"),parse("82° 53.866'N"))
vent_loc = (parse("6° 15.32'W"),parse("82° 53.83'N"))

import matplotlib.pyplot as plt
import xarray as xa
from matplotlib.patches import Rectangle

dem = xa.open_dataset(fp)
dem = dem.elevation

dem_low = dem.coarsen(x=5).mean().coarsen(y=5).mean()

plt.figure(figsize=(7,5))


#central_longitude=-6,false_easting=-30,cutoff=70,standard_parallels = (50, 50)
ax = plt.axes(projection=ccrs.LambertConformal(central_longitude=-6))
ax.coastlines()
#ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)

bathy = dem_low.plot(ax=ax, transform=ccrs.PlateCarree(), cmap='viridis', vmax=0, add_colorbar=False)
contourlines = ax.contour(dem_low.x,dem_low.y,-dem_low.to_numpy().squeeze(),levels=40, colors='black',linestyles='solid',linewidths=0.5,alpha=0.3,transform=ccrs.PlateCarree())
cbar = plt.colorbar(bathy, ax=ax, fraction=0.036, pad=0.1)
# cbar = bathy.colorbar()
cbar.set_label('Depth in m')
# cbar.ax.set_anchor((1.05, 0.5))

#plot model domain
lon_min = -7.0584416461002135
lon_max = -5.511549697642964
lat_min = 82.80289162465036
lat_max = 82.98648016125792
#ax.add_patch(Rectangle((lon_max,lat_min),(lon_min-lon_max),(lat_max-lat_min),facecolor = 'red',transform=ccrs.PlateCarree()))

ax.set_extent([lon_max,lon_min,lat_min,lat_max])

plt.title(None)
plt.tight_layout()
plt.show()
# if print_fig == True:
#     plt.savefig(fig_path+'map_model_domain.png', dpi=300)
#%% plot all Polarstern bathymetry

gebcopath = '/home/jonathan/Dokumente/SHK Maren/PS137/PS137_AuroraVent_25m_bilinear_WGS84.nc'     # path for bathymetry files
vent_loc = (parse("6° 15.32'W"),parse("82° 53.83'N"))
mooring_loc =  (parse("6° 15.043'W"),parse("82° 53.866'N"))


plt.figure(figsize=(8,6))


bathy = xa.open_dataset(gebcopath)

ax = plt.axes(projection=ccrs.LambertConformal(central_longitude=-6))
ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)

bathyplot = bathy.Band1.plot(ax=ax, transform=ccrs.PlateCarree(), cmap='viridis', add_colorbar=False)

contourlines = ax.contour(bathy.lon,bathy.lat,-bathy.Band1,levels=40, colors='black',linestyles='solid',linewidths=0.5,alpha=0.3,transform=ccrs.PlateCarree())
ax.clabel(contourlines, inline=True, fontsize=6, fmt='%d', colors = 'black')

plt.scatter(vent_loc[0],vent_loc[1],transform = ccrs.PlateCarree(),color='red',marker='*',s=100)
#plt.scatter(mooring_loc[0],mooring_loc[1],transform = ccrs.PlateCarree(),color='red',marker='*',s=100)

ax.set_extent([-5,-8,82.7,83.1]) #for thesis
#ax.set_extent([-5.5,-7.1,82.82,82.98]) #for comparing with roms grid


cbar = plt.colorbar(bathyplot, ax=ax, fraction=0.036, pad=0.1)
# cbar = bathy.colorbar()
cbar.set_label('Depth in m')
plt.tight_layout()
plt.show()

