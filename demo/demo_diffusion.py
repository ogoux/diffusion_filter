#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example of application of a diffusion filter to a gridded EWH anomaly (single month)

@author: goux
"""



import numpy as np
import netCDF4 as nc
import diffusion_filter.diffusion_operators as dif

# Packages required only for plot_anomaly
import matplotlib.pyplot as plt
import matplotlib.colors as col
import mpl_toolkits.basemap as mpb


# Function to display a single monthly anomaly using matplotlib and basemap
def plot_anomaly(grid, title):
    plt.figure(figsize=(7,5), dpi= 100, facecolor='w', edgecolor='k')
    #Create a Basemap object
    m = mpb.Basemap(projection='robin', llcrnrlat=-90, urcrnrlat=90,\
                llcrnrlon=-180, urcrnrlon=180, resolution='c', lon_0=0)
            
    # Transforms lat/lon into plotting coordinates for projection
    lon2D,lat2D=np.meshgrid(np.arange(-179.5,180), np.arange(-89.5,90))
    plon,plat = m(lon2D[:,:],lat2D[:,:]) 
        
    #Add coastlines and boundary
    m.drawcoastlines(linewidth=0.5)
    m.drawmapboundary()
    vmin, vmax = np.percentile( grid.compressed(), [0.01, 100 - 0.01])
    vabs = max([-vmin, vmax])
    cs = m.pcolor(plon, plat, np.roll(grid, grid.shape[1]//2, axis =1), cmap = plt.cm.bwr,
                  shading='auto',  norm = col.SymLogNorm(linthresh = 0.1, 
                                                vmin = -vabs, vmax = vabs))
    plt.colorbar(cs, orientation = 'horizontal')
    plt.suptitle(title)


# Import the EWH anomaly and water ratio from the netCDF file
with nc.Dataset("test_data.nc", 'r') as data:
    ewh = data['ewh'][:]
    water_ratio = data['water_ratio'][:]

# Plot the unfiltered anomaly
plot_anomaly(ewh, "Unfiltered EWH anomaly, January 2014")


# Create a mask from the water ratio and show it
mask = water_ratio<=0.9
#mask = water_ratio<=0.02
plt.pcolor(np.roll(mask, mask.shape[1]//2, axis =1))

# Length scales of the filter. Other formats for D are specified in the description
# of the function diffusion__filter
D = (360E3, # Along North-South direction on the ocean
     540E3, # Along East-West direction on the ocean
     150E3, # Along North-South direction on land
     330E3) # Along North-South direction on land
M = 4

# Filter the solution
filtered_ewh = dif.diffusion_filter(ewh, D, M, water_ratio = water_ratio, boundary_mask = mask)

# If the mask is provided as "out_of_domain_mask" instead of "boundary_mask",
# all the points flagged in it are excluded. 
#filtered_ewh = dif.diffusion_filter(ewh, D, M, water_ratio = water_ratio, out_of_domain_mask = mask)

plot_anomaly(filtered_ewh, "Filtered EWH anomaly, January 2014")
