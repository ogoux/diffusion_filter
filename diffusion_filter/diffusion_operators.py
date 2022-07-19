#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script contains a diffusion-based filter for geophysical data , described in:
    
"Mass conserving filter based on diffusion for Gravity Recovery and Climate 
Experiment (GRACE) spherical harmonics solutions, O. Goux, J. Pfeffer, A. Blasquez,
 A. T. Weaver, M. Ablain, 2021".
 
Its square root operator and its inverse operator are also provided for convenience,
but their use goes beyond the scope of the article.

@author: Olivier Goux, 2021
"""
import numpy as np
import diffusion_filter.diffusion_auxiliary as aux
import scipy.sparse.linalg as spl


    
def diffusion_filter(input_grid, D, M, water_ratio = None, boundary_mask = False,
                       out_of_domain_mask = False, surface_model = 'ellipsoid',
                       boundary_inflation_factor = 1, boundary_inflation_range = 1,
                       latitude_variability = (0,0)):
    """
    Diffusion based low pass filter for any geophysical quantity on the sphere
    or the ellipsoid.
     

    Parameters
    ----------
    input_grid : Masked array
        Gridded values at a regular geodetic latitude and longitude coordinates
        If a mask or the water ratio is used, their latitutde and longitude coordinates
        should be consistent with those of the input grid. If input_grid is two 
        dimensional, its dimensions are assumed to be (nlat,nlon). N arrays of dimensions
        (nlat,nlon) can be stacked in an array (N,nlat,nlon) to vectorize computations.
    D : float or array or tuple 
        The full diffusivity should be a tuple of two arrays: a meridional diffusivity
        field on parallels arcs of shape (nlat+1,nlon), and a zonal diffusivity field
        on meridian arcs of shpae (nlon,nlat+1). Simpler formats are also accepted and 
        expanded into this format with the following assumptions:
            - single scalar: 
                the Daley length scale is assummed to be isotropic and homogeneous.
            - tuple of two scalars: 
                the Daley length scale is assumed to be homogeneous. The
                first element refers to the meridional direction, the second to the zonal direction.
            - tuple of four scalars: 
                the Daley length scale is assumed to be homogeneous 
                in and out of boundary mask respectively. The first element refers to the meridional
                direction out of boundary_mask, the second to the parallel direction out of
                boundary_mask, the third to the meridional direction in boundary_mask, the second
                to the parallel direction in boundary_mask.
    M : int
        Number of diffusion iterations and equivalently order of the AR function corresponding
        to the smoothing kernel.
    water_ratio : array, optional
        Array of dimension (nlat,nlon) of the ocean surface ratio of eacdh grid cell.
    boundary_mask : 2D array, optional
        Array of booleans of dimension (nlat,nlon) separating the grid in two domains.
        Neumann conditions are used to restrict fluxes between the two domains 
        (typically if boundary mask is a land mask, there will be no flux between 
         land and ocean). The lat/lon ordering should be consistent with input_grid.
    out_of_domain_mask : 2D array, optional
         Array of booleans of dimension (nlat,nlon) separating the grid in two domains.
         All masked data will be excluded and cannot interact with the non masked data.
         The lat/lon ordering should be consistent with input_grid.
    surface_model : str, optional
        Model used for the shape of the Earth. If equal to 'sphere', a sphere 
        of radius a models the Earth. If equal to 'ellipsoid', a revolution 
        ellipsoid of equatorial radius a and flattening f is used. The default 
        is 'ellipsoid'.
    boundary_inflation_factor : float, optional
         An empirical correction can be applied by increasing the Daley length scale
         near the boundaries (see alpha in Equation 14 of Goux et al.). This parameter
         characterizes the amplitude of this correction. The default is 1.
    boundary_inflation_range : float, optional
         An empirical correction can be applied by increasing the Daley length scale
         near the boundaries (see beta in Equation 14 of Goux et al.). This parameter
         characterizes the width of this correction. The default is 3.
    latitude_variability: tuple, optional
        This option increases the Daley length scales at the Equator while keeping 
        it unchanged at the pole (the correction acts as a cosine of laitutde). If
        set to one, the length scales are doubled at the Equator; if set to zero nothing
        is done. A tuple of two scalars is expected, the first affect the meridional
        lengtth scales, the second the zonal length scales. The default is (0, 0).
    Returns
    -------
    output : array
        Filtered version of input_grid at the same format.

    """

    """
    input_grid should be of dimension [nlat, nlon] (or [N, nlat, nlon] to vectorize
    the computations on multiple grids). Each 2D grid is processed as a vector later on,
    which means it has to be converted in a format [nlat x nlon] (or [N, nlat x nlon]).
    If it is a Masked Array its mask is also extracted and added to the 'out_of_domain'
    mask, so that invalid values do not contaminate their neighbors. If no boundary mask
    is given and a path to the water_ratio is given, the edge between full land cells and
    cells with some water content is used as a boundary.
    """
    # =============================================================================
    input_grid = input_grid.copy()
    
    if len(input_grid.shape) == 2: 
        nlat, nlon = np.shape(input_grid)
        field = input_grid.flatten()       
        no_data_mask = input_grid.mask  if type(input_grid) == np.ma.core.MaskedArray else False
    else:
        N, nlat, nlon = np.shape(input_grid)
        field= input_grid.reshape(N, -1)
        field= field.swapaxes(0,1)
        no_data_mask = np.any(input_grid.mask, axis = 0)  if type(input_grid) == np.ma.core.MaskedArray else False
    res = (round(180/nlat, 2), round(360/nlon, 2))
   
    
    # Masks
    out_of_domain_mask = no_data_mask |  out_of_domain_mask 
    
    if (water_ratio is not None) and np.any(boundary_mask)==False:
        boundary_mask =  water_ratio <=0
    # =============================================================================
    
    """
    Initialize all quantities needed to solve (WL^{-1}) * field = W * field and
    solve it M times. 
    """
    # =========================================================================
    # Fetch diffusivity field    
    D = aux.format_D(D, res, boundary_mask = boundary_mask,
                     out_of_domain_mask = out_of_domain_mask, 
                     boundary_inflation_factor = boundary_inflation_factor,
                     boundary_inflation_range = boundary_inflation_range,
                     surface_model = surface_model, 
                     latitude_variability = latitude_variability)
   
    # Convert into diffusivity
    kappa = (D[0]**2 / (2*M-4), D[1]**2 / (2*M-4))


    # Initialize finite differences matrix
    WL_inv = aux.FD_matrix(res, kappa, dt = 1, water_ratio = water_ratio,
                       surface_model = surface_model)
    
    # Pre-compute the LU factorization in a sparse format and return a solver
    WL_operator  = spl.factorized(WL_inv)
    
    # Fetch the diagonal elements of W
    _, _, _, _, _, _, w = aux.grid_geometry(res, surface_model = surface_model,
                                        water_ratio = water_ratio)  
    wf = w.flatten()
    for i in range(M):
        # Apply W
        if len(np.shape(field)) == 1:
            field *= wf
        if len(np.shape(field)) == 2:
            field *= wf[:,np.newaxis]
        # Apply (W * L^{-1})^{-1}
        field = WL_operator(field)
        
    
    # =========================================================================
    
    
    """
    The original shape of the array is restored and points marked as out_of_domain are masked
    """
    # =========================================================================
    if len(input_grid.shape) == 2: 
        output = np.ma.array(field.reshape((nlat, nlon)))
        output.mask = out_of_domain_mask
    else:
        field = field.swapaxes(0,1)
        output = np.ma.array(field.reshape((N, nlat, nlon)))
        output.mask = np.tile(out_of_domain_mask, (N,1,1))
    # =========================================================================

    return output
