#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script contains auxiliary functions for the diffusive filter

@author: Olivier Goux, 2021
"""

# Usual packages
import numpy as np
import scipy.sparse as sp


# 2D field coordinates to vector representation coordinates
def ij_to_k(i, j, nlon):
    return i * nlon + j


#  Vector representation coordinates to 2D field coordinates
def k_to_ij(k, nlon):
    return k // nlon, k % nlon


def lat(res, centered = True):
    """
    Generate an array of latitude covering the range -90:90

    Parameters
    ----------
    res : float
        Sampling in latitude
    centered : bool, optional
        If centered is True, the latitude vector is of dimension 180//res and 
        corresponds to the centers of grid cells. If centered is False, the latitude
        vector is of dimension 1+180//res and corresponds to the parallel arcs 
        separating grid cells in the North_South direction. The default is True.

    Returns
    -------
    array
        1D array of latitudes in degrees.

    """
    if centered:
        return np.arange(- 90 + res/2, 90, res )
    else:
        return np.arange(- 90, 90 + res, res )


def lon(res, centered = True):
    """
    Generate an array of longitude covering the range 0:360

    Parameters
    ----------
    res : float
        Sampling in longitude
    centered : bool, optional
        If centered is True, the longitude vector is of dimension 360//res and 
        corresponds to the centers of grid cells. If centered is False, the longitude
        vector is of dimension 1+360//res and corresponds to the meridian arcs 
        separating grid cells in the East-West direction. The default is True.

    Returns
    -------
    array
        1D array of latitudes in degrees.

    """
    if centered:
        return np.arange(res/2, 360, res)
    else:
        return np.arange(0, 360 + res, res )


def distance_to_mask(mask):
    """
    Returns maps of the distance in pixels between each edge of the grid to the
    edge separating the two domains delimited by a mask.

    Parameters
    ----------
    mask : Array
        Array of dimensions (nlat,nlon) of booleans separating a grid in two domains.

    Returns
    -------
    dist_i : Array
        Array of meridional distances in number of pixels to the closest meridional edge
        separating the two domains of the mask.
    dist_j : TYPE
        Array of zonal distances in number of pixels to the closest zonal edge
        separating the two domains of the mask.

    """
    
    
    """
    While the mask is provided on the centers of grid cells, we need to return 
    quantities relevant to the edges of the grid. Masks of the edge between the
    two domains are defined for both directions
    """
    # =========================================================================
    nlat, nlon = np.shape(mask)
    
   
    dist_i = np.zeros((nlat+1, nlon))
    dist_j = np.zeros((nlat, nlon+1))
    
    mask_down = np.concatenate((~mask[:1, :], mask), axis = 0 )
    mask_up = np.concatenate((mask, ~mask[-1:, :]), axis = 0 )
    mask_i = mask_down ^ mask_up
    
    mid_lon = int((nlon+1)//2)
    mask_per = np.concatenate((mask[:,mid_lon:], mask, mask[:,:mid_lon] ), axis = 1)
    mask_left = np.concatenate((mask_per[:,-1:], mask_per), axis = 1 )
    mask_right = np.concatenate((mask_per, mask_per[:,:1]), axis = 1 )
    mask_j = mask_left ^ mask_right  
    # =========================================================================

    """
    The distance to the closest edge in mask_i/j is computed. Note that there is
    this computation is approximative. The smallest distance in terms of number 
    of pixels (computed here) is not necessarily the smallest distance in km, as 
    the size of grid cells is variable.
    """
    # =========================================================================
    # Indices of parallel arcs
    i_col = np.arange(0,nlat+1)
    # Indices of meridian arcs. The offest account for the periodic correction
    j_line = np.arange(0,nlon+1) + mid_lon
    
    # Loop on columns for dist_j
    for i in range(nlat):
        mask_inds = np.where(mask_j[i,:])[0]
        if np.size(mask_inds) == 0:
             dist_j[i,:] = 2*nlon
        else :
            dist_j_line = np.min(np.abs(j_line[:,np.newaxis] - mask_inds[np.newaxis, :]), axis = 1)
            dist_j[i,:] = dist_j_line
    
    # Loop on lines for dist_i
    for j in range(nlon):
        mask_inds = np.where(mask_i[:,j])[0]
        if np.size(mask_inds) == 0:
             dist_i[:,j] = 2*nlat
        else:
            dist_i_line = np.min(np.abs(i_col[:, np.newaxis] - mask_inds[np.newaxis,:]), axis = 1)
            dist_i[:, j] = dist_i_line
    # =========================================================================
    
    
    return dist_i, dist_j



def format_D(D, res, boundary_mask = False, out_of_domain_mask = False, surface_model = 'ellipsoid',
             boundary_inflation_factor = 1, boundary_inflation_range = 3 ):
    """
    If D is not inside a tuple of arrays representing the meridional and zonal 
    diffusivity fields, it is expanded to fit this format. boundary mask is used 
    to define boundaries between masked and non masked data (typically coastlines).
    out_of_domain_mask is used to exclude masked data. A local inflation of the 
    length scale based is applied based on boundary_inflation_factor and 
    boundary_inflation_range (resp. alpha and beta in Equation 14).
          
    Parameters
    ----------
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
        -----------------------------------------------------------------------
    """
    
    """
    The inputs masks are defined at the centers of grid cells, however the 
    Daley length scales are defined on edges between cells: D_i on parallel arcs
    separating cells in the meridional direction, D_j on meridian arcs separating
    cells in the zonal direction. We need to define masks which are relevant on
    these edges.
    """
    # =========================================================================
    if np.any(boundary_mask):
        # Enforce boolean type
        boundary_mask = boundary_mask.astype(bool)    
        # decentered masks 
        mask_left = np.concatenate((boundary_mask[:,-1:], boundary_mask), axis = 1 )
        mask_right = np.concatenate((boundary_mask, boundary_mask[:,:1]), axis = 1 )
        mask_down = np.concatenate((boundary_mask[:1, :], boundary_mask), axis = 0 )
        mask_up = np.concatenate((boundary_mask, boundary_mask[-1:, :]), axis = 0 )
        # boundary -> True between two cells of different values in boundary mask
        # inside -> True at boundary + between two cells equal to 1
        # outside -> True at boundary + between two cells equal to 0
        # _i -> defined on parallel arcs ie relevant for meridional direction
        # _j -> defined on meridian arcs ie relevant for zonal direction
        boundary_i = mask_down ^ mask_up
        boundary_j = mask_left ^ mask_right
        inside_boundary_i = mask_down | mask_up
        inside_boundary_j = mask_left | mask_right
    else:
        boundary_i = False
        boundary_j = False
        inside_boundary_i = False
        inside_boundary_j = False
        
    if np.any(out_of_domain_mask):
         # Enforce boolean type
        out_of_domain_mask = out_of_domain_mask.astype(bool)    
        # decentered masks 
        mask_left = np.concatenate((out_of_domain_mask[:,-1:], out_of_domain_mask), axis = 1 )
        mask_right = np.concatenate((out_of_domain_mask, out_of_domain_mask[:,:1]), axis = 1 )
        mask_down = np.concatenate((out_of_domain_mask[:1, :], out_of_domain_mask), axis = 0 )
        mask_up = np.concatenate((out_of_domain_mask, out_of_domain_mask[-1:, :]), axis = 0 )
        # out_of_domain -> True between two cells of different values in out_of_domain mask
        # inside -> True at out_of_domain + between two cells equal to 1
        # outside -> True at out_of_domain + between two cells equal to 0
        # _i -> defined on parallel arcs ie relevant for meridional direction
        # _j -> defined on meridian arcs ie relevant for zonal direction
        out_of_domain_i = mask_down | mask_up
        out_of_domain_j = mask_left | mask_right
    else:
        out_of_domain_i = False
        out_of_domain_j = False
    # =========================================================================
 
    
    """
    The full diffusivity should be a tuple of two arrays: a meridional diffusivity
    field on parallels arcs of shape (nlat+1,nlon), and a zonal diffusivity field
    on meridian arcs of shpae (nlon,nlat+1). Simpler formats are also accepted and 
    expanded into this format with the following assumptions:
        - single scalar: the Daley length scale is assummed to be isotropic and homogeneous
        - tuple of two scalars: the Daley length scale is assumed to be homogeneous. The
            first element refers to the meridional direction, the second to the zonal direction.
        - tuple of four scalars: the Daley length scale is assumed to be homogeneous 
            in and out of boundary mask respectively. The first element refers to the meridional
            direction out of boundary_mask, the second to the parallel direction out of
            boundary_mask, the third to the meridional direction in boundary_mask, the second
            to the parallel direction in boundary_mask.
    """
    # =========================================================================
    nlat = int(180 // res[0])
    nlon = int(360 // res[1])
    shape_i = (nlat + 1, nlon)
    shape_j = (nlat, nlon+1)
    
    if np.isscalar(D):
        D_i = D * np.ones(shape_i)
        D_j = D * np.ones(shape_j)
    else:
        assert type(D) == tuple
        if np.isscalar(D[0]):
            D_i = D[0] * np.ones(shape_i)
        else:
            assert np.shape(D[0]) == shape_i
            D_i = D[0]
        if np.isscalar(D[1]):
            D_j = D[1] * np.ones(shape_j)
        else:
            assert np.shape(D[1]) == shape_j
            D_j = D[1]
        if len(D) == 4 :
            if np.isscalar(D[2]):
                D_i[inside_boundary_i] = (D[2] * np.ones(shape_i))[inside_boundary_i]
            else:
                assert np.shape(D[2]) == shape_i
                D_i[inside_boundary_i] = D[2][inside_boundary_i]
            if np.isscalar(D[3]):
                D_j[inside_boundary_j] = (D[3] * np.ones(shape_j))[inside_boundary_j]
            else:
                assert np.shape(D[3]) == shape_j
                D_j[inside_boundary_j] = D[2][inside_boundary_j]
    # =========================================================================
    
    
    """
    Convolution kernels close to the boundaries are 'compacted' against the 
    boundary, which can result in less apparent filtering. An empirical compensation
    can be applied by increasing the Daley length scale near the boundaries. The
    offset starts at 0 at a distance 'boundary_inflation_range * D' of the boundary and increases
    linearly up to 'boundary_inflation_factor * D'
    """
    # =========================================================================
    combined_mask = boundary_mask | out_of_domain_mask
    if np.any(combined_mask):
        # Distance in pixels to the nearest boundary on the meridional and zonal directions resp.
        pix_dist_i, pix_dist_j = distance_to_mask(combined_mask)
        # The distance in meters to the boundary is approximated by the distance in pixels
        # time the metric coefficient at the given pixel (instead of an 'integration' of the metric
        # coefficient field from the pixel to the coast)
        _, _, ei_parallel, _, _, ej_meridian, _  = \
            grid_geometry(res, surface_model = surface_model)
        dist_i = pix_dist_i * ei_parallel
        dist_j = pix_dist_j * ej_meridian
        
        # Avoid division by zero in case of null length scale
        pos_inds_i = np.where(D_i>0)
        pos_inds_j = np.where(D_j>0)
        D_i[pos_inds_i] += boundary_inflation_factor * D_i[pos_inds_i] * \
                            np.exp( -0.5* (dist_i[pos_inds_i]/(boundary_inflation_range *  D_i[pos_inds_i]))**2  )
            
        D_j[pos_inds_j] += boundary_inflation_factor * D_j[pos_inds_j] * \
                            np.exp(-0.5*(dist_j[pos_inds_j]/(boundary_inflation_range *  D_j[pos_inds_j]))**2)       
        
    # =========================================================================
        
        
    """
    When the diffusivity is set to 0 on an edge, there will be no flux through 
    this edge. The diffusivty on the edges between the two domains of boundary_mask 
    is set to 0. This stops any flux between the two domains. The diffusivity of
    all the edges within or on the side of the domain where out_of_domain is True 
    are set to 0. There will not be any fluw (and thus filtering) in this domain.
    """
    # =========================================================================
    D_i [boundary_i | out_of_domain_i] = 0
    D_j [boundary_j | out_of_domain_j] = 0
    # =========================================================================
    
    """
    The boundary conditions on the side of the domain (periodicity on the zonal sides,
    polar on the meridional sides) are mostly enforced in the construction of the 
    finite differences operator, but the diffusivity should be consistent with them.
    """
    # =============================================================================
    # Zonal periodicity
    if np.any( D_j[:,-1] != D_j[:,0] ):
        print("Periodicity on zonal diffusivity had to be enforced")
        D_j_border = np.mean(  np.concatenate((D_j[:,-1:], D_j[:,:1]), axis = 1) , axis = 1  )
        D_j[:, -1] = D_j_border
        D_j[:, 0] = D_j_border
    # North pole
    if np.any( np.diff(D_i[0,:]) != 0 ):
        print("Consistency of meridional diffusivity at the South pole had to be enforced")
        D_i[0,:] = np.mean(D_i[0,:])
    # South pole
    if np.any( np.diff(D_i[-1,:]) != 0 ):
        print("Consistency of meridional diffusivity at the North pole had to be enforced")
        D_i[-1,:] = np.mean(D_i[-1,:])     
    # =============================================================================

    return (D_i, D_j)



def FD_matrix(res, kappa, dt, water_ratio = None, surface_model = 'ellipsoid'):
    """
    Given a full diffusivity field, the resolution of the grid, and a time step,
    this function generates the inverse matrix corresponding to one time step of 
    the implicit Euler scheme for the resolution of the anisotropic inhomogeneous
    diffusion equation on the sphere:
        
                FD_weights (...) = W * L^(-1) with L such that 
          L * (field at time tk-1) = field at time tk
    
    The product with W is modeled directly to get a symmetric matrix. Periodic
    boundary conditions are implemented at the zonal boundaries and 
    mass-conserving 'polar' conditions are used at the meridional boundaries. 
    Neumann boundaries for land, coasts, etc have to be implemented beforehand
    through the diffusivity field.

    Parameters
    ----------
    res : tuple
        Tuple of two elements : the resolutions in latitude and longituide of the grid in degrees.
    kappa : tuple of arrays
        The first element of the tuple is the meridional diffusivity field. Its
        elements are defined at South-North interfaces between grid cells, which
        gives the array a dimension (nlat +1, nlon). The second element of the 
        tuple is the zonal diffusivity field. Its elements are defined at 
        West-Easy interfaces between grid cells, which gives the array a 
        dimension (nlat +1, nlon). 
    dt : float
        Length of the time step over which the solution is integrated.
    water_ratio : array, optional
        Array of dimension (nlat,nlon) of the ocean surface ratio of eacdh grid cell.
    surface_model : str, optional
        Model used for the shape of the Earth. If equal to 'sphere', a sphere 
        of radius a models the Earth. If equal to 'ellipsoid', a revolution 
        ellipsoid of equatorial radius a and flattening f is used. The default 
        is 'ellipsoid'.
    a : float, optional
        Mean equatorial radius of the earth. The default is 6378136.6.

    Returns
    -------
    WL_inv : Sparse matrix 
        Sparse representation (CSC) of W * L^(-1).s

    """    
    nlat = int(180 // res[0])
    nlon = int(360 // res[1])
    N = nlat * nlon

    # Fetch metric coefficients -----------------------------------------------
    ei_center, ej_center, ei_parallel, ej_parallel ,ei_meridian, ej_meridian, w =\
        grid_geometry(res, surface_model = surface_model, water_ratio = water_ratio)
    
    # Generate elements 
    kappa_i, kappa_j = kappa
        
    alpha_i_plus = dt * kappa_i[1:, :] * ej_parallel[1:, :] / ei_parallel[1:, :]
    alpha_i_minus = dt * kappa_i[:-1, :] * ej_parallel[:-1, :] / ei_parallel[:-1, :]
    
    alpha_j_plus = dt * kappa_j[:, 1:] * ei_meridian[:, 1:] / ej_meridian[:, 1:]
    alpha_j_minus = dt * kappa_j[:, :-1] * ei_meridian[:, :-1] / ej_meridian[:,:-1]
    
    # 2D field are processed as one dimensional vectors. The matrix Delta should
    # be built accordingly    

    eq_diag = (w +(alpha_i_plus + alpha_i_minus + alpha_j_plus + alpha_j_minus) ).flatten()
    j_minus_diag = ( -alpha_j_minus).flatten()
    j_plus_diag = ( -alpha_j_plus).flatten()
    i_minus_diag = ( -alpha_i_minus ).flatten()
    i_plus_diag = ( -alpha_i_plus ).flatten()


    """
    The elements displaced from the main diagonals by boundary conditions are
    handled first, we need  to define lists of values, row indices and column indices
    for the storage in COO format.
    """
    # =========================================================================
    # Zonal periodic boundary conditions
    j_plus_data = [j_plus_diag[ij_to_k(i, nlon-1, nlon)] for i in range(nlat)]
    j_plus_row =  [ij_to_k(i, nlon-1, nlon) for i in range(nlat)]
    j_plus_col =  [ij_to_k(i, 0, nlon) for i in range(nlat)]
    
    j_minus_data = [j_minus_diag[ij_to_k(i, 0, nlon)] for i in range(nlat)]
    j_minus_row =  [ij_to_k(i, 0, nlon) for i in range(nlat)]
    j_minus_col =  [ij_to_k(i, nlon-1, nlon) for i in range(nlat)]

    # Meriodional 'polar' conditions    
    i_plus_data = [a/nlon for j in range(nlon) for a in [i_plus_diag[ij_to_k(nlat-1, j, nlon)]] * nlon]
    i_plus_row = [k for j in range(nlon) for k in [ij_to_k(nlat-1, j, nlon)] * nlon]
    i_plus_col = [N - j -1 for j in range(nlon)] * nlon
   
    i_minus_data = [a/nlon for j in range(nlon) for a in [i_minus_diag[ij_to_k(0, j, nlon)]] * nlon]
    i_minus_row = [k for j in range(nlon) for k in [ij_to_k(0, j, nlon)] * nlon]
    i_minus_col = [j for j in range(nlon)] * nlon
    
    BC_data = np.array(j_plus_data + j_minus_data + i_plus_data + i_minus_data)
    BC_row =  np.array(j_plus_row  + j_minus_row  + i_plus_row  + i_minus_row )
    BC_col =  np.array(j_plus_col  + j_minus_col  +  i_plus_col  + i_minus_col )
    # =========================================================================
    
    """
    The elements remaining remaining on the 5 diagonals are selected
    """
    # =========================================================================
    # j +/- diagonals
    for i in range(nlat):
        j_minus_diag[ij_to_k(i, 0, nlon)] = 0
        j_plus_diag[ij_to_k(i, nlon-1, nlon)] = 0
    j_minus_diag = j_minus_diag[1:]
    j_plus_diag = j_plus_diag[:-1]
    
    # i +/- diagonals
    i_minus_diag = i_minus_diag[nlon:]
    i_plus_diag = i_plus_diag[:-nlon]
    # =========================================================================
    
    
    """
    The sparse matrix is initialized from the diagonals in the CSC (Compressed Sparse Column)
    format (more effective for computations). Some pair of indices are defined 
    both in the diagonals and the misplaced elements. The corresponding values
    are implicitly added during the conversion.
    """
    # =========================================================================
    WLinv_diags = sp.diags([i_minus_diag, j_minus_diag, eq_diag, j_plus_diag, i_plus_diag],
                  [-nlon, -1, 0, 1, nlon], format = 'coo')

    data = np.concatenate((WLinv_diags.data, BC_data))
    row = np.concatenate((WLinv_diags.row, BC_row))
    col = np.concatenate((WLinv_diags.col, BC_col))
    
    WLinv = sp.csc_matrix((data, (row, col)), shape = (N, N))
    
    return WLinv
    # =========================================================================
    
    


def grid_geometry(res, water_ratio = None, surface_model = 'ellipsoid',
                  a = 6378136.6, f = 1./0.29825765000000E+03):
    """
    Returns multiple fields of distances and surface describing the geometry of 
    the grid for a given surface model of the earth.

    Parameters
    ----------
    res : tuple
        Tuple of two elements : the resolutions in latitude and longituide of the grid in degrees.
   water_ratio : array, optional
        Array of dimension (nlat,nlon) of the ocean surface ratio of eacdh grid cell.
    surface_model : str, optional
        Model used for the shape of the Earth. If equal to 'sphere', a sphere 
        of radius a models the Earth. If equal to 'ellipsoid', a revolution 
        ellipsoid of equatorial radius a and flattening f is used. The default 
        is 'ellipsoid'.
    a : float, optional
        Mean equatorial radius of the earth. The default is 6378136.6.
    f : float, optional
        Flattening of the earth. The default is  1./0.29825765000000E+03.

    Returns
    -------
    ei_center : 2D array of floats
        ei_center[i,j] is the width of the grid cell on the i-th line and j-th 
        column in the direction given by the coordinate i (usually meridional).
        Its dimensions are  180//res * 360//res.
    ej_center : 2D array of floats
        ej_center[i,j] is the width of the grid cell on the i-th line and j-th 
        column in the direction given by the coordinate j (usually zonal).
        Its dimensions are  180//res * 360//res.
    ei_parallel : 2D array of floats
        ei_parallel[i,j] is the distance between the centers of the grid cell above and below 
        the i-th parallel at the longitude of the j-th grid cells centers.
        Its dimensions are  ( 180//res +1 ) * 360//res.
    ej_parallel : 2D array of floats
        ej_parallel[i,j] is the length of the parallel arc on the j-th division
        of the i-th parallel.
        Its dimensions are  ( 180//res +1 ) * 360//res.
    ei_meridian : 2D array of floats
        ei_meridian[i,j] is the length of the meridian arc on the i-th division
        of the j-th meridian.
        Its dimensions are  180//res * (360//res + 1).
    ej_meridian : 2D array of floats
        ej_meridian[i,j] is the distance between the centers of the grid cell left and right 
        the j-th meridian at the latitude of the i-th grid cells centers.
        Its dimensions are  180//res * (360//res + 1).
    w : 2D array of floats
        w[i,j] is the surface of the grid cell on the i-th line and j_th column.

    """
    
    if surface_model == 'ellipsoid':
        # Excentricité secondaire au carré
        es2 =  1/(1-f)**2 - 1 
    
    
        # CENTERS
        # =========================================================================
        lat_center = lat(res[0], centered = True)
        # Geodetic to geocentric latitude
        lat_sph_center = np.arctan( (1-f)**2 * np.tan(lat_center * np.pi/180) )
        # Cell length in meridional direction
        ei_center_1D = a * (1-f)**2 * np.sqrt(1 + es2 * np.sin(lat_sph_center)**2)**3 * res[0] * np.pi/180
        # Cell length in parallel direction
        ej_center_1D = a * np.cos(lat_sph_center) * res[1] * np.pi/180
        nlon_center = int(360//res[1]) 
        w_1D = ei_center_1D * ej_center_1D * (1 + f * np.sin(lat_sph_center)**2)
        # Replicate on all longitudes
        ei_center, ej_center, w = np.repeat(ei_center_1D[:, np.newaxis], nlon_center, axis = 1),\
                                  np.repeat(ej_center_1D[:, np.newaxis], nlon_center, axis = 1),\
                                  np.repeat(w_1D[:, np.newaxis], nlon_center, axis = 1)
        if water_ratio is not None:
            w[water_ratio > 0] *= water_ratio[water_ratio > 0]
        # =========================================================================
    
        # MERIDIANS
        # =========================================================================
        nlon_meridian = nlon_center + 1
        ei_meridian, ej_meridian = np.repeat(ei_center_1D[:, np.newaxis], nlon_meridian, axis = 1),\
                                  np.repeat(ej_center_1D[:, np.newaxis], nlon_meridian, axis = 1),\
        # =========================================================================
         
        # PARALLELS
        # =========================================================================
        lat_parallel = lat(res[0], centered = False)
        # Geodetic to geocentric latitude
        lat_sph_parallel = np.arctan( (1-f)**2 * np.tan(lat_parallel * np.pi/180) )
        # Cell length in meridional direction
        ei_parallel_1D = a * (1-f)**2 * np.sqrt(1 + es2 * np.sin(lat_sph_parallel)**2)**3 * res[0] * np.pi/180
        # Cell length in parallel direction
        ej_parallel_1D = a * np.cos(lat_sph_parallel) * res[1] * np.pi/180
        # Enforce ej_parallel exactly equal to 0 instead of 1E-12 due to the degree radian conversion
        ej_parallel_1D[0], ej_parallel_1D[-1] = 0,0
        # Replicate on all longitudes
        ei_parallel, ej_parallel = np.repeat(ei_parallel_1D[:, np.newaxis], nlon_center, axis = 1),\
                                  np.repeat(ej_parallel_1D[:, np.newaxis], nlon_center, axis = 1),\
        # =========================================================================
    
    elif surface_model == 'sphere':
        # CENTERS
        # =========================================================================
        lat_center = lat(res[0], centered = True)
        # Cell length in meridional direction
        ei_center_1D = a *  res[0] * np.pi/180
        # Cell length in parallel direction
        ej_center_1D = a * np.cos(lat_center) * res[1] * np.pi/180
        nlon_center = int(360//res[1]) 
        w_1D = ei_center_1D * ej_center_1D 
        # Replicate on all longitudes
        ei_center, ej_center, w = np.repeat(ei_center_1D[:, np.newaxis], nlon_center, axis = 1),\
                                  np.repeat(ej_center_1D[:, np.newaxis], nlon_center, axis = 1),\
                                  np.repeat(w_1D[:, np.newaxis], nlon_center, axis = 1)
        if water_ratio is not None:
            w[water_ratio > 0] *= water_ratio[water_ratio > 0]
        # =========================================================================
    
        # MERIDIANS
        # =========================================================================
        nlon_meridian = nlon_center + 1
        ei_meridian, ej_meridian = np.repeat(ei_center_1D[:, np.newaxis], nlon_meridian, axis = 1),\
                                  np.repeat(ej_center_1D[:, np.newaxis], nlon_meridian, axis = 1),\
        # =========================================================================
         
        # PARALLELS
        # =========================================================================
        lat_parallel = lat(res[0], centered = False)
        # Cell length in meridional direction
        ei_parallel_1D = a * res[0] * np.pi/180
        # Cell length in parallel direction
        ej_parallel_1D = a * np.cos(lat_parallel) * res[1] * np.pi/180
        # Enforce ej_parallel exactly equal to 0 instead of 1E-12 due to the degree radian conversion
        ej_parallel_1D[0], ej_parallel_1D[-1] = 0,0
        # Replicate on all longitudes
        ei_parallel, ej_parallel = np.repeat(ei_parallel_1D[:, np.newaxis], nlon_center, axis = 1),\
                                  np.repeat(ej_parallel_1D[:, np.newaxis], nlon_center, axis = 1),\
        # =========================================================================
    
    elif surface_model == 'flat':
        # CENTERS
        # =========================================================================
        lat_center = lat(res[0], centered = True)
        # Cell length in meridional direction
        ei_center_1D = a *  res[0] * np.pi/180
        # Cell length in parallel direction
        ej_center_1D = a * res[1] * np.pi/180
        nlon_center = int(360//res[1]) 
        w_1D = ei_center_1D * ej_center_1D 
        # Replicate on all longitudes
        ei_center, ej_center, w = np.repeat(ei_center_1D[:, np.newaxis], nlon_center, axis = 1),\
                                  np.repeat(ej_center_1D[:, np.newaxis], nlon_center, axis = 1),\
                                  np.repeat(w_1D[:, np.newaxis], nlon_center, axis = 1)
        if water_ratio is not None:
            w[water_ratio > 0] *= water_ratio[water_ratio > 0]
        # =========================================================================
    
        # MERIDIANS
        # =========================================================================
        nlon_meridian = nlon_center + 1
        ei_meridian, ej_meridian = np.repeat(ei_center_1D[:, np.newaxis], nlon_meridian, axis = 1),\
                                  np.repeat(ej_center_1D[:, np.newaxis], nlon_meridian, axis = 1),\
        # =========================================================================
         
        # PARALLELS
        # =========================================================================
        lat_parallel = lat(res[0], centered = False)
        # Cell length in meridional direction
        ei_parallel_1D = a * res[0] * np.pi/180
        # Cell length in parallel direction
        ej_parallel_1D = a * res[1] * np.pi/180
        # Replicate on all longitudes
        ei_parallel, ej_parallel = np.repeat(ei_parallel_1D[:, np.newaxis], nlon_center, axis = 1),\
                                  np.repeat(ej_parallel_1D[:, np.newaxis], nlon_center, axis = 1),\
        # =========================================================================
        
    return ei_center, ej_center, ei_parallel, ej_parallel ,ei_meridian, ej_meridian, w     
    
