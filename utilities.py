import numpy as np
from numba import njit

@njit
def p(i,j,k,nx,ny,nz):
    return i + nx*(j+ny*k)

@njit
def grid_to_array(grid,nx,ny,nz):
    
    array = np.zeros(nx*ny*nz)
    
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                array[p(i,j,k,nx,ny,nz)] = grid[i,j,k]
    return array

@njit
def array_to_grid(array,nx,ny,nz):
    
    grid = np.zeros((nx,ny,nz))
    
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                 grid[i,j,k] = array[p(i,j,k,nx,ny,nz)]
    
    return grid