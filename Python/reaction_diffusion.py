import numpy as np
from numba import njit
from utilities import p

def generate_CN_matrix(dx,dy,dz,dt,D,nx,ny,nz,C_R,k_on):
    
    vx = D*dt/(dx**2)
    vy = D*dt/(dy**2)
    vz = D*dt/(dz**2)
    
    A = np.zeros((nx*ny*nz,nx*ny*nz))
    
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                
                reac = dt*k_on*C_R[i,j] if k == nz-1 else 0
                
                A[p(i,j,k,nx,ny,nz),p(i,j,k,nx,ny,nz)] = 1+2*vx+2*vy+2*vz + reac

                if i != 0 and i != nx-1:
                    A[p(i,j,k,nx,ny,nz),p(i+1,j,k,nx,ny,nz)] = -vx
                    A[p(i,j,k,nx,ny,nz),p(i-1,j,k,nx,ny,nz)] = -vx
                elif i == 0:
                    A[p(i,j,k,nx,ny,nz),p(1,j,k,nx,ny,nz)] = -2*vx
                elif i == nx-1:
                    A[p(i,j,k,nx,ny,nz),p(nx-2,j,k,nx,ny,nz)] = -2*vx
                
                if j != 0 and j != ny-1:
                    A[p(i,j,k,nx,ny,nz),p(i,j+1,k,nx,ny,nz)] = -vy
                    A[p(i,j,k,nx,ny,nz),p(i,j-1,k,nx,ny,nz)] = -vy
                elif j == 0:
                    A[p(i,j,k,nx,ny,nz),p(i,1,k,nx,ny,nz)] = -2*vy
                elif j == nx-1:
                    A[p(i,j,k,nx,ny,nz),p(i,ny-2,k,nx,ny,nz)] = -2*vy
                
                if k != 0 and k != nz-1:
                    A[p(i,j,k,nx,ny,nz),p(i,j,k+1,nx,ny,nz)] = -vz
                    A[p(i,j,k,nx,ny,nz),p(i,j,k-1,nx,ny,nz)] = -vz
                elif k == 0:
                    A[p(i,j,k,nx,ny,nz),p(i,j,1,nx,ny,nz)] = -2*vz
                elif k == nz-1:
                    A[p(i,j,k,nx,ny,nz),p(i,j,nz-2,nx,ny,nz)] = -2*vz
    
    return A

@njit
def update_CN_matrix(A,C_R,dx,dy,dz,dt,D,k_on,nx,ny,nz):
    
    vx = D*dt/(dx**2)
    vy = D*dt/(dy**2)
    vz = D*dt/(dz**2)
    
    for i in range(nx):
        for j in range(ny):
            A[p(i,j,nz-1,nx,ny,nz),p(i,j,nz-1,nx,ny,nz)] = 1+2*vx+2*vy+2*vz+dt*k_on*C_R[i,j]

@njit
def generate_b(C_N,C_R,C_RN,dx,dy,dz,dt,D,k_on,k_off,nx,ny,nz):
    
    # C_R,C_N,C_RN as 3d grids.
    
    vx = D*dt/(dx**2)
    vy = D*dt/(dy**2)
    vz = D*dt/(dz**2)
    
    b = np.zeros(nx*ny*nz)
    
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                
                reac = k_off*C_RN[i,j]*dt if k == nz-1 else 0
                
                b[p(i,j,k,nx,ny,nz)] = C_N[i,j,k]+reac

    return b