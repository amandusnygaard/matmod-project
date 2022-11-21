import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from scipy.constants import N_A
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

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

@njit
def generate_CN_matrix(dx,dy,dz,dt,D,nx,ny,nz,C_R,k_on):
    
    vx = D*dt/(2*dx**2)
    vy = D*dt/(2*dy**2)
    vz = D*dt/(2*dz**2)
    
    print(vx,vy,vz)
    
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
    
    if np.any(np.diag(A) == 0):
        print("feil i matrisen")
    return A

@njit
def update_CN_matrix(A,C_R,dx,dy,dz,dt,D,k_on,nx,ny,nz):
    
    vx = D*dt/(2*dx**2)
    vy = D*dt/(2*dy**2)
    vz = D*dt/(2*dz**2)
    
    for i in range(nx):
        for j in range(ny):
            A[p(i,j,nz-1,nx,ny,nz),p(i,j,nz-1,nx,ny,nz)] = 1+2*vx+2*vy+2*vz+dt*k_on*C_R[i,j]

@njit
def generate_b(C_N,C_R,C_RN,dx,dy,dz,dt,D,k_on,k_off,nx,ny,nz):
    
    # C_R,C_N,C_RN as 3d grids.
    
    vx = D*dt/(2*dx**2)
    vy = D*dt/(2*dy**2)
    vz = D*dt/(2*dz**2)
    
    b = np.zeros(nx*ny*nz)
    
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                
                reac = k_off*C_RN[i,j]*dt if k == nz-1 else 0
                
                b[p(i,j,k,nx,ny,nz)] = (1-2*vx-2*vy-2*vz)*C_N[i,j,k]+reac
                
                if i != 0 and i != nx-1:
                    b[p(i,j,k,nx,ny,nz)] += vx*(C_N[i+1,j,k]+C_N[i-1,j,k])
                elif i == 0:
                    b[p(i,j,k,nx,ny,nz)] += 2*vx*C_N[1,j,k]
                elif i == nx-1:
                    b[p(i,j,k,nx,ny,nz)] += 2*vx*C_N[nx-2,j,k]
                
                if j != 0 and j != ny-1:
                    b[p(i,j,k,nx,ny,nz)] += vy*(C_N[i,j+1,k]+C_N[i,j-1,k])
                elif j == 0:
                    b[p(i,j,k,nx,ny,nz)] += 2*vy*C_N[i,1,k]
                elif j == ny-1:
                    b[p(i,j,k,nx,ny,nz)] += 2*vy*C_N[i,ny-2,k]
                
                if k != 0 and k != nz-1:
                    b[p(i,j,k,nx,ny,nz)] += vz*(C_N[i,j,k+1]+C_N[i,j,k-1])
                elif k == 0:
                    b[p(i,j,k,nx,ny,nz)] += 2*vz*C_N[i,j,1]
                elif k == nz-1:
                    b[p(i,j,k,nx,ny,nz)] += 2*vz*C_N[i,j,nz-2]

    return b


#physical constants
r_c = 0.22e-6                          #[m]
h_c = 15e-9                            #[m]
alpha = 8e-7                           #[m2 s^-1]
k_on = 4e3                             #[(mol/m3)^-1 s^-1]
k_off = 5                              #[s-1]
N0 = 1600
cN0 = N0/(np.pi*r_c**2*h_c*N_A)        #[mol/m3]

#scales
l_s = h_c
c_s = cN0
T_s = 1/(c_s*k_on)
print(l_s,c_s,T_s)

#scaled constants
D = alpha*T_s/l_s**2
lam = k_on*c_s*T_s
mu = k_off*T_s
print(D,lam,mu)

n = 7
nx = ny = 2*n+1
nz = 15
print(nx,ny,nz)

dx = 2*r_c/(nx*l_s)
dy = 2*r_c/(ny*l_s)
dz = h_c/(nz*l_s)
dt = 1e-6/T_s
print(dx,dy,dz,dt)

gridN = np.zeros((nx,ny,nz))
gridN[n,n,0] = N0/(dx*dy*dz*l_s**3*N_A*c_s)
gridR = np.zeros((nx,ny))

cR0 = 1000*(1e6)**2/(dz*l_s*N_A*c_s)

for i in range(nx):
    for j in range(ny):
        if (i-n)**2+(j-n)**2 <= n**2:
            gridR[i,j] = cR0

gridRN = np.zeros((nx,ny))

A = generate_CN_matrix(dx,dy,dz,dt,D,nx,ny,nz,gridR,lam)

it = 0
t = 0

total_receptors = np.sum(gridR)

while np.sum(gridR)/total_receptors >= 0.5 and t <= 0.4e-3/T_s:

    t += dt
    it += 1

    gridR0 = np.copy(gridR)
    gridRN0 = np.copy(gridRN)
    
    b = generate_b(gridN,gridR0,gridRN0,dx,dy,dz,dt,D,lam,mu,nx,ny,nz)
    
    array_N = grid_to_array(gridN,nx,ny,nz)
    
    array_N = spsolve(csr_matrix(A),b)

    gridN = array_to_grid(array_N,nx,ny,nz)
    
    gridR = (gridR0+dt*mu*gridRN0)/(1+dt*lam*gridN[:,:,-1])
    gridRN = (gridRN0+dt*lam*gridR0*gridN[:,:,-1])/(1+mu*dt)
    
    update_CN_matrix(A,gridR,dx,dy,dz,dt,D,lam,nx,ny,nz)
    
    #if it%5 == 0:
    #print(np.sum(gridR)/total_receptors)

if np.sum(gridR)/total_receptors <= 0.5:
    print("Signal fired")
    print(f"Percent reacted: {np.sum(gridR)/total_receptors*100}")
    print(f"Time to fire signal: {t*1e3*T_s} ms")
else:
    print("Signal not fired")
    print(f"Percent reacted reached: {np.sum(gridR)/total_receptors*100}")


print(np.min(gridN[:,:,-1]))
plt.figure()
plt.imshow(gridN[:,:,-1], vmin = 0, vmax = np.max(gridN[:,:,-1]))
plt.colorbar()
plt.show()

plt.figure()
plt.imshow(gridR, vmin = 0, vmax = cR0)
plt.colorbar()
plt.show()

plt.figure()
plt.imshow(gridRN, vmin = 0, vmax = cR0)
plt.colorbar()
plt.show()

print(np.sum(gridR),np.sum(gridRN),total_receptors)