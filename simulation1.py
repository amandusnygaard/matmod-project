import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.constants import N_A

from utilities import *
from params import *
from reaction_diffusion import *

#physical constants
r_c,h_c,alpha,k_on,k_off = phys_param

# Initial number of Nerotransmitters sent out
N0 = 2000
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
dt = 1e-5/T_s
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


plt.figure()
plt.title("Concentration of N at reaction layer")
plt.imshow(gridN[:,:,-1], vmin = 0, vmax = np.max(gridN[:,:,-1]))
plt.colorbar()
plt.show()

plt.figure()
plt.title("Concentration of R at reaction layer")
plt.imshow(gridR, vmin = 0, vmax = cR0)
plt.colorbar()
plt.show()

plt.figure()
plt.title("Concentration of R-N at reaction layer")
plt.imshow(gridRN, vmin = 0, vmax = cR0)
plt.colorbar()
plt.show()

print(np.sum(gridR),np.sum(gridRN),total_receptors)