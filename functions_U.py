import numpy as np
import numpy as np
import random
from scipy.linalg import block_diag
from scipy.sparse import coo_matrix
from scipy.optimize import curve_fit
from scipy.linalg import eig, eigh
from scipy.stats import unitary_group

## LAYER DEFINITIONS

# single layer beam splitter, with some mixing angle theta and offset, periodic boundaries
def BS_layer_U(N,theta,offset=0,phi=0.):
    
    BS = [[np.cos(theta), np.sin(theta)],[-np.sin(theta), np.cos(theta)]]    
    N_BS = N//2    
    BS_layer = np.kron(np.eye(N_BS),BS)

    return np.roll(np.roll(BS_layer,offset,axis=0),offset,axis=1)
    
# single layer beam splitter, with some mixing angle theta and offset, open boundaries    
def BS_layer_U_open(N,theta,offset=0,phi=0.):
    
    BS = [[np.cos(theta), np.sin(theta)],[-np.sin(theta), np.cos(theta)]]    
    N_BS = (N-offset)//2    
    BS_layer = np.kron(np.eye(N_BS),BS)

    if offset == 0 and N%2==0:
        return BS_layer
    elif offset == 1 and N%2==0:
        return block_diag(1.,BS_layer,1.)
    elif offset == 0 and N%2==1:
        return block_diag(BS_layer,1.)
    elif offset == 1 and N%2==1:
        return block_diag(1.,BS_layer)

# single layer of (pseudo)random phases   
def phase_layer_U(N, w, type='random'):
    if type =="random":
      v_phi = -w + 2.*w*np.random.rand(N) # between -w and w
    elif type=="pseudo":
      n_lattice = np.arange(N)
      ratio, rand_phi = 1.+np.random.random(), 2.*np.pi*np.random.random()
      v_phi = w*np.cos( 2.*np.pi*ratio*n_lattice+rand_phi )

    return np.diag( np.exp(1j*v_phi) )
    
# single layer of random 2x2 unitary
def U_perc_layer(N, p=0., offset=0, d=2):
    N_gates = (N-offset) // d
    
    U = np.identity(N, dtype="complex")
    
    for i in range(N_gates):
        if p < random.random():
            ind = slice(offset+d*i,offset+d*(i+1))
            U[ind,ind] = unitary_group.rvs(d)
    
    return U
        

## GENERATE CIRCUIT FROM LAYERS

# create D-layered network
def network_U_eff(N,D,w, offset=0, theta=np.pi/4, bc="periodic", phase_type="random"):

    if bc=="periodic":
      BS_o = coo_matrix(BS_layer_U(N,theta,offset=(0+offset)%2))
      BS_e = coo_matrix(BS_layer_U(N,theta,offset=(1+offset)%2))
    elif bc=="open":
      BS_o = coo_matrix(BS_layer_U_open(N,theta,offset=(0+offset)%2))
      BS_e = coo_matrix(BS_layer_U_open(N,theta,offset=(1+offset)%2))

    phase = coo_matrix(phase_layer_U(N,w, type=phase_type))

    return (phase*BS_o*phase*BS_e)**D
    
def network_U_percolation(N,D,p=0,offset=0,d=2):
    U = np.identity(N)
    
    for i in range(D):
        U = U@U_perc_layer(N,p,offset%d,d)
        offset+=1
    
    return U
    
    
    
    
    