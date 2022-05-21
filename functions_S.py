import numpy as np
import random
from scipy.linalg import block_diag
from scipy.sparse import coo_matrix
from scipy.optimize import curve_fit
from scipy.linalg import eig, eigh

######################################################################    
## FUNCTIONS TO EVALUATE COVARIANCE MATRIX AND ENTANGLEMENT ENTROPY

# obtain covariance matrix from squeezed state, either in (a, ad) representation or (q,p)
def get_covariance_matrix(U, V, type="ada"):
  # check symplectic transform
  assert np.max( np.abs(U@V.T - V@U.T) ) < 1e-3 and np.max( np.abs(U@U.H - V@V.H - np.eye(U.shape[0]))) < 1e-3,"U, V no symplectic transform"

  #if ind is None:
   # N = U.shape[0]
    #ind = (slice(0,N), slice(0,N))

  # ad, a correlations
  aa = (U@V.T)
  adad = (np.conj(V)@U.H)
  aad = (U@U.H)
  ada = (np.conj(V)@V.T)

  if type=="ada":
    C = np.block([
              [aa,aad],
              [ada,adad]
              ])
  else:
    # p, q corrlations
    qq = 0.5*(ada + aad + aa + adad)
    pp = 0.5*(ada + aad - aa - adad)
    qp = 1j*0.5*(-ada + aad - aa + adad)
    pq = 1j*0.5*(ada - aad - aa + adad)


    C = np.block([
              [qq, qp],
              [pq, pp]
              ])
  return C

# get entropy from some correlation matrix, at a bipartition determined by cut. Full state assumed to be pure
def get_entropy(C, cut=None, ind=None, type="ada"):
  
  N = C.shape[0] // 2
  n_el = N
  
  # check for total system or zero cut, which has zero entropy if state is pure
  if cut==0 or cut==N:
      return 0.

  # check for bipartition cut, choose first or second half of block for computation
  if not cut is None:
    if cut <= N//2:
      ind = slice(0,cut)
      n_el = cut
    else:
      ind = slice(cut,N)
      n_el = N-cut

  if not ind is None:
    ind2 = (ind,ind)

  # construct reduced C
  Cc = np.block([
                 [C[ind2], np.roll(C,N,axis=1)[ind2]],
                 [np.roll(C,N,axis=0)[ind2], np.roll(C,N,axis=(0,1))[ind2]]
  ])

# The commutation matrix
  Omega = np.block([
                [np.zeros((n_el,n_el)), np.eye(n_el)],
                [-np.eye(n_el), np.zeros((n_el,n_el))]
                ])
  
  # find symplectic eigenvalues of reduced correlation matrix
  e = eig((Cc+Cc.T)@Omega)[0]
  
  # obtain occupations of diagonlized modes from this, factor i difference for (ad,a) or (p,q) case
  if type == "ada":
    n = 0.5*(np.real(e[np.real(e)>1.])-1. )

  if type =="pq":
    n = 0.5*(np.imag(e[np.imag(e)>1.])-1. )
  
  # return entropy, evaluated from occupation of Bogoliubov modes
  return np.sum( (1.+n)*np.log(1.+n) - n*np.log(n) )
  
