import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from autograd import grad
import autograd.numpy as np
import scipy.integrate

__all__ = ['Massm', 'hat', 'MatrR', 'Hp', 'Hq', 'dynamics']

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

e3 = np.array([0,0,1]) # third axis of the intertial frame

def Massm(nop,m):
  """ Mass matrix collecting the terms mij in the matrix defining the kinetic energy

  Parameters
  ----------
  nop : int
    number of pendulums
  m : float
    masses of the pendulums (supposed to be all the same in this case)

  Returns
  -------
  M : numpy.ndarray
    Mass matrix

  """
  M=np.zeros((nop,nop))
  for i in range(nop):
    M[i,i]=(nop-i)*m
    for j in range(i):
      M[i,j] = (nop-i)*m
    for k in range(i):
      M[k,i]=M[i,j]

  return M



def hat(q):
  """" 
  Isomorphism bewteen R3 and so(3). It returns the skew symmetric matrix hat(q) associated to the vector q, 
  such that hat(a)b=axb for all 3-component vectors a and b, with "x" the cross product

  Parameters
  ----------
  q : numpy.ndarray
    3-component vector

  Returns
  -------
  hat(q): numpy.ndarray
    3x3 skew-symmetric matrix (element of the Lie algebra so(3))

  """
  return np.array([[0.,-q[2],q[1]],[q[2],0.,-q[0]],[-q[1],q[0],0.]])



def MatrR(q):
  """
  Matrix defining the quadratic kinetic energy

  Parameters
  ----------
  q : numpy.ndarray
    3N-component vector of coordinates, with N the number of pendulums
    
  Returns
  -------
  R : numpy.ndarray
    3Nx3N matrix defining the kinetic energy, with N the number of pendulums
  """
  m = 1.
  L = 1.
  n = len(q)
  nn = int(n/3)
  M=Massm(nn,m)
  R = M[0,0]*np.eye(3)
  for j in range(1,nn):
        R = np.concatenate((R,M[0,j]*(np.eye(3)-np.outer(q[0:3],q[0:3]))), axis=1)
  for i in range(1,nn):
    for j in range(nn):
      if j in [i]:
        R = np.concatenate((R,M[i,i]*np.eye(3)), axis=1)
      else:
        R = np.concatenate((R,M[i,j]*(np.eye(3)-np.outer(q[3*i:3*i+3],q[3*i:3*i+3]))), axis=1)
  row=R[0:3,0:n]
  for i in range(1,nn):
    row = np.concatenate((row,R[0:3,n*i:n*i+n]), axis=0)
  return row



def Hp(z):
  """
  Gradient of the Hamiltonian with respect to the conjugate momenta

  Parameters
  ----------
  z : numpy.ndarray
    6N-component vector of generalized coordinates and momenta, with N the number of pendulums
    
  Returns
  -------
  Hp : numpy.ndarray
    3N-component vector, with N the number of pendulums

  """
  nop = int(len(z)/6)
  q = z[0:3]
  p = z[3:6]
  for i in range(1,nop):
    q = np.concatenate((q,z[3*(2*i):3*(2*i+1)]))
    #print('q=',q)
    p = np.concatenate((p,z[3*(2*i+1):3*(2*i+2)]))
    #print('p=',p)
  return 0.5 * np.linalg.solve(MatrR(q),p) + 0.5 * np.linalg.solve(MatrR(q).T,p)


def Hq(z):
  """
  Gradient of the Hamiltonian with respect to the configuration variables

  Parameters
  ----------
  z : numpy.ndarray
    6N-component vector of configuration variables and conjugate momenta, with N the number of pendulums
    
  Returns
  -------
  Hp : numpy.ndarray
    3N-component vector, with N the number of pendulums

  """
  e3 = np.array([0,0,1]) # third axis of the intertial frame
  nop = int(len(z)/6)
  g = 1
  m = 1
  L = 1
  q = z[0:3]
  p = z[3:6]
  for i in range(1,nop):
    q = np.concatenate((q,z[3*(2*i):3*(2*i+1)]))
    p = np.concatenate((p,z[3*(2*i+1):3*(2*i+2)]))
  func = lambda v: np.dot(p,np.linalg.solve(MatrR(v),p))
  nablaq = grad(func)
  nablaqq = nablaq(q)  
  res = m*g*L*e3*nop
  for j in range (2,nop+1):
    res = np.concatenate((res, (nop-j+1)*m*L*g*e3))
  return 0.5*nablaqq + res


def dynamics(t,z):
  """
  System of ODEs defining the dynamics

  Parameters
  ----------
  t : float
    1-D independent variable (time)
      
  z : numpy.ndarray
    6N-component vector of configuration variables and conjugate momenta, with N the number of pendulums

  Returns
  -------
  vec : numpy.ndarray
    3N-component vector, with N the number of pendulums

  """  
  nop = int(len(z)/6)
  vec = 0*z
  I = np.eye(3)

  q = z[0:3]
  p = z[3:6] 
  vec[:3] = (I-np.outer(q,q))@Hp(z)[:3]
  vec[3:6] = -(I-np.outer(q,q))@Hq(z)[:3] + np.cross(Hp(z)[:3],np.cross(p,q))

  for i in range(1,nop):
    q = z[3*(2*i):3*(2*i+1)]
    p = z[3*(2*i+1):3*(2*i+2)] 
    vec[3*(2*i):3*(2*i+1)]= (I-np.outer(q,q))@Hp(z)[3*i:3*(i+1)]
    vec[3*(2*i+1):3*(2*i+2)] = -(I-np.outer(q,q))@Hq(z)[3*i:3*(i+1)] + np.cross(Hp(z)[3*i:3*(i+1)],np.cross(p,q))
  
  return vec