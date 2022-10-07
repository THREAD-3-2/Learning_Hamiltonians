import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from autograd import grad
import autograd.numpy as np
import scipy.integrate

__all__ = ['hatNN', 'fManiAlgebraNN', 'expso3NN', 'expse3NN', 'expse3NNn', 'actionSE3NN', 'actionse3NNn', 'LieEulerNN', 'CF4NN', 'predictedVF', 'ExpEuler', 'RK4']

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

e3 = np.array([0,0,1]) # third axis of the intertial frame

def hatNN(q):
  """
  It returns the skew symmetric matrix associated to the vector q, such
  that hat(a)b=axb for all 3-component vectors a and b, with "x" the cross product

  Parameters
  ----------
  q: torch.Tensor
     coordinates of the training trajectory points, with shape [batch size, s], s=3

  Returns
  -------
  res : torch.Tensor
        skew symmetric matrix associated to each vector q in the batch, with shape [batch size, s, s], s=3

  """ 
  zz = torch.zeros([len(q),1]).to(device)  #len(q) number of points in the batch
  row1 = torch.cat([zz,-q[:,2:3],q[:,1:2]],axis=1).unsqueeze(1).to(device)
  row2 = torch.cat([q[:,2:3],zz,-q[:,0:1]],axis=1).unsqueeze(1).to(device)
  row3 = torch.cat([-q[:,1:2],q[:,0:1],zz],axis=1).unsqueeze(1).to(device)
  res = torch.cat([row1,row2,row3],axis=1).to(device) # 3d tensor
  return res


def fManiAlgebraNN(H,z):
  """
  Funtion f from the manifold M (phase space of the system) to the Lie algebra of the group acting on M

  Parameters
  ----------
  H: torch.Tensor
    Hamiltonian function

  z: torch.Tensor
    training trajectory points, with size [batch size, nop*2s]

  Returns
  -------
  ff : torch.Tensor
    map f : M -> g (Lie algebra)

  """ 
  nop = z.size(dim=1)//6
  HamVal = torch.sum(H(z)).to(device)
  dH = torch.autograd.grad(HamVal, z, create_graph=True)[0].to(device)

  q = z[:,0:3]
  p = z[:,3:6]
  dHq = dH[:,0:3]
  dHp = dH[:,3:6]
  xi = torch.cross(q,dHp)
  eta = torch.cross(dHq,q)+torch.cross(dHp,p)
  ff = torch.cat([xi,eta],axis=1).to(device)

  for i in range(1,nop):
    q = z[:,3*(2*i):3*(2*i+1)]
    p = z[:,3*(2*i+1):3*(2*i+2)]
    dHq = dH[:,3*(2*i):3*(2*i+1)]
    dHp = dH[:,3*(2*i+1):3*(2*i+2)]
    xi = torch.cross(q,dHp)
    eta = torch.cross(dHq,q)+torch.cross(dHp,p)
    ff = torch.cat([ff,xi,eta],axis=1)

  return ff


def expso3NN(x):
  """
  Exponential map on SO(3)

  Parameters
  ----------
  x : torch.Tensor (float32)
    element of the lie algebra so(3), represented as a vector with 3 components.

  Returns
  -------
  expA : torch.Tensor
    element of the group SO(3), i.e. 3x3 rotation matrix

  """ 
  a = torch.linalg.norm(x,axis=1).to(device)
  tol = 10**(-10)
  aa = ((a>=tol)*1)
  dd = ((a>0)*1)*((a<tol)*1)
  expA = torch.eye(3).unsqueeze(0).repeat(len(x),1,1).to(device)
  cc = aa.nonzero(as_tuple=True)
  #if a>tol:
  A = torch.zeros(len(expA)).to(device)
  B = torch.zeros(len(expA)).to(device)
  A[cc] = (torch.div(torch.sin(a[cc]),a[cc]))
  B[cc] = ((1-torch.cos(a[cc]))/a[cc]**2)

  expA += torch.einsum('i,ijk->ijk',A,hatNN(x)) + torch.einsum('i,ijk->ijk',B,torch.einsum('ijk,ikl->ijl',hatNN(x),hatNN(x)))

  #if a>0 and a<tol:
  mult1 = 1 * dd
  mult2 = 1/2 * dd
  mult3 = 1/6 * dd
  mult4 = 1/24 * dd

  pow1 = hatNN(x)
  
  pow2 = (torch.einsum('ijk,ikl->ijl',pow1,pow1))
  pow3 = (torch.einsum('ijk,ikl->ijl',pow2,pow1))
  pow4 = (torch.einsum('ijk,ikl->ijl',pow3,pow1))
  
  expA += torch.einsum('i,ijk->ijk',mult1,pow1) + torch.einsum('i,ijk->ijk',mult2,pow2) + torch.einsum('i,ijk->ijk',mult3,pow3) + torch.einsum('i,ijk->ijk',mult4,pow4)
  return expA


def expse3NN(input):
  """
  Exponential map on SE(3)

  Parameters
  ----------
  x: torch.Tensor (float32)
    element of the lie algebra se(3) represented as 6-component vector,
    i.e. as a pair (u,v) with with the 3-component vector u corresponding 
    to a skew symmetric matrix hat(u) and the 3-component vector v
    corresponding to the translational part.

  Returns
  -------
  expA : torch.Tensor
    element of the group SE(3), represented as a 3x4 matrix [A, b], 
    with A 3x3 rotation matrix and b 3-component translation vector.

  """ 
  u = input[:,:3]
  v = input[:,3:]
  a = torch.linalg.norm(u,axis=1).to(device)
  tol = 1e-10;  

  cc = a>=tol + 0
  ee = (a<tol)*(a>0) + 0

  V = torch.eye(3).unsqueeze(0).repeat(len(input),1,1).to(device) #the right matrix if a = 0, then we increment with the right quantities 

  #if a>tol:
  A = cc*(torch.div(torch.sin(a),a))
  B = cc*((1-torch.cos(a))/a**2)
  C = cc*(torch.div(1-A,a**2))

  V += torch.einsum('i,ijk->ijk',B,hatNN(u)) + torch.einsum('i,ijk->ijk',C,torch.einsum('ijk,ikl->ijl',hatNN(u),hatNN(u)))
  
  #if 0<a<tol:
  Blow = ee*(0.5-a**2/24 + a**4/720 - a**6/40320)
  Clow = ee*(1/6-a**2/120+a**4/5040-a**6/362880)
  V += torch.einsum('i,ijk->ijk',Blow,hatNN(u)) + torch.einsum('i,ijk->ijk',Clow,torch.einsum('ijk,ikl->ijl',hatNN(u),hatNN(u)))

  expA = torch.cat([expso3NN(u), torch.einsum('ijk,ik->ij',V,v).unsqueeze(2)],axis=2).to(device)

  return expA


def expse3NNn(input):
  """
  Concatenate exponentials on SE(3) in one tensor

  Parameters
  ----------
  input: torch.Tensor

  Returns
  -------
  out : torch.Tensor

  """ 
  dim0 = input.size(dim=0)
  dim1 = input.size(dim=1)
  nop = int(dim1/6)
  out = expse3NN(input[:,0:6])
  for i in range(1,nop):
    out = torch.cat([out, expse3NN(input[:,6*i:6*(i+1)])], axis=2).to(device)
  return out


def actionSE3NN(g, z):
  """
  Group action of SE3 on TS^2

  Parameters
  ----------
  g: torch.Tensor
    element of the group SE3 

  z : torch.Tensor 
    trajectory point in the phase space

  Returns
  -------
  out = torch.Tensor

  """ 
  R = g[:,:,:3]
  r = g[:,:,3]
  q = z[:,0:3]
  p = z[:,3:]
  qq = torch.einsum('ijk,ik->ij',R,q)
  pp = torch.einsum('ijk,ik->ij',R,p) + torch.einsum('ijk,ik->ij',hatNN(r),qq)
  out = torch.cat([qq,pp],axis=1).to(device)
  return out


def actionse3NNn(g, z):
  """
  Concatenate se3 actions

  Parameters
  ----------
  g : torch.Tensor
    elements of the group SE3 

  z : torch.Tensor
    trajectory points in the phase space

  Returns
  -------
  out : torch.Tensor
 
  """ 
  dim0 = z.size(dim=0)
  dim1 = z.size(dim=1)
  nop = int(dim1/6)
  out = actionSE3NN(g[:,:,0:4],z[:,0:6])
  for i in range(1,nop):
     out = torch.cat([out, actionSE3NN(g[:,:,4*i:4*(i+1)],z[:,6*i:6*(i+1)])], axis=1).to(device)
  return out


def LieEulerNN(x0,f,h,cc,H):
  """
  Lie Euler integrator

  Parameters
  ----------
  x0 : torch.Tensor
    solution at time t0

  f : function handle
    map f from the manifold M to the Lie algebra of the group acting on M

  h : float
    time step

  cc : int
    M-1

  H : neural network class
    Hamiltonian function

  Returns
  -------
  sol : solution at time t1=t0+h
 
  """ 
  num = len(x0) # number of points (bacth size)
  sol = torch.zeros([len(x0[:,0]),len(x0[0,:]),cc]).to(device)
  z  = x0
  times = 1
  h = h/times
  for i in range(cc):
    for j in range(times):
      z =  actionse3NNn(expse3NNn(h*f(H,z)),z)
    sol[:,:,i] = z
  return sol


def CF4NN(x0,f,h,cc,H):
  """
  Lie group commutator free integrator of order 4

  Parameters
  ----------
  x0 : torch.Tensor
    solution at time t0

  f : function handle
    map f from the manifold M to the Lie algebra of the group acting on M

  h : float
    time step

  cc : int
    M-1

  H : neural network class
    Hamiltonian function

  Returns
  -------
  sol : solution at time t1=t0+h
 
  """ 
  num = len(x0) #number of points
  sol = torch.zeros([len(x0[:,0]),len(x0[0,:]),cc]).to(device)
  z  = x0
  times = 2
  h = h/times
  for i in range(cc):
    for j in range(times):
      M1 = z
      A1 = f(H,M1)
      M2 = actionse3NNn(expse3NNn(0.5*h*A1),z)
      A2 = f(H,M2)
      M3 = actionse3NNn(expse3NNn(0.5*h*A2),z)
      A3 = f(H,M3)
      M4 = actionse3NNn(expse3NNn(h*A3-0.5*h*A1),M2)
      A4 = f(H,M4)
      mhalf = actionse3NNn(expse3NNn(1/12*h*(3*A1+2*A2+2*A3-A4)),z)  
      z = actionse3NNn(expse3NNn(1/12*h*(-A1+2*A2+2*A3+3*A4)),mhalf)
    sol[:,:,i] = z

  return sol


# Characterization of the Hamiltonian vector field with Hamiltonian given by HH
def predictedVF(x,HH):
  """
  Vector field predicted by using the output of the neural network as the Hamiltonian

  Parameters
  ----------
  x : torch.Tensor
    solution at time t0

  HH : function handle
    Hamiltonian

  Returns
  -------
  vec : torch.Tensor
    vector field (Hamlton equations)

  """
  z = x.clone().requires_grad_().to(device)
  nop = int(len(z[0])/6)
  HamVal = torch.sum(HH(z)) # H sarà la rete ...fatto per prendere il gradiente
  dH = torch.autograd.grad(HamVal, z, create_graph=True)[0]

  vec = torch.zeros(z.shape).to(device)
  q = z[:,0:3]
  p = z[:,3:6]
  dHq = dH[:,:3]
  dHp = dH[:,3:6]
  for i in range(1,nop):
    q = torch.concat((q,z[:,3*(2*i):3*(2*i+1)]),dim=1)
    p = torch.concat((p,z[:,3*(2*i+1):3*(2*i+2)]),dim=1)
    dHq = torch.concat((dHq,dH[:,3*(2*i):3*(2*i+1)]),dim=1)
    dHp = torch.concat((dHp,dH[:,3*(2*i+1):3*(2*i+2)]),dim=1)

  I = torch.eye(3).repeat(len(z),1,1).to(device)
  # part for the first pendulum
  
  for i in range(nop):
    mat1 = (I-torch.einsum('ij,ik->ijk',q[:,3*i:3*(i+1)],q[:,3*i:3*(i+1)]))
    dqdt = torch.einsum('ijk,ik->ij',mat1,dHp[:,3*i:3*(i+1)])
    part1 = -torch.einsum('ijk,ik->ij',mat1,dHq[:,3*i:3*(i+1)])
    part2 = torch.cross(dHp[:,3*i:3*(i+1)],torch.cross(p[:,3*i:3*(i+1)],q[:,3*i:3*(i+1)]))
    dpdt = part1 + part2
    vec[:,6*i:6*(i+1)] = torch.concat([dqdt,dpdt],dim=1)

  return vec


def ExpEuler(x0,h,cc,H):
  """
  Explicit Euler method

  Parameters
  ----------
  x0 : torch.Tensor
    solution at time t0

  f : function handle
    map f from the manifold M to the Lie algebra of the group acting on M

  h : float
    time step

  cc : int
    M-1

  H : neural network class
    Hamiltonian function

  Returns
  -------
  sol : solution at time t1=t0+h

  """
  sol = torch.zeros([len(x0[:,0]),len(x0[0,:]),cc]).to(device)
  z  = x0.clone()
  z.requires_grad_()
  for i in range(cc):
    z = z + h * predictedVF(z,H) 
    sol[:,:,i] = z.to(device)
  
  return(sol) #stored trajectories


def RK4(x0,h,cc,H):
  """
  Runge-Kutta method of order 4

  Parameters
  ----------
  x0 : torch.Tensor
    solution at time t0

  f : function handle
    map f from the manifold M to the Lie algebra of the group acting on M

  h : float
    time step

  cc : int
    M-1

  H : neural network class
    Hamiltonian function

  Returns
  -------
  sol : solution at time t1=t0+h
 
  """ 
  sol = torch.zeros([len(x0[:,0]),len(x0[0,:]),cc]).to(device)
  z  = x0.clone().requires_grad_()
  for i in range(cc):
    k1 = predictedVF(z,H) 
    k2 = predictedVF(z + 0.5 * h * k1, H)
    k3 = predictedVF(z + 0.5 * h * k2, H)
    k4 = predictedVF(z + h * k3, H)
    z = z + h/6 * (k1 + 2 * k2 + 2 * k3 + k4)
    sol[:,:,i] = z.to(device)
  return(sol) #stored trajectories


