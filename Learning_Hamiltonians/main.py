import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from autograd import grad
import autograd.numpy as np
import scipy.integrate

from trajectories import *

__all__ = ['Hamiltonian', 'predicted']

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Paramaters to generate the trajectories

e3 = np.array([0,0,1]) # third axis of the intertial frame

T = 0.1 # final time
M = 5 # number of time steps
time = np.linspace(0,T,M) # M evenly spaced times over the interval [0, T]
dt = time[1] - time[0]
s = 3 # dimensionality of the problem (dynamics with ambient space R^2s)
N = 500 # number of initial conditions (i.e. number of training trajectories)

nop = 2 #Number of pundulums


### Generation of the training and test trajectories

# initialization of the trajectories
trajectories = np.zeros([N,2*s*nop,M])

# initial conditions
q = np.zeros(int(3*nop))
p = np.zeros(int(3*nop))
for i in range(N):
  for j in range(nop):
    nrand = np.random.randn(3)
    q = nrand/np.linalg.norm(nrand,2)
    trajectories[i,3*2*j:3*(2*j+1),0] = q 
    supp = np.random.rand(3)
    trajectories[i,3*(2*j+1):3*(2*j+2),0] = np.cross(q,supp)


# Generation of the dataset with RK45 (default)
for j in range(N):
  trajectories[j,:,:] = scipy.integrate.solve_ivp(dynamics,[0,T],trajectories[j,:,0],method='RK45',t_eval = time,rtol=1e-3,atol=1e-5).y


# Visualization of trajectories

u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)

a = 1 * np.outer(np.cos(u), np.sin(v))
b = 1 * np.outer(np.sin(u), np.sin(v))
c = 1 * np.outer(np.ones(np.size(u)), np.cos(v))

for num in range(nop):
  fig = plt.figure(figsize=(12,12), dpi=80)
  ax = plt.axes(projection='3d')
  ax.plot_surface(a, b, c,  rstride=4, cstride=4, color='g', linewidth=0, alpha=0.09)
  for i in range(N):
    ax.plot3D(trajectories[i,6*num,:],trajectories[i,6*num+1,:],trajectories[i,6*num+2,:],'k-')
  plt.title(f"Trajectory segments pendulum {num+1}",fontsize=20)
  plt.show()


# Building the neural network

from nn_functions import *

X = trajectories[:,:,0] # trajectories in the phase space at time t=0 (intial comnditions)
Y = trajectories[:,:,1:M+1] # trajectory solutions at M-1 evenly spaced times over the interval [dt, T] with dt = T/(M-1)

# definition of the dataset with pytorch

from torch.utils.data import Dataset, DataLoader
class dataset(Dataset):
  def __init__(self,x,y):
    self.x = torch.from_numpy(x.astype(np.float32))
    self.y = torch.from_numpy(y.astype(np.float32))
    self.length = self.x.shape[0]
 
  def __getitem__(self,idx):
    return self.x[idx],self.y[idx]
  def __len__(self):
    return self.length

trainset = dataset(X,Y)



batch_size = 80
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)

K = 100 # number of neurons of the hidden layers for the potential energy
nlayers = 2 # number of layers of the NN

id1=np.arange(0,s*nop,s)
id2=np.arange(1,s*nop,s)
id3=np.arange(2,s*nop,s)


# Neural network

class Hamiltonian(nn.Module): # the neural network module contains layers and encapsulates paramenters
    def __init__(self, ):
        super(Hamiltonian, self).__init__()
        
        self.IV = nn.Linear(s*nop,K) # linear layer (s*nop-dimnesional input, K-dimensional output)
        self.nl = nn.Tanh() # nonlinear activation function
        self.OV = nn.Linear(K,1,bias=False) # linear output layer (K-dimnesional input, 1-dimensional output)

        ll = []
        ll.append(self.IV)
        ll.append(self.nl)
        for i in range(nlayers):
          ll.append(nn.Linear(K,K))
          ll.append(self.nl)
        ll.append(self.OV)

        # the sequential module connects the modules in ll in a cascading way, 
        # chaining outputs to inputs sequentially for each subsequent module 
        # and returning the output of the ladt module
        self.seq = nn.Sequential(*ll)

        # parameters in the neural network used to define the mass matrix in the kinetic energy
        self.Gamma = torch.nn.Parameter(data=torch.rand(nop,nop))
        self.B = torch.nn.Parameter(data=torch.rand(nop))
        self.func = nn.ReLU()

    # Definition of the mass matrix
    def MassMat(self,X):
        """
        Mass matrix defining the kinetic energy quadratic function

        Parameters
        ----------
        X: torch.Tensor
           training trajectory points in input, with shape [batch size, nop*2s]
        Returns
        -------
        row : torch.Tensor
              Mass matrix to be learned by the neural network,  with shape [batch size, nop*s, nop*s]

        """
        nop = int(X.size(dim=1)/6)
        q = X[:,:3]
        for j in range(1,nop):
          q = torch.cat([q, X[:,3*2*j:3*(2*j+1)]], axis=1)
        M = torch.transpose(self.Gamma,0,1)@self.Gamma + torch.diag(self.B)**2
        I = torch.eye(3).unsqueeze(0).repeat(len(q),1,1).to(device)
        R = M[0,0] * I
        for j in range(1,nop):
          R = torch.cat((R,M[0,j]*(I-torch.einsum('ij,ik->ijk',q[:,:3],q[:,:3]))),axis=2)
        for i in range(1,nop):
          for j in range(nop):
            if j == i:
              R = torch.cat((R,M[i,i]*I),axis=2)
            else:
              R = torch.cat((R,M[i,j]*(I-torch.einsum('ij,ik->ijk',q[:,3*i:3*i+3],q[:,3*i:3*i+3]))),axis=2)
        n = nop * s
        row=R[:,0:3,0:n]
        for i in range(1,nop):
          row = torch.cat((row,R[:,0:3,n*i:n*(i+1)]),axis=1)
        return row

    # Modelling of the Kinetic energy as a bilinear form  
    def Kinetic(self, X):
        """
        Kinetic energy in the Hamiltonian function

        Parameters
        ----------
        X: torch.Tensor
           training trajectory points in input, with shape [batch size, nop*2s]

        Returns
        -------
        row : torch.Tensor
              Kinetic energy, with shape [batch size, 1]
              
        """ 
        nop = int(X.size(dim=1)/6)
        id = torch.eye(nop,nop)
        ref = torch.ones(3,3)
        R = torch.kron(id, ref).to(device)
        U = torch.ones(3*nop,3*nop).to(device)
        p = X[:,3:6]
        for j in range(1,nop):
          p = torch.cat([p, X[:,3*(2*j+1):3*(2*j+2)]], axis=1)
        MM = self.MassMat(X) 
        k = (0.5 * torch.einsum('ij,ij->i',p,torch.linalg.solve(MM,p))).unsqueeze(1) # q dependent component
        return k
    
    # Modelling of the potential energy as a feed-forward neural network
    def Potential(self, X):
        """
        Potential energy in the Hamiltonian function

        Parameters
        ----------
        X: torch.Tensor
           training trajectory points in input, with shape [batch size, nop*2s] 

        Returns
        -------
        row : torch.Tensor
              Potential energy, with shape [batch size, 1]
              
        """ 
        nop = int(X.size(dim=1)/6)
        q = X[:,:3]
        for j in range(1,nop):
          q = torch.cat([q, X[:,3*2*j:3*(2*j+1)]], axis=1)

        v = self.seq(q)
        return v

    # Sum of the contributions to get the approximated Hamiltonian energy
    def forward(self, X):
        """
        Forward function that receives a tensor containing the input (trajectory points in the phase space) 
        and returns a tensor containing a scalar output (Hamiltonian)).

        Parameters
        ----------
        X: torch.Tensor
           training trajectory points in input, with shape [batch size, nop*2s]

        Returns
        -------
        o : torch.Tensor
            Value of the Hamiltonian, with shape [batch size, 1]

        """ 
        o = self.Potential(X) + self.Kinetic(X)
        return o


Ham = Hamiltonian()
Ham.to(device)

import torch.optim as optim

criterion = nn.MSELoss() # defining the Loss function as a mean squared error function
optimizer = torch.optim.Adam(Ham.parameters(),lr=0.01) # Adam algorithm as optimizer
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)


# Training the neural network

# Choice of the integrator
integrator = CF4NN
isLieGroupMethod = True # is integrator a Lie group method?

checkpoint = 20

for epoch in range(3):

    running_loss = 0.0

    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        inputs.requires_grad_()
        optimizer.zero_grad()

        if isLieGroupMethod:
          predicted = integrator(inputs,fManiAlgebraNN,dt,M-1,Ham)
        else:
          predicted = integrator(inputs,dt,M-1,Ham)

        loss = criterion(predicted, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if i % checkpoint == 0:  
            print('[%d, %5d] loss: %.10f' %
                  (epoch + 1, i + 1, running_loss / checkpoint))
            running_loss = 0.0

    scheduler.step()
    if epoch%10 == 0 and epoch>0:
      lr=optimizer.param_groups[0]["lr"]
      print(f"\n\nLR: {lr}, EPOCH: {epoch}\n\n")

print('Finished Training')


# Evaluating the approximation

Ham.eval(); # to pass in evaluation mode

def predicted(t,z):
  """
  Vector field predicted by using the evaluated Hamiltonian, t be used in scipy.integrate.solve_ivp
  to get trajectory segments with the evaluated Hamitlonian

  Parameters
  ----------
  t : time
      standard input to use the predicted function in scipi.integrate.solve_ivp

  HH : torch.Tensor
       trajectory point

  Returns
  -------
  vec : torch.Tensor
        vector field (Hamlton equations)

  """
  z = torch.from_numpy(z.astype(np.float32)).requires_grad_()
  z = z.to(device)
  nop = int(len(z)/6)
  
  HamVal = torch.sum(Ham(z.view(1,-1))).to(device)
  dH = torch.autograd.grad(HamVal, z, create_graph=True)[0].detach().cpu().numpy()

  vec = np.zeros(len(z))
  z = z.detach().cpu().numpy()
  q = z[0:3]
  p = z[3:6]
  dHq = dH[:3]
  dHp = dH[3:6]
  for i in range(1,nop):
    q = np.concatenate((q,z[3*(2*i):3*(2*i+1)]))
    p = np.concatenate((p,z[3*(2*i+1):3*(2*i+2)]))
    dHq = np.concatenate((dHq,dH[3*(2*i):3*(2*i+1)]))
    dHp = np.concatenate((dHp,dH[3*(2*i+1):3*(2*i+2)]))

  I = np.eye(3)

  for i in range(nop):
    dqdt = (I-np.outer(q[3*i:3*(i+1)],q[3*i:3*(i+1)]))@dHp[3*i:3*(i+1)]
    part1 = -(I-np.outer(q[3*i:3*(i+1)],q[3*i:3*(i+1)]))@dHq[3*i:3*(i+1)]
    part2 = np.cross(dHp[3*i:3*(i+1)],np.cross(p[3*i:3*(i+1)],q[3*i:3*(i+1)]))
    dpdt = part1 + part2
    vec[6*i:6*(i+1)] = np.concatenate([dqdt,dpdt])
  return vec
  
# Generate test initial conditions

MM = 15
TT = 1
Ntest = 100

predictedTraj = np.zeros([Ntest,2*s*nop,MM])
realTraj = np.zeros([Ntest,2*s*nop,MM])
timeEv = np.linspace(0,TT,MM)

q = np.zeros(int(3*nop))
p = np.zeros(int(3*nop))

for i in range(Ntest):
  for j in range(nop):
    nrand = np.random.randn(3)
    q = nrand/np.linalg.norm(nrand,2)
    predictedTraj[i,3*2*j:3*(2*j+1),0] = q #position # indici: i-esimo punto di dataset, coordinate vettore nello spazio di config, time step
    supp = np.random.rand(3)
    predictedTraj[i,3*(2*j+1):3*(2*j+2),0] = np.cross(q,supp)

realTraj[:,:,0] = predictedTraj[:,:,0]


#Generate the dataset

for j in range(Ntest):
  predictedTraj[j,:,:] = scipy.integrate.solve_ivp(predicted,[0,TT],predictedTraj[j,:,0],method='RK45',t_eval = timeEv,rtol=1e-6,atol=1e-8).y
  realTraj[j,:,:] = scipy.integrate.solve_ivp(dynamics,[0,TT],realTraj[j,:,0],method='RK45',t_eval = timeEv,rtol=1e-6,atol=1e-8).y

u = np.linspace(0, 2*np.pi, 100)
v = np.linspace(0, np.pi, 100)
x = np.outer(np.cos(u), np.sin(v)) # np.outer() -> outer vector product
y = np.outer(np.sin(u), np.sin(v))
z = np.outer(np.ones(np.size(u)), np.cos(v))

def set_axes_equal(ax: plt.Axes):
    """Set 3D plot axes to equal scale.

    Make axes of 3D plot have equal scale so that spheres appear as
    spheres and cubes as cubes.  Required since `ax.axis('equal')`
    and `ax.set_aspect('equal')` don't work on 3D.
    """
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    _set_axes_radius(ax, origin, radius)


def _set_axes_radius(ax, origin, radius):
    x, y, z = origin
    ax.set_xlim3d([x - radius, x + radius])
    ax.set_ylim3d([y - radius, y + radius])
    ax.set_zlim3d([z - radius, z + radius])


def _set_axes_radius(ax, origin, radius):
    x, y, z = origin
    ax.set_xlim3d([x - radius, x + radius])
    ax.set_ylim3d([y - radius, y + radius])
    ax.set_zlim3d([z - radius, z + radius])

for num in range(nop):
  fig = plt.figure(figsize=(20,20))
  ax = plt.axes(projection='3d')
  ax.plot_surface(x, y, z,  rstride=4, cstride=4, color='g', linewidth=0, alpha=0.09)
  
  ax.plot3D(realTraj[0,6*num,:],realTraj[0,6*num+1,:],realTraj[0,6*num+2,:],'k-',label="True Hamiltonian")
  ax.plot3D(predictedTraj[0,6*num,:],predictedTraj[0,6*num+1,:],predictedTraj[0,6*num+2,:],'r--',label="Predicted Hamiltonian")
  for i in range(Ntest):
    ax.plot3D(realTraj[i,6*num,:],realTraj[i,6*num+1,:],realTraj[i,6*num+2,:],'k-')
    ax.plot3D(predictedTraj[i,6*num,:],predictedTraj[i,6*num+1,:],predictedTraj[i,6*num+2,:],'r--')
  plt.title(f"Time evolution of the configuration variable for pendulum {num+1}",fontsize=20)
  plt.legend(fontsize=20,loc='lower right')

  # ax.set_box_aspect([1,1,1]) 
  # ax.set_proj_type('ortho') 
  set_axes_equal(ax)
  plt.savefig(f"pend{num+1}.png")
  plt.tight_layout()
  plt.show()


print("MSE on test trajectories with RK45 as integrator: ",np.mean((predictedTraj-realTraj)**2))
