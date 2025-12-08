import os
import re
import sys
import math
import copy
import torch
import pickle
import argparse
import warnings
import Model as MM
import numpy as np
import torch.optim as optim
from tqdm import trange
sys.setrecursionlimit(10**9)
warnings.filterwarnings(action="ignore")
torch.set_num_threads(1)
#-----------------------------------------------------------------------------------------------#
# Parser Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--N",type=int,default=21)
parser.add_argument("--hid",type=int,default=25)
parser.add_argument("--block",type=int,default=1)
parser.add_argument("--lmbda",type=float)
parser.add_argument("--lrp",type=float,default=1e-2)
parser.add_argument("--lrd",type=float,default=1e-4)
parser.add_argument("--epoch",type=int,default=50000)
parser.add_argument("--device",type=str,default="cpu")
#------------------------------------------------------------------------------------------------------------------------------------#
args = parser.parse_args()
η_p = args.lrp; η_d = args.lrd
epochs = args.epoch; n_blocks = args.block
lmbda = args.lmbda; device = args.device; n_hid = args.hid; N = args.N
#-----------------------------------------------------------------------------------------------#
def f2s(val:float) -> str:
    s = format(val, ".0e")
    s = re.sub(r'e([-+]?)0*(\d+)', r'e\1\2', s)
    return s

η_p_str = f2s(η_p); η_d_str = f2s(η_d)
#------------------------------------------------------------------------------------------------------------------------------------#
if not os.path.exists(os.getcwd()+"/data/"): os.mkdir(os.getcwd()+"/data/")
if not os.path.exists(os.getcwd()+"/result/"): os.mkdir(os.getcwd()+"/result/")
#------------------------------------------------------------------------------------------------------------------------------------#
# Parameters
n_models = 1000  # 학습 마지막에, 몇 개의 model을 사용하여 estimation을 할 것인가?
mid_epoch = 1000 # 몇 epoch마다 긴 경로를 만들어서 equiv measure, 입자별로 work의 값을 잴 것인가?
dt = 0.005; tstep = 1000 # Simulation timelength
ensemble = 100; L = 1.0; μ = 1.0; τ = 1.0; K = 10.0; Da = 2.0; kbT = 0.5
#------------------------------------------------------------------------------------------------------------------------------------#
# Periodic boundary condition
PBC_r = torch.zeros(size=(ensemble,N))
PBC_r[:,:+1] = -L*torch.ones(size=(ensemble,1))
PBC_r[:,-1:] = +L*torch.ones(size=(ensemble,1))
PBC = torch.cat([PBC_r, torch.zeros(size=(ensemble,N))],-1).to(device)
#------------------------------------------------------------------------------------------------------------------------------------#
# Neural networks & Simulation
Sim = MM.Simulation(N,μ,τ,K,Da,kbT)
NN = MM.EQI(N,n_hid,n_blocks,device); NN.to(device)
FM = Sim.FM().to(device); DM = Sim.DM().to(device)
# Optimizer
Optimizer_primal = optim.Adam(NN.primal_params,lr=η_p)
Optimizer_dual = optim.Adam(NN.dual_params,lr=η_d,maximize=True)
#------------------------------------------------------------------------------------------------------------------------------------#
def Shift(X):
    Y = X.detach()
    r = Y[...,:N]; a = Y[...,N:]
    domain = r//L; wrap = r%L
    shifted = domain - torch.max(domain,dim=-1,keepdim=True).values
    return torch.cat([L*shifted+wrap,a],-1)
#------------------------------------------------------------------------------------------------------------------------------------#
def roll(X,n):
    r = X[...,:N]; a = X[...,N:]
    return torch.cat([torch.roll(r,n,dims=-1),torch.roll(a,n,dims=-1)],-1)
#------------------------------------------------------------------------------------------------------------------------------------#
# Result를 저장하는 list
ψ_T = []; P_T = []; M_T = {}; E_T = {}
#------------------------------------------------------------------------------------------------------------------------------------#
# Initial conditions
X_i = torch.cat([L*torch.rand(size=(ensemble,N)),
                 1*torch.zeros(size=(ensemble,N))],-1).to(device)
# Randomize the initial condition
with torch.no_grad():
    We = [NN.ME(NN.Equiv[n],device) for n in range(n_blocks+2)]
    Wn = [NN.MN(NN.Noneq[n],device) for n in range(n_blocks+2)]
    for t in range(tstep): X_i = X_i + dt*(X_i @ FM + PBC + NN(We,Wn,Shift(X_i))) + torch.normal(mean=0.,std=math.sqrt(dt),size=(ensemble,2*N)).to(device) @ DM
#  ------------------------------------------------------------------------------------------------------------------------------------#
# Store 0 epoch model
M_T["State_0"] = copy.deepcopy(X_i)
M_T["Model_0"] = copy.deepcopy(NN.state_dict())
P_T.append(copy.deepcopy(NN.gammas.clone().detach().cpu().numpy()))
# Train neural network
for epoch in trange(epochs,desc="TRAIN λ = {0:.2f}".format(lmbda)):
    # Make weight matrix
    We = [NN.ME(NN.Equiv[n],device) for n in range(n_blocks+2)]
    Wn = [NN.MN(NN.Noneq[n],device) for n in range(n_blocks+2)]
    # generate trajectory
    fn = []; traj = [X_i]; noise = []
    for _ in range(tstep):
        δu = NN(We,Wn,Shift(X_i))
        dW = torch.normal(mean=0.,std=math.sqrt(dt),size=(ensemble,2*N)).to(device) @ DM
        X_i = X_i + dt*(X_i @ FM + PBC + δu) + dW
        # append
        fn.append(δu)
        traj.append(X_i)
        noise.append(dW)
    FN = torch.stack(fn,dim=0)
    tj = torch.stack(traj,dim=0)
    Noise = torch.stack(noise,dim=0)
    # compute the observable
    dX = (tj[1:]-tj[:-1]-Noise)[...,:N]
    am = (tj[1:]+tj[:-1]-Noise)[...,N:]/2
    Work = 1/μ*(am*dX).sum((0,-1)) / (dt*tstep)
    Girs = dt/2*(FN[...,:N]**2/(2*μ*kbT) + FN[...,N:]**2/(2*Da/(τ**2))).sum((0,-1)) / (dt*tstep)
    # update parameters
    Loss = Girs-lmbda*Work
    total = Loss.mean() + torch.sum(NN.gammas*NN.lambdas)
    Optimizer_dual.zero_grad()
    Optimizer_primal.zero_grad()
    total.backward()
    Optimizer_dual.step()
    Optimizer_primal.step()
    NN.zero_grad()
    # update initial condition
    X_i = copy.deepcopy(tj[-1].detach())
    # store data
    ψ_T.append(-Loss.detach().cpu().numpy().mean())
    P_T.append(copy.deepcopy(NN.gammas.clone().detach().cpu().numpy()))
    if (epoch+1) % mid_epoch == 0:
        M_T["State_{0}".format(epoch+1)] = copy.deepcopy(X_i).cpu()
        M_T["Model_{0}".format(epoch+1)] = copy.deepcopy(NN.cpu().state_dict())
    if epoch > (epochs-(n_models+1)):
        E_T["State_{0}".format(epoch-(epochs-(n_models+1)))] = copy.deepcopy(X_i).cpu()
        E_T["Model_{0}".format(epoch-(epochs-(n_models+1)))] = copy.deepcopy(NN.cpu().state_dict())
    if np.isnan(ψ_T[-1]): raise Exception("Diverge")
    NN.to(device)
#------------------------------------------------------------------------------------------------------------------------------------#
# Result
with open(os.getcwd()+"/data/Equivariant_SCGF_N={0}_H={1}_B={2}_λ={3}_lrp={4}_lrd={5}.pkl".format(N,n_hid,n_blocks,lmbda,η_p_str,η_d_str),"wb") as fa: pickle.dump(ψ_T,fa)
with open(os.getcwd()+"/data/Equivariant_PARM_N={0}_H={1}_B={2}_λ={3}_lrp={4}_lrd={5}.pkl".format(N,n_hid,n_blocks,lmbda,η_p_str,η_d_str),"wb") as fb: pickle.dump(P_T,fb)
torch.save(E_T,os.getcwd()+"/data/Equivariant_ET_N={0}_H={1}_B={2}_λ={3}_lrp={4}_lrd={5}.pth".format(N,n_hid,n_blocks,lmbda,η_p_str,η_d_str))
torch.save(M_T,os.getcwd()+"/data/Equivariant_MT_N={0}_H={1}_B={2}_λ={3}_lrp={4}_lrd={5}.pth".format(N,n_hid,n_blocks,lmbda,η_p_str,η_d_str))
#------------------------------------------------------------------------------------------------------------------------------------#
# tstep = int(1e4)
# equi_ensemble = 100; repeat = int(equi_ensemble/ensemble)
# # Periodic boundary condition
# PBC_r = torch.zeros(size=(equi_ensemble,N))
# PBC_r[:,:+1] = -L*torch.ones(size=(equi_ensemble,1))
# PBC_r[:,-1:] = +L*torch.ones(size=(equi_ensemble,1))
# PBC = torch.cat([PBC_r, torch.zeros(size=(equi_ensemble,N))],-1).to(device)
# # estimate!
# WORK = []; SCGF = []; FORK = []
# with torch.no_grad():
#     # Checkpoint = torch.load(os.getcwd()+"/data/Equivariant_ET_N={0}_H={1}_B={2}_λ={3}_lrp={4}_lrd={5}.pth".format(N,n_hid,n_blocks,lmbda,η_p_str,η_d_str),map_location=device)
#     for n in trange(n_models,desc='Estimate SCGF'):
#         # Load model & state
#         NN = MM.EQI(N,n_hid,n_blocks,device).to(device)
#         # NN.load_state_dict(Checkpoint['Model_{0}'.format(n+1)])
#         # X_i = Checkpoint['State_{0}'.format(n+1)].repeat(repeat,1).to(device)
#         NN.load_state_dict(E_T['Model_{0}'.format(n+1)])
#         X_i = E_T['State_{0}'.format(n+1)].repeat(repeat,1).to(device)
#         # Make weight
#         fn = []; traj = [X_i]; noise = []
#         We = [NN.ME(NN.Equiv[n],device) for n in range(n_blocks+2)]
#         Wn = [NN.MN(NN.Noneq[n],device) for n in range(n_blocks+2)]
#         # generate trajectory
#         for t in range(tstep):
#             δu = NN(We,Wn,Shift(X_i))
#             dW = torch.normal(mean=0.,std=math.sqrt(dt),size=(equi_ensemble,2*N)).to(device) @ DM
#             X_i = X_i + dt*(X_i @ FM + PBC + δu) + dW
#             # append
#             fn.append(δu)
#             traj.append(X_i)
#             noise.append(dW)
#         FN = torch.stack(fn,dim=0)
#         tj = torch.stack(traj,dim=0)
#         Noise = torch.stack(noise,dim=0)
#         # compute the observable
#         dX = (tj[1:]-tj[:-1]-Noise)[...,:N]
#         am = (tj[1:]+tj[:-1]-Noise)[...,N:]/2
#         Work = 1/μ*(am*dX).sum((0,-1)).mean() / (dt*tstep)
#         Girs = dt/2*(FN[...,:N]**2/(2*μ*kbT) + FN[...,N:]**2/(2*Da/(τ**2))).sum((0,-1)).mean() / (dt*tstep)
#         Scgf = Girs - lmbda*Work
#         # store data
#         WORK.append(Work.cpu().item())
#         SCGF.append(Scgf.cpu().item())
# # Convert to numpy array
# SCGF = np.array(SCGF).mean()
# WORK = np.array(WORK).mean()
# # save the result & clear memory
# E_T = {}
# with open(os.getcwd()+"/result/Equivariant_LSCGF_N={0}_H={1}_B={2}_λ={3}_lrp={4}_lrd={5}.pkl".format(N,n_hid,n_blocks,lmbda,η_p_str,η_d_str),"wb") as fc: pickle.dump(SCGF,fc)
# with open(os.getcwd()+"/result/Equivariant_LWORK_N={0}_H={1}_B={2}_λ={3}_lrp={4}_lrd={5}.pkl".format(N,n_hid,n_blocks,lmbda,η_p_str,η_d_str),"wb") as fd: pickle.dump(WORK,fd)
# #------------------------------------------------------------------------------------------------------------------------------------#
# # Compute the equivariance measure dynamics
# error_ensemble = 100; repeat = int(error_ensemble/ensemble)
# # Periodic boundary condition
# PBC_r = torch.zeros(size=(error_ensemble,N))
# PBC_r[:,:+1] = -L*torch.ones(size=(error_ensemble,1))
# PBC_r[:,-1:] = +L*torch.ones(size=(error_ensemble,1))
# PBC = torch.cat([PBC_r, torch.zeros(size=(error_ensemble,N))],-1).to(device)
# # parameter
# tstep = int(1e4)
# Epochs = [mid_epoch*e for e in range(epochs//mid_epoch+1)]
# # Compute error!
# with torch.no_grad():
#     Errors = []
#     for e in trange(len(Epochs),desc="Error"):
#         # Load model & state
#         # Checkpoint = torch.load(os.getcwd()+"/data/Equivariant_MT_N={0}_H={1}_B={2}_λ={3}_lrp={4}_lrd={5}.pth".format(N,n_hid,n_blocks,lmbda,η_p_str,η_d_str),map_location=device)
#         NN = MM.EQI(N,n_hid,n_blocks,device).to(device)
#         # NN.load_state_dict(Checkpoint["Model_{0}".format(Epochs[e])])
#         # X_i = Checkpoint["State_{0}".format(Epochs[e])].repeat(repeat,1).to(device)   
#         NN.load_state_dict(M_T["Model_{0}".format(Epochs[e])])
#         X_i = M_T["State_{0}".format(Epochs[e])].repeat(repeat,1).to(device)
#         # # Randomize the initial conditions
#         TJs = [X_i]
#         We = [NN.ME(NN.Equiv[n],device) for n in range(n_blocks+2)]
#         Wn = [NN.MN(NN.Noneq[n],device) for n in range(n_blocks+2)]
#         for t in range(tstep):
#             X_i = X_i + dt*(X_i @ FM + NN(We,Wn,Shift(X_i)) + PBC) + torch.normal(mean=0.,std=math.sqrt(dt),size=(error_ensemble,2*N)).to(device) @ DM
#             TJs.append(X_i)
#         TJs = Shift(torch.stack(TJs[:-1],dim=0)).to(device)
#         # compute force from the long trajectory
#         Error = torch.tensor([0.0, 0.0])
#         for n in range(N-1):
#             TJ = TJs.clone()
#             FN_p = NN(We,Wn,roll(TJ,n+1)) # input  permute
#             p_FN = roll(NN(We,Wn,TJ),n+1) # output permute
#             # Mean squared error
#             Err_r = torch.mean((FN_p[...,:N]-p_FN[...,:N])**2).item()
#             Err_a = torch.mean((FN_p[...,N:]-p_FN[...,N:])**2).item()
#             Error = Error + torch.tensor([Err_r, Err_a])
#         Errors.append(copy.deepcopy(Error.numpy()/(N-1)))
# with open(os.getcwd()+"/result/Equivariant_ERROR_N={0}_H={1}_B={2}_λ={3}_lrp={4}_lrd={5}.pkl".format(N,n_hid,n_blocks,lmbda,η_p_str,η_d_str),"wb") as ff: pickle.dump(Errors,ff)
# #------------------------------------------------------------------------------------------------------------------------------------#