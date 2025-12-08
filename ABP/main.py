import os
import sys
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
parser.add_argument("--N",type=int,default=40)
parser.add_argument("--hid",type=int,default=50)
parser.add_argument("--block",type=int,default=4)
parser.add_argument("--lmbda",type=float)
parser.add_argument("--dt",type=float,default=0.0001)
parser.add_argument("--tstep",type=int,default=1000)
parser.add_argument("--lrp",type=float,default=1e-3)
parser.add_argument("--lrd",type=float,default=1e-6)
parser.add_argument("--epoch",type=int,default=25000)
parser.add_argument("--device",type=str,default="cpu")
torch.set_num_threads(1)
#------------------------------------------------------------------------------------------------------------------------------------#
args = parser.parse_args()
η_p = args.lrp; η_d = args.lrd
lmbda = args.lmbda; device = args.device; n_hid = args.hid; N = args.N; epochs = args.epoch; n_blocks = args.block; tstep = args.tstep; dt = args.dt
#------------------------------------------------------------------------------------------------------------------------------------#
if not os.path.exists("data/"): os.mkdir("data/")
if not os.path.exists("result/"): os.mkdir("result/")
#------------------------------------------------------------------------------------------------------------------------------------#
# Parameters
mid_epoch = 1000 # 몇 epoch마다 긴 경로를 만들어서 equiv measure를 잴 것인가?
mid_model = epochs // mid_epoch # 결국 총 몇개의 모델이 사용되는가?
ensemble = 50 # 이 값의 2배가 실제 학습에 사용됨
Dt = 1.0; Dr = 3*Dt; v = 10; π = np.pi
ρ  = 0.1; L = np.sqrt(N/ρ)+0.00001
#------------------------------------------------------------------------------------------------------------------------------------#
# Neural networks & Simulation
SM = MM.Simulation(dt,N,L,Dt,v,device).to(device)
NN = MM.EQI(N,n_hid,n_blocks,device).to(device)
# Diffusion matrix
DM_r, DM_θ = SM.diffusion_matrix()
# Optimizer
Optimizer_primal = optim.Adam(NN.primal_params,lr=η_p)
Optimizer_dual = optim.Adam(NN.dual_params,lr=η_d,maximize=True)
#------------------------------------------------------------------------------------------------------------------------------------#
# Initial condition
lmbda0 = lmbda
lmbda1 = -0.05
r_i = (L*torch.rand(size=(2*ensemble,2*N)).to(device)) % L
θ_i = (2*π*torch.rand(size=(2*ensemble,N)).to(device)) % (2*π)
bias = torch.cat([lmbda0*torch.ones(size=(ensemble,1)),
                  lmbda1*torch.ones(size=(ensemble,1))],dim=0).to(device)
# Randomize the initial condition
with torch.no_grad():
    Wb = NN.MB(device)
    We = [NN.ME(NN.Equiv[n],device) for n in range(n_blocks+2)]
    Wn = [NN.MN(NN.Noneq[n],device) for n in range(n_blocks+2)]
    # run simulation
    for _ in range(tstep):
        θ_t = θ_i + torch.normal(mean=0.,std=np.sqrt(dt),size=(2*ensemble,1*N)).to(device) @ DM_θ
        r_t = r_i + dt*(SM.F(r_i,θ_i) + NN(We,Wb,Wn,SM.Input(r_i),bias)) + torch.normal(mean=0.,std=np.sqrt(dt),size=(2*ensemble,2*N)).to(device) @ DM_r
        # Update
        θ_i = θ_t; r_i = r_t
    r_i = r_i % L
    θ_i = θ_i % (2*π)
#------------------------------------------------------------------------------------------------------------------------------------#
def Replica(tjr,FN,F0,r_i,θ_i):
    swaps = ensemble
    with torch.no_grad():
        swap = [np.random.choice(ensemble, swaps, replace=False),
                np.random.choice(ensemble, swaps, replace=False)+ensemble]
        M_T0 = torch.sum((SM.MIC(tjr[1:,swap[0]] - tjr[:-1,swap[0]])/dt - (FN+F0)[:,swap[0]])**2, (0,-1))
        M_T1 = torch.sum((SM.MIC(tjr[1:,swap[1]] - tjr[:-1,swap[1]])/dt - (FN+F0)[:,swap[0]])**2, (0,-1))
        M_T = - (M_T0 - M_T1).cpu().detach().numpy()*dt/4
        j = np.argwhere(np.random.rand(swaps) < np.exp(M_T)).T[0]
        # swap initial condition
        θ_i[[*swap[0][j],*swap[1][j]]] = θ_i[[*swap[1][j],*swap[0][j]]]
        r_i[[*swap[0][j],*swap[1][j]]] = r_i[[*swap[1][j],*swap[0][j]]]
    return r_i.detach(), θ_i.detach()
# ------------------------------------------------------------------------------------------------------------------------------------#
# Result를 저장하는 list
ψ_T = []; S_T = []; P_T = [copy.deepcopy(NN.gammas.clone().detach().cpu().numpy())]
NN.to(device)
# ------------------------------------------------------------------------------------------------------------------------------------#
# Train neural network
normalization = (dt*tstep)*N
for epoch in trange(epochs,desc="TRAIN_on λ = {0:.3f}".format(lmbda)):
    with torch.no_grad():
        # Make weight matrix
        Wb = NN.MB(device)
        We = [NN.ME(NN.Equiv[n],device) for n in range(n_blocks+2)]
        Wn = [NN.MN(NN.Noneq[n],device) for n in range(n_blocks+2)]
        traj_r = [r_i]; traj_θ = [θ_i]
        for _ in range(tstep):
            θ_t = θ_i + torch.normal(mean=0.,std=np.sqrt(dt),size=(2*ensemble,1*N)).to(device) @ DM_θ
            r_t = r_i + dt*(NN(We,Wb,Wn,SM.Input(r_i),bias)+SM.F(r_i,θ_i)) + torch.normal(mean=0.,std=np.sqrt(dt),size=(2*ensemble,2*N)).to(device) @ DM_r
            # update
            θ_i = θ_t; r_i = r_t
            traj_r.append(r_i); traj_θ.append(θ_i)
        # concatenate
        tjr = torch.stack(traj_r[:-1],dim=0)
        tjθ = torch.stack(traj_θ,dim=0)
        # Computing loss function
        dX1 = tjr[...,:N].unsqueeze(-1).expand(*tjr.shape[:-1],N,N)
        dX2 = tjr[...,N:].unsqueeze(-1).expand(*tjr.shape[:-1],N,N)
        dX1 = SM.MIC(dX1-dX1.transpose(-1,-2))
        dX2 = SM.MIC(dX2-dX2.transpose(-1,-2))
    # Compute force
    F0 = SM.drift(tjθ[:-1],dX1,dX2)
    FN = NN([NN.ME(NN.Equiv[n],device) for n in range(n_blocks+2)],
             NN.MB(device),
            [NN.MN(NN.Noneq[n],device) for n in range(n_blocks+2)],
             SM.Input(tjr),bias.expand([tstep,2*ensemble,1]))
    # compute the observables
    dX = dt*(F0+FN)
    b = torch.cat([torch.cos(tjθ), torch.sin(tjθ)],dim=-1)
    Girs = torch.sum(FN**2,(0,-1))/Dt/2*dt/normalization
    Entropy = v/Dt*torch.sum((b[1:]+b[:-1])/2*dX,(0,-1))/normalization
    # loss
    Loss = Girs-bias.squeeze()*Entropy*N
    dual = torch.sum(NN.gammas*NN.lambdas)
    total = Loss.mean() + dual
    Optimizer_dual.zero_grad()
    Optimizer_primal.zero_grad()
    total.backward()
    Optimizer_dual.step()
    Optimizer_primal.step()
    # Store data
    ψ_T.append(-Loss.cpu().detach().numpy()/N)
    S_T.append(+Entropy.cpu().detach().numpy())
    P_T.append(copy.deepcopy(NN.gammas.clone().detach().cpu().numpy()))
    # update initial condition
    θ_i = tjθ[-1].detach()
    r_i = traj_r[-1].detach()
    # Replica exchange
    with torch.no_grad():
        if lmbda0 != lmbda1: r_i, θ_i = Replica(torch.stack(traj_r,dim=0),FN,F0,r_i,θ_i)
        if np.isnan(ψ_T[-1].mean()): raise Exception("Diverge")
    NN.to(device)

#  마지막으로 학습된 neural network, state를 저장한다. (phase transition 확인 위함)
with torch.no_grad():
    # Make weight matrix
    Wb = NN.MB(device)
    We = [NN.ME(NN.Equiv[n],device) for n in range(n_blocks+2)]
    Wn = [NN.MN(NN.Noneq[n],device) for n in range(n_blocks+2)]
    traj_r = [r_i]; traj_θ = [θ_i]
    for _ in range(10*tstep):
        θ_t = θ_i + torch.normal(mean=0.,std=np.sqrt(dt),size=(2*ensemble,1*N)).to(device) @ DM_θ
        r_t = r_i + dt*(NN(We,Wb,Wn,SM.Input(r_i),bias)+SM.F(r_i,θ_i)) + torch.normal(mean=0.,std=np.sqrt(dt),size=(2*ensemble,2*N)).to(device) @ DM_r
        # update the condition
        θ_i = θ_t; r_i = r_t
    # 이제 이 경로의 마지막 condition을 저장하자
    State = {"Angle": copy.deepcopy(θ_i).cpu().detach(),
             "Position": copy.deepcopy(r_i).cpu().detach()%L}

# Result
with open("data/Equivariant_SCGF_N={0}_H={1}_B={2}_λ={3}.pkl".format(N,n_hid,n_blocks,lmbda),"wb") as fa: pickle.dump(ψ_T,fa)
with open("data/Equivariant_ENTR_N={0}_H={1}_B={2}_λ={3}.pkl".format(N,n_hid,n_blocks,lmbda),"wb") as fb: pickle.dump(S_T,fb)
with open("data/Equivariant_PARM_N={0}_H={1}_B={2}_λ={3}.pkl".format(N,n_hid,n_blocks,lmbda),"wb") as fc: pickle.dump(P_T,fc)
torch.save(State,"data/Equivariant_MODEL_N={0}_H={1}_B={2}_λ={3}.pt".format(N,n_hid,n_blocks,lmbda))