import math
import torch
#-----------------------------------------------------------------------------------------------#
class Simulation(torch.nn.Module):
    def __init__(self,dt,N,L,Dt,v,device):
        super(Simulation,self).__init__()
        self.N = N
        self.L = L
        self.v = v
        self.dt = dt
        self.Dt = Dt
        self.Dr = 3*Dt
        self.π = math.pi
        self.epsilon = 1.0
        self.device = device
        self.corrector = 2**(1/3)*torch.eye(N).to(device)

    def PBC(self, r, θ):
        r = r % self.L
        θ = θ % (2*self.π)
        return r, θ

    def MIC(self, r):
        return (r+self.L/2)%self.L - self.L/2

    def Input(self, X):
        Y = X.detach() % self.L
        x = Y[...,:self.N]; y = Y[...,self.N:]
        return torch.cat([self.MIC(x-x.mean(-1,keepdim=True)), self.MIC(y-y.mean(-1,keepdim=True))],-1)

    def diffusion_matrix(self):
        DM_r = math.sqrt(2*self.Dt)*torch.eye(2*self.N)
        DM_θ = math.sqrt(2*self.Dr)*torch.eye(1*self.N)
        return DM_r.to(self.device), DM_θ.to(self.device) 

    def drift(self, θ, dX1, dX2):
        l_ij = dX1**2 + dX2**2 + self.corrector.expand([*θ.shape,self.N])
        dU = torch.relu(l_ij**(-13/2)-1/2*l_ij**(-7/2))
        Fx = 48*self.epsilon*(dU*dX1/torch.sqrt(l_ij)).sum(-1) + self.v*torch.cos(θ)
        Fy = 48*self.epsilon*(dU*dX2/torch.sqrt(l_ij)).sum(-1) + self.v*torch.sin(θ)
        return torch.cat([Fx, Fy],dim=-1)

    def F(self, r, θ):
        r, θ = self.PBC(r,θ) # periodic boundary condition
        dX1 = r[...,:self.N].unsqueeze(-1).expand([*θ.shape,self.N])
        dX2 = r[...,self.N:].unsqueeze(-1).expand([*θ.shape,self.N])
        dX1 = self.MIC(dX1-dX1.transpose(-1,-2))
        dX2 = self.MIC(dX2-dX2.transpose(-1,-2))
        return self.drift(θ,dX1,dX2)
#-----------------------------------------------------------------------------------------------#
class Deepset(torch.nn.Module):
    def __init__(self,dim_in,dim_out):
        super(Deepset,self).__init__()
        gain = 5/3
        # parameter
        self.A = torch.nn.Parameter(torch.empty(dim_in,dim_out,1,1))
        self.B = torch.nn.Parameter(torch.empty(dim_in,dim_out,1,1))
        torch.nn.init.xavier_uniform_(self.A.squeeze(-1).squeeze(-1),gain=gain)
        torch.nn.init.xavier_uniform_(self.B.squeeze(-1).squeeze(-1),gain=gain)
        
class Linear(torch.nn.Module):
    def __init__(self,N,dim_in,dim_out,encoder=False):
        super(Linear,self).__init__()
        self.encoder = encoder
        self.Wd = torch.nn.Parameter(torch.empty(dim_in,dim_out)); torch.nn.init.orthogonal_(self.Wd)
        self.Wn = torch.nn.Parameter(torch.empty(N,N)); torch.nn.init.orthogonal_(self.Wn)
        if encoder: self.Wbs = torch.nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty(size=(1,N*dim_out))))

class EQI(torch.nn.Module):
    def __init__(self,N,n_hid,n_block,device):
        super(EQI,self).__init__()
        self.N = N
        self.n_hid = n_hid
        self.device = device
        self.n_block = n_block
        self.activation = torch.nn.Tanh()
        # block matrix
        self.diag = torch.eye(N)
        self.offd = torch.eye(N)-1/N*torch.ones(size=(N,N))
        self.bias = torch.ones(size=(1,N))
        # Neural network
        self.Λ = torch.nn.Parameter(torch.ones(size=(n_hid,1)))
        ## Equivariant model
        equiv_model = []
        equiv_model.append(Deepset(2,n_hid))
        for _ in range(n_block): equiv_model.append(Deepset(n_hid,n_hid))
        equiv_model.append(Deepset(n_hid,2))
        self.Equiv = torch.nn.ModuleList(equiv_model)
        ## Nonequivariant model
        nonequiv_model = []
        nonequiv_model.append(Linear(N,2,n_hid,encoder=True))
        for _ in range(n_block): nonequiv_model.append(Linear(N,n_hid,n_hid))
        nonequiv_model.append(Linear(N,n_hid,2))
        self.Noneq = torch.nn.ModuleList(nonequiv_model)
        ## residual strength parameter
        self.gammas = torch.nn.Parameter(torch.ones(n_block+2,))
        self.lambdas = torch.nn.Parameter(torch.zeros(n_block+2,))

    def ME(self,module,device):
        dim_in, dim_out = module.A.shape[0], module.A.shape[1]
        return (module.A.to(device)*(self.diag.to(device)) +\
                module.B.to(device)*(self.offd.to(device))).transpose(1,2).contiguous().view(self.N*dim_in,self.N*dim_out)

    def MB(self,device):
        return (self.Λ.to(device)*(self.bias.to(device))).view(1,-1)

    def MN(self,module,device):
        W = torch.kron(module.Wn.to(device), module.Wd.to(device))
        if module.encoder: W = torch.cat([W,module.Wbs.to(device)],dim=0) # W.shape = (N*dim_in+1,N*dim_out)
        return W.to(device)

    def forward(self,We,Wb,Wn,X,bias):
        X = self.activation(X@(We[0]) + bias@Wb) + self.gammas[0]*self.activation(torch.cat([X,bias],dim=-1)@(Wn[0]))
        for n in range(self.n_block): X = self.activation(X@(We[n+1])) + self.gammas[n+1]*self.activation(X@(Wn[n+1]))
        X = X@(We[-1]) + self.gammas[-1]*(X@(Wn[-1]))
        return X
    
    @property
    def primal_params(self):
        return [p for n,p in self.named_parameters() if not n.endswith("lambdas")]

    @property
    def dual_params(self):
        return [self.lambdas]