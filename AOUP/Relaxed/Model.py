import math
import torch
#-----------------------------------------------------------------------------------------------#
class Simulation(torch.nn.Module):
    def __init__(self,N,μ,τ,K,Da,kbT):
        super().__init__()
        self.N = N
        self.K = K
        self.μ = μ
        self.τ = τ
        self.Da = Da
        self.kbT = kbT

    def DM(self):
        matrix = torch.zeros(size=(2*self.N,2*self.N))
        matrix[:self.N,:self.N] = math.sqrt(2*self.μ*self.kbT)*torch.eye(self.N)
        matrix[self.N:,self.N:] = math.sqrt(2*self.Da/(self.τ**2))*torch.eye(self.N)
        return matrix

    def FM(self):
        Fr_r = self.μ*self.K*(-2*torch.diag(torch.ones(self.N),0)    +\
                                 torch.diag(torch.ones(self.N-1),+1) +\
                                 torch.diag(torch.ones(self.N-1),-1))
        Fr_r[0,-1] = self.μ*self.K; Fr_r[-1,0] = self.μ*self.K
        Fr_a = torch.eye(self.N)
        Fa_r = torch.zeros(size=(self.N,self.N))
        Fa_a = -1/self.τ*torch.eye(self.N)
        return torch.cat([torch.cat([Fr_r,Fr_a],dim=0),
                          torch.cat([Fa_r,Fa_a],dim=0)],-1)
#-----------------------------------------------------------------------------------------------#
class Deepset(torch.nn.Module):
    def __init__(self,dim_in,dim_out,output=False):
        super(Deepset,self).__init__()
        gain = 1.0 if output else 5/3
        self.A = torch.nn.Parameter(torch.empty(dim_in,dim_out,1,1))
        self.B = torch.nn.Parameter(torch.empty(dim_in,dim_out,1,1))
        torch.nn.init.xavier_uniform_(self.A.squeeze(-1).squeeze(-1),gain=gain)
        torch.nn.init.xavier_uniform_(self.B.squeeze(-1).squeeze(-1),gain=gain)

class Linear(torch.nn.Module):
    def __init__(self,N,dim_in,dim_out,output=False):
        super(Linear,self).__init__()
        self.Wn = torch.nn.Parameter(torch.empty(N, N))
        self.Wd = torch.nn.Parameter(torch.empty(dim_in, dim_out))
        if output: torch.nn.init.xavier_uniform_(self.Wd,gain=1.0)
        else:
            torch.nn.init.orthogonal_(self.Wd)
            self.Wd.data /= math.sqrt(N)
        torch.nn.init.orthogonal_(self.Wn)

class EQI(torch.nn.Module):
    def __init__(self,N,n_hid,n_block,device):
        super(EQI,self).__init__()
        self.N = N
        self.n_hid = n_hid
        self.n_block = n_block
        self.activation = torch.nn.Tanh()
        # block matrix
        self.diag = torch.eye(N) 
        self.offd = torch.eye(N) - 1/N*torch.ones(size=(N,N))
        # Neural network
        equiv_model = []; Noneq_model = []
        equiv_model.append(Deepset(2,n_hid))
        for _ in range(n_block): equiv_model.append(Deepset(n_hid,n_hid))
        equiv_model.append(Deepset(n_hid,2,output=True))
        self.Equiv = torch.nn.ModuleList(equiv_model).to(device)
        Noneq_model.append(Linear(N,2,n_hid))
        for _ in range(n_block): Noneq_model.append(Linear(N,n_hid,n_hid))
        Noneq_model.append(Linear(N,n_hid,2,output=True))
        self.Noneq = torch.nn.ModuleList(Noneq_model).to(device)
        ## residual strength parameter
        self.gammas = torch.nn.Parameter(torch.ones(n_block+2,).to(device))
        self.lambdas = torch.nn.Parameter(torch.zeros(n_block+2,).to(device))

    def ME(self,module,device):
        dim_in, dim_out = module.A.shape[0], module.A.shape[1]
        return (module.A*(self.diag.to(device))+module.B*(self.offd.to(device))).transpose(1,2).contiguous().view(self.N*dim_in,self.N*dim_out)

    def MN(self, module, device):
        W = torch.kron(module.Wn, module.Wd)
        return W.to(device)

    def forward(self,We,Wn,X):
        X = self.activation(X@(We[0])) + self.gammas[0]*self.activation(X@(Wn[0]))
        for n in range(self.n_block): X = self.activation(X@(We[n+1])) + self.gammas[n+1]*self.activation(X@(Wn[n+1]))
        X = X@(We[-1]) + self.gammas[-1]*(X@(Wn[-1]))
        return X
    
    @property
    def primal_params(self):
        return [p for n,p in self.named_parameters() if not n.endswith("lambdas")]

    @property
    def dual_params(self):
        return [self.lambdas]