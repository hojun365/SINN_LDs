import math
import torch
#-----------------------------------------------------------------------------------------------#
class Simulation():
    def __init__(self,N,μ,τ,K,Da,kbT):
        super(Simulation,self).__init__()
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
        ## force to a
        Fa_r = torch.zeros(size=(self.N,self.N))
        Fa_a = -torch.eye(self.N)/self.τ
        Fr_matrix = torch.cat([Fr_r,Fr_a],dim=0)
        Fa_matrix = torch.cat([Fa_r,Fa_a],dim=0)
        return torch.cat([Fr_matrix,Fa_matrix],dim=-1)
#-----------------------------------------------------------------------------------------------#
class Ritz(torch.nn.Module):
    def __init__(self,n_hid):
        super(Ritz,self).__init__()
        self.activation = torch.nn.Tanh()
        self._forward = torch.nn.Sequential(*[torch.nn.Linear(n_hid,n_hid,bias=False), self.activation,
                                              torch.nn.Linear(n_hid,n_hid,bias=False), self.activation])
    
    def forward(self,X):
        return self._forward(X) + X

class FCN(torch.nn.Module):
    def __init__(self,N,n_hid,n_block):
        super(FCN,self).__init__()
        self.activation = torch.nn.Tanh()
        # forward
        model = [torch.nn.Linear(2*N,n_hid,bias=False), self.activation]
        for _ in range(n_block): model.append(Ritz(n_hid))
        model.append(torch.nn.Linear(n_hid,2*N,bias=False))
        self._forward = torch.nn.Sequential(*model)

    def forward(self,X):
        return self._forward(X)