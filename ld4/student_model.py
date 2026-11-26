import math
import torch
from torch import nn
class CausalAttention(nn.Module):
    def __init__(self,dim,heads):
        super().__init__();assert dim%heads==0
        self.heads=heads;self.d=dim//heads
        self.qkv=nn.Linear(dim,3*dim);self.out=nn.Linear(dim,dim)
    def forward(self,x):
        b,t,d=x.shape
        q,k,v=[a.reshape(b,t,self.heads,self.d).transpose(1,2) for a in self.qkv(x).chunk(3,-1)]
        scores=q@k.transpose(-2,-1)/math.sqrt(self.d)
        future=torch.triu(torch.ones(t,t,dtype=torch.bool,device=x.device),diagonal=1)
        scores=scores.masked_fill(future,float('-inf'))
        weights=torch.softmax(scores,dim=-1)
        mixed=weights@v
        mixed=mixed.transpose(1,2).contiguous().reshape(b,t,d)
        return self.out(mixed)
