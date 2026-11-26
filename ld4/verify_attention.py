import torch,importlib,argparse
from student_model import CausalAttention as Attention
torch.manual_seed(5);m=Attention(8,2);x=torch.randn(2,5,8,requires_grad=True)
y=m(x);assert y.shape==x.shape and torch.isfinite(y).all()
future=x.detach().clone();future[:,3:]+=10
assert torch.allclose(y[:,:3],m(future)[:,:3],atol=1e-6),'Future leakage'
y[:,:3].square().sum().backward();assert x.grad[:,3:].abs().max()==0,'Future gradient leakage'
assert any(p.grad is not None and p.grad.abs().sum()>0 for p in m.parameters()),'No learning gradient'
with torch.no_grad():
    m=Attention(4,2);m.qkv.weight.zero_();m.qkv.bias.zero_();m.qkv.weight[8:]=torch.eye(4);m.out.weight.copy_(torch.eye(4));m.out.bias.zero_()
x=torch.tensor([[[1.,2.,3.,4.],[3.,4.,5.,6.],[5.,6.,7.,8.]]]);expected=torch.tensor([[[1.,2.,3.,4.],[2.,3.,4.,5.],[3.,4.,5.,6.]]])
assert torch.allclose(m(x),expected,atol=1e-6),'Wrong causal prefix weighted sum or head merging'
print('PASS: shape, future mask, gradient isolation, known value sum')
