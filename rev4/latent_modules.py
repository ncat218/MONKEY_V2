"""Small residual feature refiners; no resizing, pretrained external weights or RGB output."""
import torch
import hashlib
from torch import nn
from torch.nn import functional as F


class ChannelNorm(nn.Module):
    def __init__(self,channels):
        super().__init__()
        self.weight=nn.Parameter(torch.ones(1,channels,1,1))
        self.bias=nn.Parameter(torch.zeros(1,channels,1,1))

    def forward(self,x):
        mean=x.mean(1,keepdim=True)
        variance=(x-mean).square().mean(1,keepdim=True)
        return (x-mean)*torch.rsqrt(variance+1e-5)*self.weight+self.bias


class ResidualRefiner(nn.Module):
    """Plain 96-wide or Restormer-GDFN-inspired gated 64-wide residual."""
    def __init__(self,kind,channels=256):
        super().__init__()
        if kind not in ('plain','gated'):
            raise ValueError('Unknown latent refiner kind')
        self.kind=kind
        hidden=96 if kind=='plain' else 64
        incoming=hidden if kind=='plain' else 2*hidden
        self.norm=ChannelNorm(channels)
        self.project_in=nn.Conv2d(channels,incoming,1,bias=False)
        self.depthwise=nn.Conv2d(incoming,incoming,3,padding=1,groups=incoming,bias=False)
        self.project_out=nn.Conv2d(hidden,channels,1,bias=False)
        nn.init.zeros_(self.project_out.weight)

    def forward(self,x):
        value=self.depthwise(self.project_in(self.norm(x)))
        if self.kind=='gated':
            u,v=value.chunk(2,dim=1)
            value=F.gelu(u)*v
        else:
            value=F.gelu(value)
        return x+self.project_out(value)


def build_refiner(kind,seed=42):
    if kind=='head':
        return nn.Identity()
    # Adapter initialization must not consume the training/data RNG stream.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return ResidualRefiner(kind)


def load_base_state(model,state):
    """Allow only the explicitly new feature_adapter keys when loading B1."""
    actual=model.state_dict()
    expected_missing={k for k in actual if k.startswith('feature_adapter.')}
    if set(actual)-set(state)!=expected_missing or set(state)-set(actual):
        raise ValueError('B1 schema mismatch beyond the new feature_adapter keys')
    if any(actual[k].shape!=v.shape or actual[k].dtype!=v.dtype for k,v in state.items()):
        raise ValueError('B1 tensor shape/dtype mismatch')
    result=model.load_state_dict(state,strict=False)
    if set(result.missing_keys)!=expected_missing or result.unexpected_keys:
        raise ValueError('Unexpected state_dict load result')
    if not all(torch.equal(model.state_dict()[k].cpu(),v.cpu()) for k,v in state.items()):
        raise ValueError('Loaded B1 base tensors differ')


def frozen_digest(model):
    digest=hashlib.sha256()
    for component in ('backbone','neck'):
        for key,value in getattr(model,component).state_dict().items():
            digest.update((component+'.'+key+str(value.shape)+str(value.dtype)).encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()
