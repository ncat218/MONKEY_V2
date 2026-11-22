"""Pure helpers for pinned DySample; zero offsets reproduce bilinear upsampling."""
import torch
from rev4.dysample_vendor.dysample import DySample


def sampler(channels,scale):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        module=DySample(channels,scale=scale,style='lp',groups=4,dyscope=False)
    torch.nn.init.zeros_(module.offset.weight)
    torch.nn.init.zeros_(module.offset.bias)
    return module


def load_shared(model,state):
    actual=model.state_dict()
    missing=set(actual)-set(state)
    allowed=('neck.asf_conv.','neck.asf_attn.','neck.top_sample.','neck.output_sample.')
    if set(state)-set(actual) or any(not key.startswith(allowed) for key in missing):
        raise ValueError('Unexpected B1 key/schema mismatch outside declared new modules')
    if any(actual[k].shape!=v.shape or actual[k].dtype!=v.dtype for k,v in state.items()):
        raise ValueError('B1 shared tensor shape/dtype mismatch')
    result=model.load_state_dict(state,strict=False)
    if set(result.missing_keys)!=missing or result.unexpected_keys:raise ValueError('Shared load mismatch')
    if not all(torch.equal(model.state_dict()[k].cpu(),v.cpu()) for k,v in state.items()):
        raise ValueError('Loaded base tensors differ from B1')


def state_digest(module):
    import hashlib
    h=hashlib.sha256()
    for key,value in module.state_dict().items():
        h.update((key+str(value.shape)+str(value.dtype)).encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
