"""Manuscript equations: two attention levels, return normalization and readout."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

def segment_sum(values,index,n):
    shape=(n,*values.shape[1:])
    return values.new_zeros(shape).index_add_(0,index,values)

def segment_softmax(scores,index,n):
    maxima=scores.new_full((n,),-torch.inf)
    maxima.scatter_reduce_(0,index,scores.detach(),reduce='amax',include_self=True)
    weights=(scores-maxima[index]).exp()
    return weights/segment_sum(weights,index,n)[index].clamp_min(1e-12)

class AttentionLayer(nn.Module):
    def __init__(self,d):
        super().__init__();self.d=d
        self.Wv=nn.Linear(d,d,bias=False);self.Wh=nn.Linear(d,d,bias=False)
        self.Wq=nn.Linear(d,d,bias=False);self.Wk=nn.Linear(d,d,bias=False);self.Wo=nn.Linear(d,d,bias=False)
        self.a=nn.Parameter(torch.empty(d));self.bias=nn.Parameter(torch.zeros(d))
        for layer in [self.Wv,self.Wh,self.Wq,self.Wk,self.Wo]:nn.init.xavier_uniform_(layer.weight)
        nn.init.xavier_uniform_(self.a.unsqueeze(0))

    def forward(self,x,g):
        v,e=g.vertices,g.edges
        degree=torch.bincount(e,minlength=g.n_edges).to(x.dtype).clamp_min(1)
        mean=segment_sum(x[v],e,g.n_edges)/degree[:,None]
        vx=self.Wv(x);context=self.Wh(mean)
        score=(torch.tanh(vx[v]+context[e]+self.bias)*self.a).sum(-1)
        alpha=segment_softmax(score,e,g.n_edges)
        z=segment_sum(alpha[:,None]*vx[v],e,g.n_edges)
        query,key,value=self.Wq(z),self.Wk(z),self.Wo(z)
        b=(query[g.query]*key[g.neighbor]).sum(-1)/math.sqrt(self.d)
        beta=segment_softmax(b,g.query,g.n_edges)
        zhat=segment_sum(beta[:,None]*value[g.neighbor],g.query,g.n_edges)
        denom=segment_sum(alpha,v,g.n_vertices)
        pi=alpha/denom[v].clamp_min(1e-12)
        message=segment_sum(pi[:,None]*zhat[e],v,g.n_vertices)
        out=.5*x+.5*message
        return torch.where((denom>0)[:,None],out,x)

class DHCLDD(nn.Module):
    def __init__(self,n_users,n_items,d=64,layers=2,checkpoint_layers=True):
        super().__init__();self.n_users=n_users;self.n_items=n_items
        self.embedding=nn.Embedding(n_users+n_items,d)
        nn.init.xavier_uniform_(self.embedding.weight)
        self.layers=nn.ModuleList([AttentionLayer(d) for _ in range(layers)])
        self.checkpoint_layers=checkpoint_layers

    def encode(self,g):
        x=self.embedding.weight;total=x
        for layer in self.layers:
            if self.training and torch.is_grad_enabled() and self.checkpoint_layers:
                x=checkpoint(layer,x,g,use_reentrant=False)
            else:x=layer(x,g)
            total=total+x
        return total/(len(self.layers)+1)

    def regularization(self,entity_ids):
        entities=torch.unique(entity_ids)
        emb=self.embedding(entities).square().sum(-1).mean()
        parameters=list(self.layers.parameters())
        theta=sum(p.square().sum() for p in parameters)/sum(p.numel() for p in parameters)
        return emb+theta

def directed_infonce(x,y,tau,chunk=256):
    """Exact full-denominator InfoNCE; chunking only changes memory usage."""
    n=len(x)
    if n<2:return (x.sum()+y.sum())*0
    total=x.new_zeros(())
    for begin in range(0,n,chunk):
        end=min(begin+chunk,n)
        labels=torch.arange(begin,end,device=x.device)
        def loss(a,b,target):return F.cross_entropy((a@b.T)/tau,target,reduction='sum')
        if torch.is_grad_enabled():total=total+checkpoint(loss,x[begin:end],y,labels,use_reentrant=False)
        else:total=total+loss(x[begin:end],y,labels)
    return total/n

def contrastive_loss(original,filtered,users,items,tau=.2,chunk=256):
    loss=original.new_zeros(())
    for ids in [torch.unique(users),torch.unique(items)]:
        a=F.normalize(original[ids],dim=-1,eps=1e-8)
        b=F.normalize(filtered[ids],dim=-1,eps=1e-8)
        loss=loss+.5*(directed_infonce(a,b,tau,chunk)+directed_infonce(b,a,tau,chunk))
    return loss
