import torch

def energy_fraction_loss(pred, target, weight=None):

    batch_size = pred.size()[0]
    
    target  = (torch.abs(target)<1e-5)*1e-5+(torch.abs(target)>1e-5)*target
    w_sqsum = (pred-target)**2
    if weight is not None:
        w_sqsum = w_sqsum*weight
        target  = target*weight
    loss      = torch.sum(w_sqsum/target)
    loss      = loss/batch_size

    return loss

def energy_fraction_loss(pred, target, weight=None):

    batch_size = pred.size()[0]
    target     = (torch.abs(target)<1e-5)*1e-5+(torch.abs(target)>1e-5)*target
    w_sqsum = (pred-target)**2
    if weight is not None:
        w_sqsum = w_sqsum*weight
        target  = target*weight
    target    = torch.abs(target)
    loss      = torch.sum(w_sqsum/target)
    loss      = loss/batch_size

    return loss

def compressed_loss(pred, target, weight=None):
    
    batch_size = pred.size()[0]
    logtarget  = torch.log(target)
    logpred    = torch.log(pred)
    logtarget  = (torch.abs(logtarget)<1e-5)*1e-5+(torch.abs(logtarget)>1e-5)*target

    w_sqsum    = (logpred-logtarget)**2
    if weight is not None:
        w_sqsum   = w_sqsum*weight
        logtarget = logtarget*weight
    loss       = torch.sum(w_sqsum)/logtarget
