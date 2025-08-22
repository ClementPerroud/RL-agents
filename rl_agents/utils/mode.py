from contextlib import contextmanager
import torch

@contextmanager
def eval_mode(net : torch.nn.Module):
    '''Temporarily switch to evaluation mode.'''
    istrain = net.training
    try:
        net.eval()
        yield net
    finally:
        if istrain:
            net.train()

@contextmanager
def train_mode(net : torch.nn.Module):
    '''Temporarily switch to evaluation mode.'''
    iseval = not net.training
    try:
        net.train()
        yield net
    finally:
        if iseval:
            net.eval()