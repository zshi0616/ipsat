import os 
import numpy as np 
import torch

def naive_baseline_loss(x, y, criterion, zero=False):
    predict = x[:, -1, :]
    if zero:
        predict = torch.zeros_like(predict)
    
    # repeat predict n*m to n*3*m
    predict = predict.unsqueeze(1).repeat(1, y.shape[1], 1)
    
    loss = criterion(predict, y)
    return loss.item()

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count > 0:
          self.avg = self.sum / self.count
