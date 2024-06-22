import torch
import numpy as np
from thop import profile
from thop import clever_format


def clip_gradient(optimizer, grad_clip):
    """
    For calibrating misalignment gradient via cliping gradient technique
    :param optimizer:
    :param grad_clip:
    :return:
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=30):
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay
    
    
def diy_lr(args, epoch, base_lr, optimizer):
    if args.warmup and epoch < args.warmup_period:
        lr_ = base_lr * ((epoch + 1) / args.warmup_period)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_
    else:
        if args.warmup:
            shift_iter = epoch - args.warmup_period
            assert shift_iter >= 0, f'Shift iter is {shift_iter}, smaller than zero'
        else:
            shift_iter = epoch
        lr_ = base_lr * (1.0 - shift_iter / args.epoch) ** 0.9  # learning rate adjustment depends on the max iterations
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_
    
    
    
class AvgMeter(object):
    def __init__(self, num=40):
        self.num = num
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.losses = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.losses.append(val)

    def show(self):
        return torch.mean(torch.stack(self.losses[np.maximum(len(self.losses)-self.num, 0):]))


def CalParams(model, input_tensor):
    """
    Usage:
        Calculate Params and FLOPs via [THOP](https://github.com/Lyken17/pytorch-OpCounter)
    Necessarity:
        from thop import profile
        from thop import clever_format
    :param model:
    :param input_tensor:
    :return:
    """
    flops, params = profile(model, inputs=(input_tensor,))
    flops, params = clever_format([flops, params], "%.3f")
    print('[Statistics Information]\nFLOPs: {}\nParams: {}'.format(flops, params))