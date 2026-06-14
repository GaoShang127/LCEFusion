#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch
import logging

logger = logging.getLogger()


class Optimizer(object):
    def __init__(self,
                 model,             # Model to be optimized.
                 lr0,               # Learning rate; p' = p - lr * dp, controlling the parameter update magnitude.
                 momentum,          # Momentum factor; the previous update state affects the current parameter update.
                 wd,                # Weight decay factor; regularizes parameters to prevent overfitting and control model complexity.
                 warmup_steps,      # Number of warmup steps; the learning rate is adjusted to the target value after warmup.
                 warmup_start_lr,   # Warmup learning rate; the initial smaller learning rate.
                 max_iter,          # Maximum number of iterations.
                 power,             # Decay power; controls the learning rate decay speed after warmup.
                 it,                # Current iteration.
                 *args, **kwargs):
        self.warmup_steps = warmup_steps            # Get the number of warmup iterations.
        self.warmup_start_lr = warmup_start_lr      # Get the initial warmup learning rate.
        self.lr0 = lr0                              # Get the input learning rate.
        self.lr = self.lr0                          # Set lr to the same value as lr0.
        self.max_iter = float(max_iter)             # Get the maximum number of iterations.
        self.power = power                          # Get the learning rate decay speed.
        self.it = it                                # Get the current iteration.
        # Use get_params() to retrieve the model parameters.
        # get_params() is defined in BiSeNet in model_TII.
        wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = model.get_params()

        # Parameter list for gradient-based optimization, with different settings for different parameter groups.
        param_list = [
                {'params': wd_params},
                {'params': nowd_params, 'weight_decay': 0},
                {'params': lr_mul_wd_params, 'lr_mul': True},
                {'params': lr_mul_nowd_params, 'weight_decay': 0, 'lr_mul': True}]

        self.optim = torch.optim.SGD(
                param_list,             # Network parameters to be optimized.
                lr=lr0,                 # Learning rate for parameter updates.
                momentum=momentum,      # Momentum factor.
                weight_decay=wd)        # Weight decay.

        # Warmup factor.
        # This can be regarded as the learning rate growth factor at each iteration,
        # used to compute the learning rate for the current iteration.
        # e.g., 1st iteration: warmup learning rate
        #       2nd iteration: warmup learning rate * warmup factor
        #       3rd iteration: warmup learning rate * warmup factor * warmup factor
        #       ...
        self.warmup_factor = (self.lr0 / self.warmup_start_lr) ** (1. / self.warmup_steps)


    def get_lr(self):
        # When the current iteration does not exceed the warmup steps.
        if self.it <= self.warmup_steps:
            # Learning rate: warmup learning rate * (warmup factor ^ iteration).
            lr = self.warmup_start_lr * (self.warmup_factor ** self.it)
        # When the current iteration exceeds the warmup steps.
        else:
            # After the warmup steps, the learning rate gradually decays.
            factor = (1 - (self.it - self.warmup_steps) / (self.max_iter - self.warmup_steps)) ** self.power
            lr = self.lr0 * factor
        return lr


    def step(self):
        # Get the current learning rate through get_lr().
        self.lr = self.get_lr()
        # optimizer.param_groups is a list whose elements are dictionaries.
        # The dictionary keys include ['params', 'lr', 'betas', 'eps', 'weight_decay', 'amsgrad'].
        # The parameter values can be changed through dictionary operations.
        # Iterate over the parameter groups to be optimized; four parameter groups are defined.
        for pg in self.optim.param_groups:
            # If the key 'lr_mul' exists in the parameter group, return its value; otherwise, return False.
            if pg.get('lr_mul', False):
                # Multiply the learning rate by 10.
                pg['lr'] = self.lr * 10
            # If the key 'lr_mul' does not exist, False is returned.
            else:
                # Keep the learning rate unchanged.
                pg['lr'] = self.lr
        # optim.defaults stores the optimizer hyperparameters.
        # After applying the learning rate adjustment strategy, the learning rate in optim.defaults
        # will not be updated automatically, so the same adjustment needs to be applied.
        if self.optim.defaults.get('lr_mul', False):
            self.optim.defaults['lr'] = self.lr * 10
        else:
            self.optim.defaults['lr'] = self.lr
        self.it += 1
        self.optim.step()
        # When the current iteration reaches warmup_steps + 2, log that warmup is finished
        # and the poly learning rate strategy starts.
        if self.it == self.warmup_steps + 2:
            logger.info('==> warmup done, start to implement poly lr strategy')

    def zero_grad(self):
        self.optim.zero_grad()

