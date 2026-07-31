#!/usr/bin/env python3
# encoding: utf-8
# @Time    : 2018/8/1 上午1:50
# @Author  : yuchangqian
# @Contact : changqian_yu@163.com
# @File    : lr_policy.py.py

from abc import ABCMeta, abstractmethod


class BaseLR():
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_lr(self, cur_iter): pass


class PolyLR(BaseLR):
    def __init__(self, start_lr, lr_power, total_iters):
        self.start_lr = start_lr
        self.lr_power = lr_power
        self.total_iters = total_iters + 0.0

    def get_lr(self, cur_iter):
        return self.start_lr * (
                (1 - float(cur_iter) / self.total_iters) ** self.lr_power)


class WarmUpPolyLR(BaseLR):
    def __init__(self, start_lr, lr_power, total_iters, warmup_steps):
        self.start_lr = start_lr
        self.lr_power = lr_power
        self.total_iters = total_iters + 0.0
        self.warmup_steps = warmup_steps

    def get_lr(self, cur_iter):
        if cur_iter < self.warmup_steps:
            return self.start_lr * (cur_iter / self.warmup_steps)
        else:
            return self.start_lr * (
                    (1 - float(cur_iter) / self.total_iters) ** self.lr_power)


#多项式衰减学习策略（支持多参数组）
class MultiGroupWarmUpPolyLR:
    def __init__(self, optimizer, base_lrs, lr_power, total_iters, warmup_steps):
        """
        多参数组的学习率调度器

        Args:
            optimizer: 优化器对象
            base_lrs: 每个参数组的初始学习率列表 [lr1, lr2, ...]
            lr_power: 多项式衰减指数
            total_iters: 总迭代次数
            warmup_steps: 预热步数
        """
        self.optimizer = optimizer
        self.base_lrs = base_lrs
        self.lr_power = lr_power
        self.total_iters = total_iters
        self.warmup_steps = warmup_steps

        # 验证base_lrs长度与参数组数量一致
        assert len(base_lrs) == len(optimizer.param_groups), \
            f"base_lrs长度({len(base_lrs)})与参数组数量({len(optimizer.param_groups)})不匹配"

    def get_lr(self, cur_iter, group_idx):
        """
        获取指定参数组在指定迭代次数的学习率

        Args:
            cur_iter: 当前迭代次数
            group_idx: 参数组索引
        """
        base_lr = self.base_lrs[group_idx]

        if cur_iter < self.warmup_steps:
            # 预热阶段：从0线性增长到base_lr
            return base_lr * (cur_iter / self.warmup_steps)
        else:
            # 多项式衰减阶段
            progress = float(cur_iter) / self.total_iters
            return base_lr * ((1 - progress) ** self.lr_power)

    def step(self, cur_iter):
        """更新所有参数组的学习率"""
        for i, param_group in enumerate(self.optimizer.param_groups):
            new_lr = self.get_lr(cur_iter, i)
            param_group['lr'] = new_lr

    def get_current_lrs(self):
        """获取当前所有参数组的学习率"""
        return [group['lr'] for group in self.optimizer.param_groups]

class MultiStageLR(BaseLR):
    def __init__(self, lr_stages):
        assert type(lr_stages) in [list, tuple] and len(lr_stages[0]) == 2, \
            'lr_stages must be list or tuple, with [iters, lr] format'
        self._lr_stagess = lr_stages

    def get_lr(self, epoch):
        for it_lr in self._lr_stagess:
            if epoch < it_lr[0]:
                return it_lr[1]


class LinearIncreaseLR(BaseLR):
    def __init__(self, start_lr, end_lr, warm_iters):
        self._start_lr = start_lr
        self._end_lr = end_lr
        self._warm_iters = warm_iters
        self._delta_lr = (end_lr - start_lr) / warm_iters

    def get_lr(self, cur_epoch):
        return self._start_lr + cur_epoch * self._delta_lr