import math
from bisect import bisect_right
from torch.optim.lr_scheduler import _LRScheduler


def to_tuple(x, L):
    if type(x) in (int, float):
        return [x] * L
    if type(x) in (list, tuple):
        if len(x) != L:
            raise ValueError("length of {} ({}) != {}".format(x, len(x), L))
        return tuple(x)
    raise ValueError("input {} has unkown type {}".format(x, type(x)))


class WarmupLR(_LRScheduler):
    def __init__(
        self, optimizer, warmup_epochs=0, warmup_powers=1, warmup_lrs=0, last_epoch=-1
    ):
        self.num_groups = len(optimizer.param_groups)
        self.warmup_epochs = to_tuple(warmup_epochs, self.num_groups)
        self.warmup_powers = to_tuple(warmup_powers, self.num_groups)
        self.warmup_lrs = to_tuple(warmup_lrs, self.num_groups)
        super(WarmupLR, self).__init__(optimizer, last_epoch)
        assert self.num_groups == len(self.base_lrs)

    def get_lr(self):
        curr_lrs = []
        for group_index in range(self.num_groups):
            if self.last_epoch < self.warmup_epochs[group_index]:
                progress = self.last_epoch / self.warmup_epochs[group_index]
                factor = progress ** self.warmup_powers[group_index]
                lr_gap = self.base_lrs[group_index] - self.warmup_lrs[group_index]
                curr_lrs.append(factor * lr_gap + self.warmup_lrs[group_index])
            else:
                curr_lrs.append(self.get_single_lr_after_warmup(group_index))
        return curr_lrs

    def get_single_lr_after_warmup(self, group_index):
        raise NotImplementedError


class WarmupMultiStepLR(WarmupLR):
    def __init__(
        self,
        optimizer,
        milestones,
        gamma=0.1,
        warmup_epochs=0,
        warmup_powers=1,
        warmup_lrs=0,
        last_epoch=-1,
    ):

        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of"
                " increasing integers. Got %s" % repr(milestones)
            )
        self.milestones = milestones
        self.gamma = gamma
        super(WarmupMultiStepLR, self).__init__(
            optimizer, warmup_epochs, warmup_powers, warmup_lrs, last_epoch
        )
        if self.milestones[0] <= max(self.warmup_epochs):
            raise ValueError(
                "milstones[0] ({}) <= max(warmup_epochs) ({})".format(
                    milestones[0], max(self.warmup_epochs)
                )
            )

    def get_single_lr_after_warmup(self, group_index):
        factor = self.gamma ** bisect_right(self.milestones, self.last_epoch)
        return self.base_lrs[group_index] * factor


class WarmupCosineAnnealingLR(WarmupLR):
    def __init__(
        self,
        optimizer,
        total_epoch,
        final_factor=0,
        warmup_epochs=0,
        warmup_powers=1,
        warmup_lrs=0,
        last_epoch=-1,
    ):
        self.total_epoch = total_epoch
        self.final_factor = final_factor
        super(WarmupCosineAnnealingLR, self).__init__(
            optimizer, warmup_epochs, warmup_powers, warmup_lrs, last_epoch
        )

    def get_single_lr_after_warmup(self, group_index):
        warmup_epoch = self.warmup_epochs[group_index]
        progress = (self.last_epoch - warmup_epoch) / (self.total_epoch - warmup_epoch)
        progress = min(progress, 1.0)
        cosine_progress = (math.cos(math.pi * progress) + 1) / 2
        factor = cosine_progress * (1 - self.final_factor) + self.final_factor
        return self.base_lrs[group_index] * factor
