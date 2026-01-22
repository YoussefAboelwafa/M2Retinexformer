import math
from collections import Counter
from torch.optim.lr_scheduler import _LRScheduler
import torch


class MultiStepRestartLR(_LRScheduler):
    """MultiStep with restarts learning rate scheme.

    Args:
        optimizer (torch.nn.optimizer): Torch optimizer.
        milestones (list): Iterations that will decrease learning rate.
        gamma (float): Decrease ratio. Default: 0.1.
        restarts (list): Restart iterations. Default: [0].
        restart_weights (list): Restart weights at each restart iteration.
            Default: [1].
        last_epoch (int): Used in _LRScheduler. Default: -1.
    """

    def __init__(
        self,
        optimizer,
        milestones,
        gamma=0.1,
        restarts=(0,),
        restart_weights=(1,),
        last_epoch=-1,
    ):
        self.milestones = Counter(milestones)
        self.gamma = gamma
        self.restarts = restarts
        self.restart_weights = restart_weights
        assert len(self.restarts) == len(
            self.restart_weights
        ), "restarts and their weights do not match."
        super(MultiStepRestartLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch in self.restarts:
            weight = self.restart_weights[self.restarts.index(self.last_epoch)]
            return [
                group["initial_lr"] * weight for group in self.optimizer.param_groups
            ]
        if self.last_epoch not in self.milestones:
            return [group["lr"] for group in self.optimizer.param_groups]
        return [
            group["lr"] * self.gamma ** self.milestones[self.last_epoch]
            for group in self.optimizer.param_groups
        ]


class LinearLR(_LRScheduler):
    """

    Args:
        optimizer (torch.nn.optimizer): Torch optimizer.
        milestones (list): Iterations that will decrease learning rate.
        gamma (float): Decrease ratio. Default: 0.1.
        last_epoch (int): Used in _LRScheduler. Default: -1.
    """

    def __init__(self, optimizer, total_iter, last_epoch=-1):
        self.total_iter = total_iter
        super(LinearLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        process = self.last_epoch / self.total_iter
        weight = 1 - process
        # print('get lr ', [weight * group['initial_lr'] for group in self.optimizer.param_groups])
        return [weight * group["initial_lr"] for group in self.optimizer.param_groups]


class VibrateLR(_LRScheduler):
    """

    Args:
        optimizer (torch.nn.optimizer): Torch optimizer.
        milestones (list): Iterations that will decrease learning rate.
        gamma (float): Decrease ratio. Default: 0.1.
        last_epoch (int): Used in _LRScheduler. Default: -1.
    """

    def __init__(self, optimizer, total_iter, last_epoch=-1):
        self.total_iter = total_iter
        super(VibrateLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        process = self.last_epoch / self.total_iter

        f = 0.1
        if process < 3 / 8:
            f = 1 - process * 8 / 3
        elif process < 5 / 8:
            f = 0.2

        T = self.total_iter // 80
        Th = T // 2

        t = self.last_epoch % T

        f2 = t / Th
        if t >= Th:
            f2 = 2 - f2

        weight = f * f2

        if self.last_epoch < Th:
            weight = max(0.1, weight)

        # print('f {}, T {}, Th {}, t {}, f2 {}'.format(f, T, Th, t, f2))
        return [weight * group["initial_lr"] for group in self.optimizer.param_groups]


def get_position_from_periods(iteration, cumulative_period):
    """Get the position from a period list.

    It will return the index of the right-closest number in the period list.
    For example, the cumulative_period = [100, 200, 300, 400],
    if iteration == 50, return 0;
    if iteration == 210, return 2;
    if iteration == 300, return 2.

    Args:
        iteration (int): Current iteration.
        cumulative_period (list[int]): Cumulative period list.

    Returns:
        int: The position of the right-closest number in the period list.
    """
    for i, period in enumerate(cumulative_period):
        if iteration <= period:
            return i
    # If iteration exceeds all periods, return the last period index
    return len(cumulative_period) - 1


class CosineAnnealingRestartLR(_LRScheduler):
    """Cosine annealing with restarts learning rate scheme.

    An example of config:
    periods = [10, 10, 10, 10]
    restart_weights = [1, 0.5, 0.5, 0.5]
    eta_min=1e-7

    It has four cycles, each has 10 iterations. At 10th, 20th, 30th, the
    scheduler will restart with the weights in restart_weights.

    Args:
        optimizer (torch.nn.optimizer): Torch optimizer.
        periods (list): Period for each cosine anneling cycle.
        restart_weights (list): Restart weights at each restart iteration.
            Default: [1].
        eta_min (float): The mimimum lr. Default: 0.
        last_epoch (int): Used in _LRScheduler. Default: -1.
    """

    def __init__(
        self, optimizer, periods, restart_weights=(1,), eta_min=0, last_epoch=-1
    ):
        self.periods = periods
        self.restart_weights = restart_weights
        self.eta_min = eta_min
        assert len(self.periods) == len(
            self.restart_weights
        ), "periods and restart_weights should have the same length."
        self.cumulative_period = [
            sum(self.periods[0 : i + 1]) for i in range(0, len(self.periods))
        ]
        super(CosineAnnealingRestartLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        idx = get_position_from_periods(self.last_epoch, self.cumulative_period)
        current_weight = self.restart_weights[idx]
        nearest_restart = 0 if idx == 0 else self.cumulative_period[idx - 1]
        current_period = self.periods[idx]

        return [
            self.eta_min
            + current_weight
            * 0.5
            * (base_lr - self.eta_min)
            * (
                1
                + math.cos(
                    math.pi * ((self.last_epoch - nearest_restart) / current_period)
                )
            )
            for base_lr in self.base_lrs
        ]


class CosineAnnealingRestartCyclicLR(_LRScheduler):
    """Cosine annealing with restarts learning rate scheme.
    An example of config:
    periods = [10, 10, 10, 10]
    restart_weights = [1, 0.5, 0.5, 0.5]
    eta_min=1e-7
    It has four cycles, each has 10 iterations. At 10th, 20th, 30th, the
    scheduler will restart with the weights in restart_weights.
    Args:
        optimizer (torch.nn.optimizer): Torch optimizer.
        periods (list): Period for each cosine anneling cycle.
        restart_weights (list): Restart weights at each restart iteration.
            Default: [1].
        eta_min (float): The mimimum lr. Default: 0.
        last_epoch (int): Used in _LRScheduler. Default: -1.
    """

    def __init__(
        self, optimizer, periods, restart_weights=(1,), eta_mins=(0,), last_epoch=-1
    ):
        self.periods = periods
        self.restart_weights = restart_weights
        self.eta_mins = eta_mins
        assert len(self.periods) == len(
            self.restart_weights
        ), "periods and restart_weights should have the same length."
        self.cumulative_period = [
            sum(self.periods[0 : i + 1]) for i in range(0, len(self.periods))
        ]
        super(CosineAnnealingRestartCyclicLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        idx = get_position_from_periods(self.last_epoch, self.cumulative_period)
        current_weight = self.restart_weights[idx]
        nearest_restart = 0 if idx == 0 else self.cumulative_period[idx - 1]
        current_period = self.periods[idx]
        eta_min = self.eta_mins[idx]

        return [
            eta_min
            + current_weight
            * 0.5
            * (base_lr - eta_min)
            * (
                1
                + math.cos(
                    math.pi * ((self.last_epoch - nearest_restart) / current_period)
                )
            )
            for base_lr in self.base_lrs
        ]


class ReduceLROnPlateauWrapper:
    """Wrapper for ReduceLROnPlateau to work with basicsr's scheduler interface.

    Unlike other schedulers that inherit from _LRScheduler, ReduceLROnPlateau
    requires a metric to be passed to step(). This wrapper provides compatibility
    with basicsr's training loop.

    Args:
        optimizer (torch.nn.optimizer): Torch optimizer.
        mode (str): One of 'min' or 'max'. In 'min' mode, lr will be reduced
            when the quantity monitored has stopped decreasing; in 'max' mode
            it will be reduced when the quantity monitored has stopped increasing.
            Default: 'min'.
        factor (float): Factor by which the learning rate will be reduced.
            new_lr = lr * factor. Default: 0.1.
        patience (int): Number of epochs with no improvement after which learning
            rate will be reduced. Default: 10.
        threshold (float): Threshold for measuring the new optimum, to only focus
            on significant changes. Default: 1e-4.
        threshold_mode (str): One of 'rel' or 'abs'. In 'rel' mode,
            dynamic_threshold = best * (1 + threshold) in 'max' mode or
            best * (1 - threshold) in 'min' mode. In 'abs' mode,
            dynamic_threshold = best + threshold in 'max' mode or
            best - threshold in 'min' mode. Default: 'rel'.
        cooldown (int): Number of epochs to wait before resuming normal operation
            after lr has been reduced. Default: 0.
        min_lr (float or list): A scalar or a list of scalars. A lower bound on
            the learning rate of all param groups. Default: 0.
        eps (float): Minimal decay applied to lr. If the difference between new
            and old lr is smaller than eps, the update is ignored. Default: 1e-8.
        verbose (bool): If True, prints a message to stdout for each update.
            Default: False.
    """

    def __init__(
        self,
        optimizer,
        mode="min",
        factor=0.1,
        patience=10,
        threshold=1e-4,
        threshold_mode="rel",
        cooldown=0,
        min_lr=0,
        eps=1e-8,
        verbose=False,
        metric="psnr",  # Which validation metric to monitor (used by training loop)
    ):
        self.optimizer = optimizer
        self.metric = metric  # Store which metric to use for stepping
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=mode,
            factor=factor,
            patience=patience,
            threshold=threshold,
            threshold_mode=threshold_mode,
            cooldown=cooldown,
            min_lr=min_lr,
            eps=eps,
            verbose=verbose,
        )
        self._last_lr = [group["lr"] for group in optimizer.param_groups]
        self.last_epoch = 0

    def step(self, metrics=None):
        """Step the scheduler.

        Args:
            metrics (float, optional): The metric value to check for improvement.
                If None, the scheduler step is skipped (for compatibility with
                iteration-based stepping).
        """
        if metrics is not None:
            self.scheduler.step(metrics)
            self._last_lr = [group["lr"] for group in self.optimizer.param_groups]
        self.last_epoch += 1

    def get_last_lr(self):
        """Return last computed learning rate by current scheduler."""
        return self._last_lr

    def state_dict(self):
        """Return the state of the scheduler as a dict."""
        return {
            "scheduler_state_dict": self.scheduler.state_dict(),
            "last_epoch": self.last_epoch,
            "_last_lr": self._last_lr,
        }

    def load_state_dict(self, state_dict):
        """Load the scheduler's state."""
        self.scheduler.load_state_dict(state_dict["scheduler_state_dict"])
        self.last_epoch = state_dict["last_epoch"]
        self._last_lr = state_dict["_last_lr"]
