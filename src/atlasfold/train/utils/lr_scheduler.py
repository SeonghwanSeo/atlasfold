import torch


class AF2LRScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        base_lr: float = 1e-3,
        num_warmup_steps: int = 1000,
        decay_start_step: int = 50000,
        decay_factor: float = 0.95,
    ) -> None:
        """Initialize the learning rate scheduler.

        Parameters
        ----------
        optimizer : torch.optim.Optimizer
            The optimizer.
        base_lr : float
            The base learning rate, by default 1.0e-3
        num_warmup_steps : int
            The number of warmup steps, by default 1000
        decay_start_step : int
            The step number to start decay, by default 50000
        decay_factor : float
            The decay factor, by default 0.95
        """
        self.base_lr: float = base_lr
        self.max_lr: float = base_lr
        self.num_warmup_steps: int = num_warmup_steps
        self.decay_start_step: int = decay_start_step
        self.decay_factor: float = decay_factor
        super().__init__(optimizer)

    def state_dict(self) -> dict:
        state_dict = {k: v for k, v in self.__dict__.items() if k not in ["optimizer"]}
        return state_dict

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)

    def get_lr(self):
        step = self.last_epoch
        if step <= self.num_warmup_steps:
            lr_ratio = step / self.num_warmup_steps
        elif step > self.decay_start_step:
            lr_ratio = self.decay_factor
        else:
            lr_ratio = 1.0
        lr = lr_ratio * self.base_lr
        return [lr for _ in self.optimizer.param_groups]
