import random

import numpy as np
import torch
from prettytable import PrettyTable
from torch.optim.lr_scheduler import LambdaLR


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """ Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def gpu_information_summary(show=True):
    n_gpu = torch.cuda.device_count()
    gpu_name = torch.cuda.get_device_name()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    table = PrettyTable()
    table.field_names = ["Key", "Value"]
    table.add_row(["GPU", gpu_name])
    table.add_row(["Number of GPUs", n_gpu])
    if show:
        print(table)
    return n_gpu, device


def set_seed(seed_value, n_gpu):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed_value)
