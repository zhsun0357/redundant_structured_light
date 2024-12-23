import torch
import numpy as np


def save_on_master(*args, **kwargs):
    torch.save(*args, **kwargs)


def calculate_error(pred_disp, disp):
    return np.sum(np.abs(pred_disp - disp) >= 2.0) / disp.shape[0] / disp.shape[1]
