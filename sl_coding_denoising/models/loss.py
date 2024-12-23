#!/usr/local/bin/python
import cv2
import torch
import random
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F


class pred_loss(nn.Module):
    """the loss between the input and synthesized input"""

    def __init__(self):
        super(pred_loss, self).__init__()

    def forward(self, pred_disp, disp):

        loss = torch.mean(torch.abs(pred_disp - disp))

        # print("loss:{}".format(str(round(loss.item(), 10))))

        return loss
