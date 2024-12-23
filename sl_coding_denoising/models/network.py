import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy
import torch.nn.functional as F
from models.modules import *


class disp_net(nn.Module):

    def __init__(self, out_c=1, top_k=10):
        super(disp_net, self).__init__()

        self.encoder = ResEncoder(top_k * 2)
        self.decoder = ResDecoder(out_c)

        self.down_scales = 2 ** 4

    def forward(self, x):
        b, c, h_inp, w_inp = x.shape

        hb, wb = self.down_scales, self.down_scales
        pad_h = (hb - h_inp % hb) % hb
        pad_w = (wb - w_inp % wb) % wb

        x = F.pad(x, [0, pad_w, 0, pad_h], mode='reflect')

        enc_ftrs = self.encoder(x)
        out = self.decoder(enc_ftrs)

        return out[:, :, :h_inp, :w_inp] + x[:, 0:1, :h_inp, :w_inp]
