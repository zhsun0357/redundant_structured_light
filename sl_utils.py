import cv2
import os
import imageio
from tqdm import tqdm
import pdb
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import json
import pickle
import time

import scipy
import scipy.signal as signal
from scipy.io import loadmat
from scipy import ndimage

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
## if has GPU, use cuda


def get_code_all(code, resy):
    """
    insert two frames (first and last, all 0 and all 1) for normalization
    """
    code = code.transpose()[:resy]
    code_all = np.zeros((code.shape[0], code.shape[1]+2))
    code_all[:,0] = np.zeros_like(code[...,0])
    code_all[:,-1] = np.ones_like(code[...,0])
    code_all[:,1:-1] = code
    return code_all


def disp_warp_batch(x, dispx, dispy, interp_mode='bilinear', padding_mode='zeros'):
    """
    use disparity to warp a batch of images/patterns
    """
    x = torch.from_numpy(x.transpose(2,0,1)).unsqueeze(1).float().to(device)
    B, C, H, W = x.size()
    disp = torch.cat((torch.from_numpy(np.repeat(dispx[np.newaxis,...], B, axis = 0)).unsqueeze(3), \
                      torch.from_numpy(np.repeat(dispy[np.newaxis,...], B, axis = 0)).unsqueeze(3)), dim = 3).float().to(device)
    
    assert x.size()[-2:] == disp.size()[1:3]
    # mesh grid
    grid_y, grid_x = torch.meshgrid(torch.arange(0, H), torch.arange(0, W))
    grid = torch.stack((grid_x, grid_y), 2).float()  # W(x), H(y), 2
    grid.requires_grad = False
    grid = grid.type_as(x)
    vgrid = grid + disp
    # scale grid to [-1,1]
    vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(W, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(H, 1) - 1.0
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
    # pdb.set_trace()
    output = F.grid_sample(x, vgrid_scaled, mode=interp_mode, padding_mode=padding_mode)
    return output


def sl_simu_batch(pattern_all, intensity, disp, pr, sr, poiss_K, noise_level, t_exp, disp_clip):
    """
    simulate imaging process with noise
    """
    cam_imgs = []
    t_calib = 3/4/(pr+sr)
    ## make sure no saturation
    assert (pr+sr)*t_exp/(pattern_all.shape[2]-2) <= 3/4
    assert (pr+sr)*t_calib <= 3/4
    pattern_warp = disp_warp_batch(pattern_all.astype(np.float32), -disp.astype(np.float32)+0.5, \
                        0.5*np.ones_like(disp).astype(np.float32))
    cam_imgs = torch.from_numpy(intensity).to(device).unsqueeze(0).unsqueeze(1) * (pattern_warp*pr + sr)
    cam_imgs[1:-1] *= t_exp/(pattern_all.shape[2]-2)
    cam_imgs[0] *= t_calib
    cam_imgs[-1] *= t_calib
    
    sigma_maps = torch.sqrt(cam_imgs * poiss_K/4096 + noise_level**2)
    cam_imgs += sigma_maps * torch.randn(size = cam_imgs.shape).to(device)
    cam_imgs = torch.clip(cam_imgs, 0.0, 1.0)
    cam_imgs = torch.round(cam_imgs * 4096).float()/4096
    cam_imgs[0] *= t_exp/(pattern_all.shape[2]-2)/t_calib
    cam_imgs[-1] *= t_exp/(pattern_all.shape[2]-2)/t_calib
    cam_imgs = torch.clip((cam_imgs - cam_imgs[:1])/(cam_imgs[-1:] - cam_imgs[:1]+1e-9), 0.0, 1.0)
    # cam_imgs = (cam_imgs - cam_imgs[:1])/(cam_imgs[-1:] - cam_imgs[:1]+1e-9)
    cam_imgs[...,:disp_clip] = -100
    return cam_imgs


def sl_simu_batch_cnc(pattern_all, intensity, disp, pr, sr, poiss_K, noise_level, t_exp, disp_clip):
    """
    Another normalization method through projecting codes and inverse codes
    In principle, this is equivalent to 
    However, it is easier to implement in real-world experiments
    """
    cam_imgs = []
    assert (pr+sr)*t_exp/(pattern_all.shape[2]) <= 3/4
    ## make sure no saturation
    proj_imgs = intensity[...,np.newaxis] * (pattern_all * pr + sr) 
    proj_imgs *= t_exp/(pattern_all.shape[2])
    proj_nimgs = intensity[...,np.newaxis] * ((1-pattern_all) * pr + sr) 
    proj_nimgs *= t_exp/(pattern_all.shape[2])
    cam_imgs = disp_warp_batch(proj_imgs.astype(np.float32), -disp.astype(np.float32)+0.5, \
                            0.5*np.ones_like(disp).astype(np.float32))
    cam_nimgs = disp_warp_batch(proj_nimgs.astype(np.float32), -disp.astype(np.float32)+0.5, \
                            0.5*np.ones_like(disp).astype(np.float32))
    
    sigma_maps = torch.sqrt(cam_imgs * poiss_K/4096 + noise_level**2)
    cam_imgs += sigma_maps * torch.randn(size = cam_imgs.shape).to(device)
    n_sigma_maps = torch.sqrt(cam_nimgs * poiss_K/4096 + noise_level**2)
    cam_nimgs += n_sigma_maps * torch.randn(size = cam_nimgs.shape).to(device)
    
    cam_imgs = torch.clip(cam_imgs, 0.0, 1.0)
    cam_imgs = torch.round(cam_imgs * 4096).float()/4096

    cam_nimgs = torch.clip(cam_nimgs, 0.0, 1.0)
    cam_nimgs = torch.round(cam_nimgs * 4096).float()/4096
    
    cam_imgs = cam_imgs - cam_nimgs
    cam_imgs[...,:disp_clip] = -100
    return cam_imgs


"""
compute cost volume to get best match correlation index
"""
def fast_sl_rec(code1d, code1d_bias, cam_imgs, resx, resy):
    corr = code1d_bias - 2*torch.matmul(code1d, cam_imgs)
    corr_idx = torch.argmin(corr, dim = 0).cpu().numpy().reshape(resx, resy)
    return corr_idx


def fast_sl_rec_cnc(code1d, cam_imgs, resx, resy):
    corr = torch.matmul(code1d, cam_imgs)
    corr_idx = torch.argmax(corr, dim = 0).cpu().numpy().reshape(resx, resy)
    del corr, code1d, cam_imgs
    return corr_idx