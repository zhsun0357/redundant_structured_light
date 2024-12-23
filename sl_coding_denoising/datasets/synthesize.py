import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

import sys

from datasets.IO import *
import time

from scipy.io import loadmat


def disp_warp(x, dispx, dispy, interp_mode='bilinear', padding_mode='zeros'):
    x = torch.from_numpy(x).unsqueeze(0).unsqueeze(1).float()
    disp = torch.cat((torch.from_numpy(dispx).unsqueeze(0).unsqueeze(3), \
                      torch.from_numpy(dispy).unsqueeze(0).unsqueeze(3)), dim=3).float()
    assert x.size()[-2:] == disp.size()[1:3]
    B, C, H, W = x.size()
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

    return output[0, 0].numpy()


def sl_simu(pattern_all, intensity, disp, pr, sr, poiss_K, noise_level, t_exp):
    cam_imgs = []
    t_calib = 3 / 4 / (pr + sr)
    # assert (pr + sr) * t_exp / (pattern_all.shape[2] - 2) <= 3 / 4
    # assert (pr + sr) * t_calib <= 3 / 4
    ## make sure no saturation

    for ii in range(pattern_all.shape[2]):
        if (ii == 0) or (ii == pattern_all.shape[2] - 1):
            proj_img = intensity * (pattern_all[:, :, ii] * pr + sr) * t_calib
        else:
            proj_img = intensity * (pattern_all[:, :, ii] * pr + sr) * t_exp / (pattern_all.shape[2] - 2)

        cam_img = disp_warp(proj_img.astype(np.float32), -disp.astype(np.float32) + 0.5, \
                            0.5 * np.ones_like(disp).astype(np.float32))

        sigma_map = np.sqrt(cam_img * poiss_K / 4096 + noise_level ** 2)
        cam_img += sigma_map * np.random.normal(size=cam_img.shape)
        cam_img = np.clip(cam_img, 0.0, 1.0)
        cam_img = (cam_img * 4096).astype(np.int64).astype(np.float32) / 4096
        cam_imgs.append(cam_img)

    cam_imgs = np.asarray(cam_imgs).transpose(1, 2, 0)
    cam_imgs[:, :, 0] *= t_exp / (pattern_all.shape[2] - 2) / t_calib
    cam_imgs[:, :, -1] *= t_exp / (pattern_all.shape[2] - 2) / t_calib
    # pdb.set_trace()
    cam_imgs = np.clip((cam_imgs - cam_imgs[:, :, :1]) / (cam_imgs[:, :, -1:] - cam_imgs[:, :, :1] + 1e-9), -1.0, 1.0)
    cam_imgs[:, :100, :] = -100
    return cam_imgs


def disp_warp_batch(x, dispx, dispy, interp_mode='bilinear', padding_mode='zeros'):
    x = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(1).float().cuda()
    B, C, H, W = x.size()
    disp = torch.cat((torch.from_numpy(np.repeat(dispx[np.newaxis, ...], B, axis=0)).unsqueeze(3),
                      torch.from_numpy(np.repeat(dispy[np.newaxis, ...], B, axis=0)).unsqueeze(3)),
                     dim=3).float().cuda()

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


def sl_simu_batch(pattern_all, intensity, disp, pr, sr, poiss_K, noise_level, t_exp):
    t_calib = 3 / 4 / (pr + sr)
    assert (pr + sr) * t_exp / (pattern_all.shape[2] - 2) <= 3 / 4
    # assert (pr + sr) * t_calib <= 3 / 4
    ## make sure no saturation
    # proj_imgs = intensity[..., np.newaxis] * (pattern_all * pr + sr)
    # proj_imgs[..., 1:-1] *= t_exp / (pattern_all.shape[2] - 2)
    # proj_imgs[..., 0] *= t_calib
    # proj_imgs[..., -1] *= t_calib
    # cam_imgs = disp_warp_batch(proj_imgs.astype(np.float32), -disp.astype(np.float32) + 0.5,
    #                            0.5 * np.ones_like(disp).astype(np.float32))

    pattern_warp = disp_warp_batch(pattern_all.astype(np.float32), -disp.astype(np.float32) + 0.5,
                                   0.5 * np.ones_like(disp).astype(np.float32))
    cam_imgs = torch.from_numpy(intensity).cuda().unsqueeze(0).unsqueeze(1) * (pattern_warp * pr + sr)
    cam_imgs[1:-1] *= t_exp / (pattern_all.shape[2] - 2)
    cam_imgs[0] *= t_calib
    cam_imgs[-1] *= t_calib

    sigma_maps = torch.sqrt(cam_imgs * poiss_K / 4096 + noise_level ** 2).cuda()
    cam_imgs += sigma_maps * torch.randn_like(cam_imgs)
    cam_imgs = torch.clip(cam_imgs, 0.0, 1.0)
    cam_imgs = torch.round(cam_imgs * 4096).float() / 4096
    cam_imgs[0] *= t_exp / (pattern_all.shape[2] - 2) / t_calib
    cam_imgs[-1] *= t_exp / (pattern_all.shape[2] - 2) / t_calib
    cam_imgs = torch.clip((cam_imgs - cam_imgs[:1]) / (cam_imgs[-1:] - cam_imgs[:1] + 1e-9), -1.0, 1.0)
    cam_imgs[..., :100] = -100

    return cam_imgs


"""
fast pytorch implementation of cost volume
"""


# def fast_sl_rec(code1d, code1d_bias, cam_imgs, resx, resy):
#     corr = code1d_bias - 2 * torch.matmul(code1d, cam_imgs)
#     corr_idx = torch.argmin(corr, dim=0).cpu().numpy().reshape(resx, resy)
#     return corr_idx


def load_code():
    pattern_uncode = loadmat('data/pattern_uncode_binary_10.mat')['c_uncoded_vec']
    pattern_uncode = pattern_uncode.transpose()[np.newaxis, :, :]
    pattern_uncode_all = np.zeros((pattern_uncode.shape[0], pattern_uncode.shape[1], pattern_uncode.shape[2] + 2))
    pattern_uncode_all[..., 0] = np.zeros_like(pattern_uncode[..., 0])
    pattern_uncode_all[..., -1] = np.ones_like(pattern_uncode[..., 0])
    pattern_uncode_all[..., 1:-1] = pattern_uncode
    # pattern_uncode_all = pattern_uncode_all[:, :resy, :]

    pattern_golay = loadmat('data/golay2codes_n22_k10_num1024_gray.mat')['C']
    pattern_golay = pattern_golay[:, :pattern_uncode.shape[1]]
    print(np.unique(pattern_golay))
    pattern_golay = pattern_golay.transpose()[np.newaxis, :, :]
    pattern_golay_all = np.zeros((pattern_golay.shape[0], pattern_golay.shape[1], pattern_golay.shape[2] + 2))
    pattern_golay_all[..., 0] = np.zeros_like(pattern_golay[..., 0])
    pattern_golay_all[..., -1] = np.ones_like(pattern_golay[..., 0])
    pattern_golay_all[..., 1:-1] = pattern_golay
    # pattern_golay_all = pattern_golay_all[:, :resy, :]

    return pattern_uncode_all, pattern_golay_all


def noisy_synthesize(left_path, pattern, noise_level=0.001, t_exp=1.0, pr=1.0, sr=2.0,
                     poiss_K=6.0, resx=540, resy=729):
    # disp_file = '/media/data1/szh/FT3D/disparity/TRAIN/A/0743/left/0013.pfm'
    disp_file = left_path.replace("frames_cleanpass", "disparity").replace("png", "pfm")
    disp, _ = readPFM(disp_file)
    disp = disp.astype(np.int64).astype(np.float32)

    pattern = np.repeat(pattern, disp.shape[0], axis=0)
    pattern = pattern[:, :resy, :]

    """
    disparity is clipped to 0-99, might not be good?
    """
    disp = np.clip(disp[:, :resy], 0.0, 99.0)

    rgb = imageio.imread(left_path).astype(np.float32) / 255.0
    intensity = np.mean(rgb, axis=2)
    intensity = (intensity + 1.0) / 2.0
    # intensity = np.clip(intensity, 0.2, 1.0)

    intensity = intensity[:, :resy]

    # cam_imgs = sl_simu(pattern, intensity, disp, pr, sr, poiss_K, noise_level, t_exp)
    cam_imgs = sl_simu_batch(pattern, intensity, disp, pr, sr, poiss_K, noise_level, t_exp)

    return cam_imgs, disp


def get_patch_corr(pattern, cam_imgs, disp, top_k, crop=False, start_h=0, start_w=0, patch_size=256):
    cam_imgs = cam_imgs.squeeze().permute(1, 2, 0)

    resx, resy, c = cam_imgs.shape

    codes = pattern[0]

    code1d = torch.from_numpy(codes.astype(np.float32)).cuda()
    code1d_bias = torch.sum(code1d ** 2, dim=1).reshape(-1, 1)

    # x_np, y_np = np.meshgrid(np.arange(disp.shape[1]), np.arange(disp.shape[0]))

    y, x = torch.meshgrid(torch.arange(disp.shape[0]), torch.arange(disp.shape[1]))
    x = x[..., None].repeat(1, 1, top_k).cuda()

    if crop:
        cam_imgs = cam_imgs[start_h:start_h + patch_size,
                   start_w + 100:start_w + patch_size + 100, :].reshape(-1, c).permute(1, 0)

        corr = code1d_bias - 2 * torch.matmul(code1d, cam_imgs)
        corr = corr.reshape(-1, patch_size, patch_size).permute(1, 2, 0)
        top_corr, top_idx = torch.topk(corr, k=top_k, dim=-1, largest=False, sorted=True)

        # top_corr = top_corr.numpy()
        # top_idx = top_idx.numpy()

        top_disp = torch.clamp(
            x[start_h:start_h + patch_size, start_w + 100:start_w + patch_size + 100, :] - top_idx, 0.0, 99.0)

        disp = disp[start_h:start_h + patch_size, start_w + 100:start_w + patch_size + 100]

        # cam_imgs = cam_imgs.permute(1, 0).reshape(patch_size, patch_size, c).cpu().numpy()

    else:

        cam_imgs = cam_imgs.reshape(-1, c).permute(1, 0)
        corr = code1d_bias - 2 * torch.matmul(code1d, cam_imgs)
        corr = corr.reshape(-1, resx, resy).permute((1, 2, 0))
        top_corr, top_idx = torch.topk(corr, k=top_k, dim=-1, largest=False, sorted=True)

        # top_corr = top_corr.numpy()
        # top_idx = top_idx.numpy()

        top_disp = torch.clamp(x[:, 100:, :] - top_idx[:, 100:, :], 0.0, 99.0)

        # top_disp = x[:, 100:, :] - top_idx[:, 100:, :]

        """
        do not contain the last 100 columns since they are not in the overlapped region
        this is the major reason why we clip disparity from 0-99
        first scaling then clipping might be better for neural network trainings
        """

        disp = disp[:, 100:]
        top_corr = top_corr[:, 100:]

        # cam_imgs = cam_imgs.permute(1, 0).reshape(resx, resy, c).cpu().numpy()

    # fig, ax = plt.subplots(1, 3, figsize=(10, 5))
    # ax[0].imshow(top_disp.cpu().numpy()[..., 0], vmin=0.0, vmax=100.0)
    # ax[1].imshow(cam_imgs[:, :, 3], vmin=0.0, vmax=1.0, cmap='gray')
    # ax[2].imshow(disp, vmin=0.0, vmax=100.0)
    # # ax[2].imshow(disp, vmin=0.0, vmax=100.0)
    # # print('Error rate: ', np.sum(np.abs(x[:, 100:] - corr_idx[:, 100:] - \
    # #                                     disp[:, 100:]) >= 2.0) / disp[:, 100:].shape[0] / disp[:, 100:].shape[1])
    # # print("--- %s seconds ---" % (time.time() - start_time))
    # plt.show()

    return top_corr, top_disp, torch.from_numpy(disp[..., np.newaxis]).cuda()


def get_patch_corr_all(pattern, cam_imgs, disp):
    cam_imgs = cam_imgs.squeeze().permute(1, 2, 0)

    resx, resy, c = cam_imgs.shape

    codes = pattern[0]

    code1d = torch.from_numpy(codes.astype(np.float32)).cuda()
    code1d_bias = torch.sum(code1d ** 2, dim=1).reshape(-1, 1)

    # x_np, y_np = np.meshgrid(np.arange(disp.shape[1]), np.arange(disp.shape[0]))

    cam_imgs = cam_imgs.reshape(-1, c).permute(1, 0)

    corr = code1d_bias - 2 * torch.matmul(code1d, cam_imgs)
    corr = corr.reshape(-1, resx, resy).permute(1, 2, 0)
    disp = disp[:, 100:]

    return corr, disp
