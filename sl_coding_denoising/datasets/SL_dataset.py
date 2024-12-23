# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import json
import os
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
import glob
import torch.utils.data as data
import torch
import torch.nn.functional as F

from scipy.interpolate import interp1d

from scipy.io import loadmat

import matplotlib.pyplot as plt

from datasets.synthesize import *


class Training_dataset(data.Dataset):
    def __init__(self, img_folder, patch_size, top_k):
        self.img_folder = img_folder
        self.image_path_list = glob.glob(os.path.join(img_folder, "*", "*", "left", "*.png"))
        self.pattern_uncode_all, self.pattern_golay_all = load_code()
        self.patch_size = patch_size
        self.top_k = top_k

        print(len(self.image_path_list))

    def __getitem__(self, idx):
        img_path = self.image_path_list[idx]

        # if np.random.rand() < 0.5:
        #     top_corr, top_idx, disp = noisy_synthesize(img_path, self.pattern_golay_all, top_k=self.top_k)
        # else:
        #     top_corr, top_idx, disp = noisy_synthesize(img_path, self.pattern_uncode_all, top_k=self.top_k)

        # projector_light = 10 ** np.random.uniform(np.log10(0.3), np.log10(0.8))
        projector_light = 10 ** np.random.uniform(np.log10(0.05), np.log10(2.0))
        poiss_K = 10 ** np.random.uniform(np.log10(2.0), np.log10(6.0))
        cam_imgs, disp = noisy_synthesize(img_path, self.pattern_golay_all, pr=projector_light, poiss_K=poiss_K)

        h_start, w_start = self.random_crop(cam_imgs, self.patch_size)

        top_corr, top_idx, disp = get_patch_corr(self.pattern_golay_all, cam_imgs, disp, self.top_k, crop=True,
                                                 start_h=h_start, start_w=w_start, patch_size=self.patch_size)

        # # sorter = torch.argsort(top_idx, -1)
        #
        # index = torch.searchsorted(torch.flip(top_idx, [-1]), disp, right=True)
        #
        # a = torch.from_numpy(np.array([1, 2, 3, 4]))
        #
        # index = torch.searchsorted(a, torch.from_numpy(np.array([1])), right=False)

        return top_corr.permute(2, 0, 1), top_idx.permute(2, 0, 1) / 100, disp.permute(2, 0, 1) / 100

    def __len__(self):
        return len(self.image_path_list)

    def random_crop(self, top_corr, size):
        c, _, h, w = top_corr.shape

        h_start = np.random.randint(0, h - size)
        w_start = np.random.randint(0, w - size - 100)

        return h_start, w_start


class Testing_dataset(data.Dataset):
    def __init__(self, img_folder, top_k):
        self.img_folder = img_folder
        self.image_path_list = glob.glob(os.path.join(img_folder, "*", "*", "left", "*.png"))
        self.pattern_uncode_all, self.pattern_golay_all = load_code()
        self.top_k = top_k
        # self.patch_size = patch_size
        print(len(self.image_path_list))

    def __getitem__(self, idx):
        img_path = self.image_path_list[idx]

        cam_imgs, disp = noisy_synthesize(img_path, self.pattern_golay_all)

        top_corr, top_idx, disp = get_patch_corr(self.pattern_golay_all, cam_imgs, disp, self.top_k)

        return top_corr.permute(2, 0, 1), top_idx.permute(2, 0, 1) / 100, disp.permute(2, 0, 1) / 100, img_path

    def __len__(self):
        return len(self.image_path_list)


def ToTensor(image):
    return torch.from_numpy(image).permute(2, 0, 1).float()
