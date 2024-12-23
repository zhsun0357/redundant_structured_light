import math
import os

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from models.network import disp_net
from datasets.SL_dataset import Training_dataset, Testing_dataset
from torch.utils.tensorboard import SummaryWriter

from engine import train_one_epoch, evaluate
from models.loss import pred_loss

import utils.misc as utils


def get_args_parser():
    parser = argparse.ArgumentParser('Set args', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--patch_size', default=128, type=int)
    parser.add_argument('--top_k', default=10, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default=50, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # dataset parameters
    parser.add_argument('--dataset_path')

    parser.add_argument('--output_dir', default='checkpoints/',
                        help='path where to save, empty for no saving')

    parser.add_argument('--result_dir',
                        help='path where to save, empty for no saving')

    parser.add_argument('--log_dir', default='logs/',
                        help='path where to save, empty for no saving')

    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)

    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--eval', action='store_true')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')

    parser.add_argument('--num_workers', default=4, type=int)

    return parser


def main(args):
    device = torch.device(args.device)

    torch.backends.cuda.matmul.allow_tf32 = False

    writer = SummaryWriter(args.log_dir)

    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.multiprocessing.set_start_method('spawn')

    dataset_train = Training_dataset(img_folder=os.path.join(args.dataset_path, "TRAIN"),
                                     patch_size=args.patch_size, top_k=args.top_k)
    dataset_val = Testing_dataset(img_folder=os.path.join(args.dataset_path, "TEST"), top_k=args.top_k)

    train_data_loader = DataLoader(dataset=dataset_train,
                                   num_workers=args.num_workers,
                                   batch_size=args.batch_size,
                                   shuffle=True,
                                   pin_memory=False)

    val_loader = DataLoader(dataset=dataset_val,
                            num_workers=1,
                            batch_size=1,
                            shuffle=False,
                            pin_memory=False)

    criterion = pred_loss()
    output_dir = Path(args.output_dir)

    model = disp_net(top_k=args.top_k).cuda()

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param = list(model.parameters())

    print('number of params:', n_parameters)

    optimizer = torch.optim.Adam(param,
                                 # [{'params': param}, {'params': model.scale_factors, 'lr': args.lr}],
                                 lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay)

    per_epoch_iteration = len(train_data_loader) // 4
    total_iteration = per_epoch_iteration * args.epochs

    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop, gamma=0.5)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_iteration, eta_min=1e-6)

    if args.resume:

        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

    if args.eval:
        mean_error = evaluate(model, val_loader, device, args.result_dir)
        print("mean_error: " + str(mean_error))
        return

    print("Start training")
    start_time = time.time()

    min_mean_error = math.inf

    for epoch in range(args.start_epoch, args.epochs):

        train_one_epoch(model, criterion, lr_scheduler, train_data_loader, optimizer, device, epoch, writer,
                        args.batch_size)

        if (epoch + 1) % 1 == 0:
            mean_error = evaluate(model, val_loader, device, args.result_dir)

            writer.add_scalar('testing/mean_error', mean_error, epoch)

            if args.output_dir:
                checkpoint_paths = [output_dir / 'checkpoint.pth']
                # extra checkpoint before LR drop and every 100 epochs
                if min_mean_error > mean_error:
                    min_mean_error = mean_error
                    checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')

                for checkpoint_path in checkpoint_paths:
                    utils.save_on_master({
                        'model': model.state_dict(),
                        'network': [model],
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'args': args,
                    }, checkpoint_path)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.log_dir:
        Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    main(args)
