import os
import importlib
import argparse
import numpy as np
import random
import matplotlib.pyplot as plt
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torch.utils.data
import torch.backends.cudnn
import torchvision.utils
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.utils import save_image
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils import GradualWarmupScheduler, Set_seed
from mynet import Mynet
from myloader import Myloader

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=)
    parser.add_argument('--out_dir', type=str, default=)
    parser.add_argument('--seed', type=int, default=)
    parser.add_argument('--num_workers', type=int, default=)
    parser.add_argument('--total_steps', type=int, default=)
    parser.add_argument('--val_steps', type=int, default=)
    parser.add_argument('--warmup_steps', type=int, default=)
    parser.add_argument('--batch_size', type=int, default=)
    parser.add_argument('--accumulate', type=int, default=)
    parser.add_argument('--learning_rate', type=float, default=)
    parser.add_argument('--weight_decay', type=float, default=)

    args = parser.parse_args()

    assert os.path.exists(args.dataset)

    return args

def validation(args, step, model, criterion, val_loader, device):
    model.eval()


    val_loss = 0

    for x, y in val_loader:
        with torch.no_grad():
            pred = model(x.to(device))
        loss = criterion(pred, y.to(device))
        val_loss+=(loss.item())

    val_loss/= len(val_loader)
    
    return val_loss



def train(args, model, optimizer, scheduler, criterion, train_loader, val_loader, device):
    model.train()


    train_loss = 0
    error = 1e10

    train_iterator = iter(train_loader)

    count = 0
    for step in tqdm(range(args.total_steps)):

        try:
            x, y = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            x, y = next(train_iterator)
        count+=x.shape[0]

        pred = model(x.to(device))
        loss = criterion(pred, y.to(device))
        loss = loss / args.accumulate
        loss.backward()

        train_loss+= loss.item()

        if (step+1) % args.accumulate == 0:
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
        
        if (step+1) % args.val_steps == 0:
            
            # TRAINING RECORD
            train_loss = train_loss * args.accumulate / args.val_steps
    
            #VALIDATION
            val_loss = validation(args, step+1, model, criterion, val_loader, device)

            # PRINT
            tqdm.write(f"Step: {str(step+1).zfill(8)} | Train: {train_loss} | Validation: {val_loss}")

            # SAVE MODEL
            if val_loss < error:
                tqdm.write(f"Save model at step {str(step+1).zfill(6)}")
                torch.save(model.state_dict(), os.path.join(args.out_dir, f'model_state.pth'))
                error = val_loss

            train_loss = 0
            count = 0


def main():

    # DEVICE
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    # ARGUMENTS
    args = parse_args()

    # RANDOM SEED
    Set_seed(args.seed)

    # OUTPUT DIRECTORY
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    # RECORD ARGS
    args_json = os.path.join(args.outdir, 'args.json')
    with open(args_json, 'w') as fout:
        json.dump(vars(args), fout, indent=2)

    # DATASET (TEyeD)
    print("Preparing data ...")
    train_loader, val_loader = Myloader(args)

    # MODEL & LOSS
    model = Mynet()
    model.to(device)
    criterion = nn.MSELoss()

    # OPTIMIZER
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # SCHEDULER
    scheduler = CosineAnnealingLR(optimizer, T_max=(args.total_steps-args.warmup_steps)/args.accumulate)
    scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=args.warmup_steps//args.accumulate, after_scheduler=scheduler)
    optimizer.zero_grad()
    optimizer.step()

    # TRAINING
    train(args, model, optimizer, scheduler_warmup, criterion, train_loader, val_loader, device, log)

if __name__ == '__main__':
    main()
