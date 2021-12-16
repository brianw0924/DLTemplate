import os
import numpy as np
import glob
import random
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image


tfm = transforms.Compose([
    transforms.Resize((144,192)),
    transforms.Grayscale(),
    transforms.ToTensor(),
])

class Mydataset(data.Dataset):
    def __init__(self, data_dir):

    def __getitem__(self, index):

    def __len__(self):

def Myloader(args):

    train_dataset = 

    val_dataset = 

    train_loader = data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True,
    )
    return train_loader, val_loader
