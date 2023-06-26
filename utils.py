from PIL import Image
from tqdm.auto import tqdm 
import glob, os, sys

from sklearn.model_selection import train_test_split

from settings import * 

import torch 
from torch import nn
from torch.utils.data import DataLoader, Dataset

class MVTecDataset(nn.Module):
    def __init__(self, args, paths):
        self.paths = paths 
        self.args = args
        self.labels = args.labels 
        self.H, self.W = args.height, args.width
        self.transforms = albumentations.Compose([
        albumentations.Resize(self.H, self.W)
        ])

        for path in self.paths:
            imgs = scaler(np.array(Image.open(path)))
            if self.labels in ['zipper', 'screw', 'grid']:
                imgs = np.expand_dims(imgs, axis=-1)

            self.imgs.append(imgs)
            if 'good' in path:
                self.labels.append(0)
            else:
                self.labels.append(1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        imgs = self.imgs[idx]
        labels = self.labels[idx]

        imgs = self.transforms(image=imgs)['image']
        imgs = np.transpose(imgs, (2, 0, 1))

        return (
                torch.tensor(imgs, dtype=torch.float),
                torch.tensor(labels, dtype=torch.float)
        )



def get_loader(args, mode='train'):
    types = 'train' if mode == 'train' else 'test'
    paths = glob.glob(os.path.join(DATA_DIR, f'{args.labels}/{types}/*/*.png'))
    d_set = MVTecDataset(args, paths)

    return DataLoader(d_set, batch_size=args.batch_size, shuffle=args.shuffle)



def scaler(x):
  _min = np.min(x)
  _max = np.max(x)

  outs = (x-_min) / (_max - _min)
  return outs 
  
