import torch 
import torchgeometry
import numpy as np 

from PIL import Image
import PIL

import random

from sklearn.metrics import roc_curve, auc 
from sklearn.manifold import TSNE
from sklearn.utils import shuffle

import matplotlib.pyplot as plt 
import matplotlib as mpl
import matplotlib.colors as mcl
import matplotlib.cm as cm


### Data Augmentation

def ShearX(img, magnitude):
    return img.transform(
        img.size, 
        Image.AFFINE, 
        (1, magnitude * random.choice([-1, 1]), 0, 0, 1, 0), 
        Image.BICUBIC, 
        fillcolor=FILLCOLOR,
    )

def TranslateX(img, magnitude):
    return img.transform(
        img.size, 
        Image.AFFINE, 
        (1, 0, magnitude * img.size[0] * random.choice([-1, 1]), 0, 1, 0), 
        fillcolor=FILLCOLOR,
    )

def Rorate(img, magnitude):
    rot = img.convert('RGBA').rotate(magnitude)
    return Image.composite(
        rot, Image.new('RGBA', rot.size, FILLCOLOR_RGBA), rot).convert(img.mode)

def Invert(img):
    return PIL.ImageOps.invert(img)

def Equalize(img):
    return PIL.ImageOps.equalize(img)

def AutoContrast(img):
    return PIL.ImageOps.autocontrast(img)
    
def Posterize(img, magnitude):
    return PIL.ImageOps.posterize(img, magnitude)

def Cutout(img, magnitude):
    if magnitude == 0.0:
        return img 
    w, h = img.size 
    xy = get_rand_bbox_coord(w, h, magnitude)

    img = img.copy()
    PIL.ImageDraw.Draw(img).rectangle(xy, fill=FILLCOLOR)
    return img


def torch2npy(tensor):
    if len(tensor.shape) == 4:
        tensor = tensor.unsqueeze(0)
        
    npy = tensor.detach().cpu().numpy()
    return npy

def ratio(src, dst):
    return (dst - src) / src * 100


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def roc_measure(target, preds):
    fpr, tpr, _ = roc_curve(target, preds, pos_label=1)
    auroc = metrics.auc(fpr, tpr)
    return auroc


def scaler(x):
    if isinstance(x, torch.Tensor):
        _min = torch.min(x)
        _max = torch.max(x)
        
    elif isinstance(x, np.ndarray):
        _min = np.min(x)
        _max = np.max(x)

    outs = (x - _min) / (_max - _min)
    return outs

    
def loss_ssim(preds, target):
    ssim = torchgeometry.losses.SSIM(11, reduction='mean')
    x_ssim = ssim(preds, target)
    
    return x_ssim


## Associated Figure 
def plot_imshow(img, reshape=(1, 2, 0), fname='img.pdf', savefig=False):
    if isinstance(img, torch.Tensor):
        img = torch2npy(img)
    img = np.transpose(img, axes=reshape)
    
    plt.figure(figsize = (6, 6))
    plt.imshow(img)
    
    if savefig:
        plt.savefig(f'{fname}.pdf', dpi=100)
    plt.show()

def plot_segmap(target, preds, norm=True, t=(1, 2, 0), fname='segmap.pdf', savefig=False, alpha=0.5):
    target = torch2npy(target, norm, t)
    preds = torch2npy(preds, norm, t)
    
    cmap = cm.hot
    vmax = preds.max()
    vmin = preds.min()
    norms = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    
    colormapping = cm.ScalarMappable(norm=norms, cmap=cmap)
    
    plt.colorbar(colormapping, ax=plt.gca())
    plt.imshow(target)
    plt.imshow(preds, cmap=cmap, alpha=alpha)
    
    if savefig:
        plt.savefig(fname, dpi=200)
    plt.show()


def plot_roc(labels, scores, fname='roc.pdf', modelname="", savefig=False):

    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
            lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver operating characteristic {modelname}')
    plt.legend(loc="lower right")
    if savefig:
        plt.savefig(fname)
    plt.show()

    return roc_auc

def plot_tsne(labels, embeds, fname='tsne.pdf', savefig=False):
    tsne = TSNE(n_components=2, verbose=1, perplexity=30, n_iter=500)
    embeds, labels = shuffle(embeds, labels)
    tsne_results = tsne.fit_transform(embeds)
    fig, ax = plt.subplots(1)
    colormap = ["b", "r", "c", "y"]

    ax.scatter(tsne_results[:,0], tsne_results[:,1], color=[colormap[l] for l in labels])
    if savefig:
        fig.savefig(fname)
    plt.close()


def plot_scatter(labels, ab_score, fname='scatter.pdf', savefig=False):
    an_idx = np.where(np.array(labels) == 1)[0]
    n_idx = np.where(np.array(labels) == 0)[0]
    
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(range(len(labels))[len(an_idx):], ab_score[n_idx], color=['steelblue']*len(norm_idx), marker='o')    
    ax.scatter(range(len(labels))[:len(an_idx)], ab_score[an_idx], color=['darkorange']*len(abnorm_idx), marker='^')
    
    if savefig:
        plt.savefig(fname, dpi=200)
    
    plt.show()
