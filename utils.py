import torch 
import torchgeometry
import numpy as np 

from sklearn.metrics import roc_curve, auc 


import matplotlib.pyplot as plt 
import matplotlib as mpl
import matplotlib.colors as mcl
import matplotlib.cm as cm

def torch2npy(tensor):
    if len(tensor.shape) == 4:
        tensor = tensor.unsqueeze(0)
        
    npy = tensor.detach().cpu().numpy()
    return npy


def figure(img, reshape=(1, 2, 0), savefig=False, fname=None):
    if isinstance(img, torch.Tensor):
        img = torch2npy(img)
    img = np.transpose(img, axes=reshape)
    
    plt.figure(figsize = (6, 6))
    plt.imshow(img)
    
    if savefig:
        plt.savefig(f'{fname}.pdf', dpi=100)
    plt.show()


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
        
    elif isinstance(x, np.array):
        _min = np.min(x)
        _max = np.max(x)

    outs = (x - _min) / (_max - _min)
    return outs

    
def loss_ssim(preds, target):
    ssim = torchgeometry.losses.SSIM(11, reduction='mean')
    x_ssim = ssim(preds, target)
    
    return x_ssim



def segmap_figure(target, preds, norm=True, t=(1, 2, 0), save=None, path=None, alpha=0.5):
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
    
    if save:
        plt.savefig(path, dpi=200)
    plt.show()
