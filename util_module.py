import numpy as np
import os
import pandas as pd
from collections import Counter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable, grad
from torchvision import models
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchvision import transforms, datasets

from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, auc, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.manifold import TSNE

# import captum
# from captum.attr import IntegratedGradients, Occlusion, LayerGradCam, LayerAttribution
# from captum.attr import visualization as viz

from typing import NamedTuple, List

from tqdm.notebook import tqdm
from matplotlib import pyplot as plt
from matplotlib.colors import CenteredNorm
from astropy.io import fits
from astropy.visualization import ImageNormalize, LogStretch

import copy
import glob
import re
import sys
import argparse
import json
import warnings
import os

from e2cnn import gspaces
from e2cnn import nn as e2nn

import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_require_grad(model, requires_grad=True):
    for param in model.parameters():
        param.requires_grad = requires_grad
       

def csa_loss(x, y, class_eq, margin=15, return_distance=False):
    dist = F.pairwise_distance(x, y)
#     print('positive distance', torch.mean(dist*(class_eq)))
#     print('negative distance', torch.mean(dist*(1-class_eq)))
    loss = class_eq * dist.pow(2)
    loss += (1 - class_eq) * (margin - dist).clamp(min=0).pow(2)
    if return_distance:
        return loss.mean(), torch.mean(dist*(class_eq)), torch.mean(dist*(1-class_eq))  # they could have diff number of classes inside batch
    else:
        return loss.mean()


def gradient_penalty(critic, h_s, h_t):
    # based on: https://github.com/caogang/wgan-gp/blob/master/gan_cifar10.py#L116
    alpha = torch.rand(h_s.size(0), 1).to(device)
    differences = h_s - h_t
    interpolates = h_s + (alpha * differences)
    interpolates = torch.stack([interpolates, h_s, h_t]).requires_grad_()
    # print(h_s.get_device(), h_t.get_device(), interpolates.get_device())

    preds = critic(interpolates)
    gradients = grad(preds, interpolates,
                     grad_outputs=torch.ones_like(preds),
                     retain_graph=True, create_graph=True)[0]
    gradient_norm = gradients.norm(2, dim=1)
    gradient_penalty = ((gradient_norm - 1)**2).mean()
    return gradient_penalty


class EarlyStopping:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def plot_roc_curves(evaluation_results, args, output_dir):
    plt.figure(figsize=(8, 8))

    plt.plot(evaluation_results['source']['fpr'], evaluation_results['source']['tpr'], 'r', label=f'{args.encoder} {evaluation_results["source"]["roc_auc"]:.3f}')

    for DA_method in evaluation_results.keys():
        if DA_method not in ['sim', 'source']:
            plt.plot(evaluation_results[DA_method]['fpr'], evaluation_results[DA_method]['tpr'], label=f'{DA_method} {evaluation_results[DA_method]["roc_auc"]:.3f}')
    
    plt.legend(loc='lower right', fontsize='x-large')
    plt.ylabel('True Positive Rate', fontsize=15)
    plt.xlabel('False Positive Rate', fontsize=15)
    plt.rcParams['xtick.labelsize'] = 15
    plt.rcParams['ytick.labelsize'] = 15
    plt.gca().set_aspect('equal')

    # Construct the path for saving the image
    results_dir = os.path.join(output_dir, 'evaluation_results')
    os.makedirs(results_dir, exist_ok=True)
    iteration_str = f'_{args.iter}' if hasattr(args, 'iter') else ''
    image_file = os.path.join(results_dir, f'ROC_curve_{args.encoder}_{args.src_dataset}_{args.subsample}{args.savename}{iteration_str}.png')

    plt.savefig(image_file, dpi=120)
    plt.close()
    print(f"ROC curve saved to {image_file}")


def plot_roc_curves_log(evaluation_results, args, output_dir):
    plt.figure(figsize=(10, 8))
    plt.plot(evaluation_results['source']['fpr'], evaluation_results['source']['tpr'], 'r', label=f'{args.encoder} {evaluation_results["source"]["roc_auc"]:.3f}')
    for DA_method in evaluation_results.keys():
        if DA_method not in ['sim', 'source']:
            plt.plot(evaluation_results[DA_method]['fpr'], evaluation_results[DA_method]['tpr'], label=f'{DA_method} {evaluation_results[DA_method]["roc_auc"]:.3f}')
    plt.legend(loc='lower right', fontsize='x-large')
    plt.ylabel('True Positive Rate', fontsize=15)
    plt.xlabel('False Positive Rate', fontsize=15)
    plt.rcParams['xtick.labelsize'] = 15
    plt.rcParams['ytick.labelsize'] = 15
    plt.xlim(1e-3, 1)
    plt.xscale('log')
    plt.gca().set_aspect('auto')

    # Construct the path for saving the image
    results_dir = os.path.join(output_dir, 'evaluation_results')
    os.makedirs(results_dir, exist_ok=True)
    iteration_str = f'_{args.iter}' if hasattr(args, 'iter') else ''
    image_file = os.path.join(results_dir, f'ROC_curve_log_{args.encoder}_{args.src_dataset}_{args.subsample}{args.savename}{iteration_str}.png')

    plt.savefig(image_file, dpi=120)
    plt.close()

def plot_precision_recall_curves(evaluation_results, args, output_dir):
    plt.figure(figsize=(8, 8))
    for method_name in evaluation_results.keys():
        p1 = plt.plot(evaluation_results[method_name]['pr_threshold'], evaluation_results[method_name]['precision'][:-1], label=f'{method_name}')
        plt.plot(evaluation_results[method_name]['pr_threshold'], evaluation_results[method_name]['recall'][:-1], ls='--', color=p1[0].get_color())
    plt.legend()
    plt.ylabel('Precision/Recall', fontsize=15)
    plt.xlabel('Threshold', fontsize=15)
    plt.rcParams['xtick.labelsize'] = 15
    plt.rcParams['ytick.labelsize'] = 15
    plt.gca().set_aspect('equal')

    # Construct the path for saving the image
    results_dir = os.path.join(output_dir, 'evaluation_results')
    os.makedirs(results_dir, exist_ok=True)
    iteration_str = f'_{args.iter}' if hasattr(args, 'iter') else ''
    image_file = os.path.join(results_dir, f'PrecRec_curve_{args.encoder}_{args.src_dataset}_{args.subsample}{args.savename}{iteration_str}.png')

    plt.savefig(image_file, dpi=120)
    plt.close()
    print(f"Precision-Recall curve saved to {image_file}")

# import math
# from torch import default_generator, randperm
# from torch._utils import _accumulate
# from torch.utils.data.dataset import Subset
# import warnings

# def random_split(dataset, lengths,
#                  generator=default_generator):
#     r"""
#     Randomly split a dataset into non-overlapping new datasets of given lengths.

#     If a list of fractions that sum up to 1 is given,
#     the lengths will be computed automatically as
#     floor(frac * len(dataset)) for each fraction provided.

#     After computing the lengths, if there are any remainders, 1 count will be
#     distributed in round-robin fashion to the lengths
#     until there are no remainders left.

#     Optionally fix the generator for reproducible results, e.g.:

#     >>> random_split(range(10), [3, 7], generator=torch.Generator().manual_seed(42))
#     >>> random_split(range(30), [0.3, 0.3, 0.4], generator=torch.Generator(
#     ...   ).manual_seed(42))

#     Args:
#         dataset (Dataset): Dataset to be split
#         lengths (sequence): lengths or fractions of splits to be produced
#         generator (Generator): Generator used for the random permutation.
#     """
#     if math.isclose(sum(lengths), 1) and sum(lengths) <= 1:
#         subset_lengths: List[int] = []
#         for i, frac in enumerate(lengths):
#             if frac < 0 or frac > 1:
#                 raise ValueError(f"Fraction at index {i} is not between 0 and 1")
#             n_items_in_split = int(
#                 math.floor(len(dataset) * frac)  # type: ignore[arg-type]
#             )
#             subset_lengths.append(n_items_in_split)
#         remainder = len(dataset) - sum(subset_lengths)  # type: ignore[arg-type]
#         # add 1 to all the lengths in round-robin fashion until the remainder is 0
#         for i in range(remainder):
#             idx_to_add_at = i % len(subset_lengths)
#             subset_lengths[idx_to_add_at] += 1
#         lengths = subset_lengths
#         for i, length in enumerate(lengths):
#             if length == 0:
#                 warnings.warn(f"Length of split at index {i} is 0. "
#                               f"This might result in an empty dataset.")

#     # Cannot verify that dataset is Sized
#     if sum(lengths) != len(dataset):    # type: ignore[arg-type]
#         raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

#     indices = randperm(sum(lengths), generator=generator).tolist()  # type: ignore[call-overload]
#     return [Subset(dataset, indices[offset - length : offset]) for offset, length in zip(_accumulate(lengths), lengths)]
