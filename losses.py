import torch
import torch.nn.functional as F
from torch import nn as nn
from torch.autograd import Variable
from torch.nn import MSELoss, SmoothL1Loss, L1Loss
import numpy as np

import os
import pytorch_lightning as pl
#seed = int(os.environ.get("PL_GLOBAL_SEED"))
#pl.utilities.seed.seed_everything(seed=seed)

class SyncFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, tensor):
        ctx.batch_size = tensor.shape[0]

        gathered_tensor = [torch.zeros_like(tensor) for _ in range(torch.distributed.get_world_size())]

        torch.distributed.all_gather(gathered_tensor, tensor)
        gathered_tensor = torch.cat(gathered_tensor, 0)

        return gathered_tensor

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        torch.distributed.all_reduce(grad_input, op=torch.distributed.ReduceOp.SUM, async_op=False)

        idx_from = torch.distributed.get_rank() * ctx.batch_size
        idx_to = (torch.distributed.get_rank() + 1) * ctx.batch_size
        return grad_input[idx_from:idx_to]

class PixelwiseContrastiveLoss(torch.nn.Module):
    '''
    The Pixel wise Contrastive Loss
    '''
    def __init__(self,
                 neg_multiplier,
                 n_max_pos=128,
                 boundary_aware=False,
                 boundary_loc='both',
                 sampling_type='full',
                 temperature=0.1):
        super(PixelwiseContrastiveLoss, self).__init__()
        self.cosine = nn.CosineSimilarity(dim=-1, eps=1e-8)
        self.n_max_pos = n_max_pos
        self.n_max_neg = n_max_pos * neg_multiplier
        self.boundary_aware = boundary_aware
        self.boundary_loc = boundary_loc
        self.sampling_type = sampling_type # 'full', 'random', 'linear'
        self.temperature = temperature

    def extract_boundary(self, real_label, is_pos=True):
        if not is_pos:
            real_label = 1 - real_label

        gt_b = F.max_pool2d(1 - real_label, kernel_size=5, stride=1, padding=2)
        gt_b_in = 1 - gt_b
        gt_b -= 1 - real_label
        return gt_b, gt_b_in

    def sample_pixels(self, label, n):
        cand_pixels = torch.nonzero(label)
        sample_idx = torch.randperm(cand_pixels.shape[0])[:n]
        sample_pixels = cand_pixels[sample_idx]
        return sample_pixels

    def _sample_balance(self, cand_pixels, n):
        batch_idx = cand_pixels[:,0]
        bs = batch_idx.max() + 1
        n_per_sample = n // bs
        sample_idx = []
        accum = 0
        for b in range(bs):
            n_features = int((batch_idx == b).sum().cpu())
            temp_idx = np.random.permutation(n_features)[:n_per_sample] + accum
            sample_idx += temp_idx.tolist()
            accum += n_features
        return sample_idx

    def split_n(self, n, boundary_type, limit, split_param=None):
        if n < limit:
            valid_n = n
        else:
            valid_n = limit

        if boundary_type == 'full':
            return valid_n, n-valid_n
        elif boundary_type == 'exclude':
            return 0, valid_n
        elif boundary_type == 'random':
            n_bd = int(torch.rand(1) * valid_n)
            n_not_bd = valid_n - n_bd
            return n_bd, n_not_bd
        elif boundary_type == 'linear':
            current_epoch, max_epoch = split_param
            n_bd = int(current_epoch/max_epoch * valid_n)
            n_not_bd = valid_n - n_bd
            return n_bd, n_not_bd
        elif boundary_type == 'fixed':
            n_bd = int(0.2 * valid_n)
            n_not_bd = valid_n - n_bd
            return n_bd, n_not_bd

    def forward(self, predict_seg_map, real_label, split_param=None, vector="embedding"):
        if self.boundary_aware:
            if self.boundary_loc == 'pos':
                pos_b, pos_b_in = self.extract_boundary(real_label)
                n_pos_bd, n_pos_not_bd = self.split_n(self.n_max_pos,
                                                      self.sampling_type,
                                                      limit=pos_b.sum(),
                                                      split_param=split_param)
                neg_b, neg_b_in = 1-real_label, 1-real_label
                n_neg_bd, n_neg_not_bd = 0, self.n_max_neg
            elif self.boundary_loc == 'neg':
                neg_b, neg_b_in = self.extract_boundary(real_label, is_pos=False)
                n_neg_bd, n_neg_not_bd = self.split_n(self.n_max_neg,
                                                      self.sampling_type,
                                                      limit=neg_b.sum(),
                                                      split_param=split_param)
                pos_b, pos_b_in = real_label, real_label
                n_pos_bd, n_pos_not_bd = 0, self.n_max_pos
            elif self.boundary_loc == 'both':
                pos_b, pos_b_in = self.extract_boundary(real_label)
                neg_b, neg_b_in = self.extract_boundary(real_label, is_pos=False)
                n_pos_bd, n_pos_not_bd = self.split_n(self.n_max_pos,
                                                      self.sampling_type,
                                                      limit=pos_b.sum(),
                                                      split_param=split_param)
                n_neg_bd = n_pos_bd
                n_neg_not_bd = self.n_max_neg - n_neg_bd
        else:
            pos_b, pos_b_in = real_label, real_label
            neg_b, neg_b_in = 1-real_label, 1-real_label
            n_pos_bd, n_pos_not_bd = 0, self.n_max_pos
            n_neg_bd, n_neg_not_bd = 0, self.n_max_neg

        # sample positive pixels
        pos_b_pixels = self.sample_pixels(pos_b, n_pos_bd)
        pos_b_in_pixels = self.sample_pixels(pos_b_in, n_pos_not_bd)
        pos_pixels = torch.cat((pos_b_pixels, pos_b_in_pixels), dim=0).detach()
        pos_pixels = tuple(pos_pixels.t())

        # sample negative pixels
        neg_b_pixels = self.sample_pixels(neg_b, n_neg_bd)
        neg_b_in_pixels = self.sample_pixels(neg_b_in, n_neg_not_bd)
        neg_pixels = torch.cat((neg_b_pixels, neg_b_in_pixels), dim=0).detach()
        neg_pixels = tuple(neg_pixels.t())

        if vector == "embedding" or vector == "first"  :
            positive_logits = predict_seg_map[pos_pixels[0], :, pos_pixels[2], pos_pixels[3]]
            negative_logits = predict_seg_map[neg_pixels[0], :, neg_pixels[2], neg_pixels[3]]
        elif vector == "second" :
            positive_logits = predict_seg_map[pos_pixels[0], :, pos_pixels[2], pos_pixels[3]]
            positive_logits = positive_logits[:int(len(positive_logits)/4)]

            negative_logits = predict_seg_map[neg_pixels[0], :, neg_pixels[2], neg_pixels[3]]
            negative_logits = negative_logits[:int(len(negative_logits)/4)]
        elif vector == "third" :
            positive_logits = predict_seg_map[pos_pixels[0], :, pos_pixels[2], pos_pixels[3]]
            positive_logits = positive_logits[:int(len(positive_logits)/16)]

            negative_logits = predict_seg_map[neg_pixels[0], :, neg_pixels[2], neg_pixels[3]]
            negative_logits = negative_logits[:int(len(negative_logits)/16)]
        elif vector == "four" :
            positive_logits = predict_seg_map[pos_pixels[0], :, pos_pixels[2], pos_pixels[3]]
            positive_logits = positive_logits[:int(len(positive_logits)/64)]

            negative_logits = predict_seg_map[neg_pixels[0], :, neg_pixels[2], neg_pixels[3]]
            negative_logits = negative_logits[:int(len(negative_logits)/64)]

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            all_positive_logits = SyncFunction.apply(positive_logits)
            all_negative_logits = SyncFunction.apply(negative_logits)
        else:
            all_positive_logits = positive_logits
            all_negative_logits = negative_logits

        pos_nll = self._compute_loss(positive_logits,
                                     all_positive_logits,
                                     all_negative_logits)

        return pos_nll

    def _compute_loss(self, pos, all_pos, all_negs):
        positive_sim = self.cosine(pos.unsqueeze(1),
                                   all_pos.unsqueeze(0))
        exp_positive_sim = torch.exp(positive_sim/self.temperature)
        off_diagonal = torch.ones(exp_positive_sim.shape).type_as(exp_positive_sim)
        off_diagonal = off_diagonal.fill_diagonal_(0.0)
        exp_positive_sim = exp_positive_sim * off_diagonal
        positive_row_sum = torch.sum(exp_positive_sim, dim=1)

        negative_sim = self.cosine(pos.unsqueeze(1),
                                   all_negs.unsqueeze(0))
        exp_negative_sim = torch.exp(negative_sim/self.temperature)
        negative_row_sum = torch.sum(exp_negative_sim, dim=1)

        likelihood = positive_row_sum / (positive_row_sum + negative_row_sum)
        nll = -torch.log(likelihood).mean()

        return nll


def compute_per_channel_dice(input, target, epsilon=1e-5, ignore_index=None, weight=None):
    # assumes that input is a normalized probability

    # input and target shapes must match
    assert input.size() == target.size(), "'input' and 'target' must have the same shape"

    # mask ignore_index if present
    if ignore_index is not None:
        mask = target.clone().ne_(ignore_index)
        mask.requires_grad = False

        input = input * mask
        target = target * mask

    input = flatten(input)
    target = flatten(target)

    input = input.float()
    target = target.float()
    # Compute per channel Dice Coefficient
    intersect = (input * target).sum(-1)
    if weight is not None:
        intersect = weight * intersect

    denominator = (input + target).sum(-1)
    return 2. * intersect / denominator.clamp(min=epsilon)


class DiceLoss(nn.Module):
    """Computes Dice Loss, which just 1 - DiceCoefficient described above.
    Additionally allows per-class weights to be provided.
    """

    def __init__(self, epsilon=1e-5, weight=None, ignore_index=None, sigmoid_normalization=True,
                 skip_last_target=False):
        super(DiceLoss, self).__init__()
        self.epsilon = epsilon
        self.register_buffer('weight', weight)
        self.ignore_index = ignore_index
        # The output from the network during training is assumed to be un-normalized probabilities and we would
        # like to normalize the logits. Since Dice (or soft Dice in this case) is usually used for binary data,
        # normalizing the channels with Sigmoid is the default choice even for multi-class segmentation problems.
        # However if one would like to apply Softmax in order to get the proper probability distribution from the
        # output, just specify sigmoid_normalization=False.
        if sigmoid_normalization:
            self.normalization = nn.Sigmoid()
        else:
            self.normalization = nn.Softmax(dim=1)
        # if True skip the last channel in the target
        self.skip_last_target = skip_last_target

    def forward(self, input, target):
        # get probabilities from logits
        input = self.normalization(input)
        if self.weight is not None:
            weight = Variable(self.weight, requires_grad=False)
        else:
            weight = None

        if self.skip_last_target:
            target = target[:, :-1, ...]

        per_channel_dice = compute_per_channel_dice(input, target, epsilon=self.epsilon, ignore_index=self.ignore_index,
                                                    weight=weight)
        # Average the Dice score across all channels/classes
        return torch.mean(1. - per_channel_dice)

def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.view(C, -1)
