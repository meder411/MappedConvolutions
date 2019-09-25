import torch
import torch.nn as nn


class BerHuLoss(nn.Module):
    '''Computes the BerHu loss as defined in:
    Laina, Iro, et al. "Deeper depth prediction with fully convolutional residual networks."
    3D Vision (3DV), 2016 Fourth International Conference on. IEEE, 2016.'''

    def __init__(self):
        super(BerHuLoss, self).__init__()

    def forward(self, pred, gt, mask=None):
        # Mask out invalid values initially
        abs_diff = mask * (pred - gt).abs()

        # Compute c based on 20% the maximal per-batch error (like Laina et al)
        c = 0.2 * abs_diff.max()

        # Compute the less-than-or-equal-to-c loss
        leq_c = abs_diff[abs_diff <= c]

        # Compute the greater-than-c loss
        gt_c = ((abs_diff[abs_diff > c]**2 + c**2) / (2 * c))

        # Return the average per pixel
        if mask is not None:
            return (leq_c.sum() + gt_c.sum()) / (mask > 0).sum().float()

        else:
            return (leq_c.sum() + gt_c.sum()) / gt.numel()
