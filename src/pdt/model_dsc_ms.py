"""DSC/MS super-resolution CNN (PLAN.md section 5.3).

Modern PyTorch re-implementation of the Fukami-Fukagata-Taira DSC/MS
architecture, independently written from the plan's frozen structure table
(fixed spec: 140,203 trainable parameters).

Interface contract of THIS project (not a claim of reproducing the authors'
data pipeline): the input is the normalized LR observation replicated by
region-constant copy to the HR size (2 channels), the output is the
normalized HR field (2 channels, linear head).  ALL convolutions use
circular padding (periodic plane).  External average pooling defines the
observation; internal MAX pooling only extracts features (PLAN.md 5.3).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

EXPECTED_PARAM_COUNT = 140203


class CircConv2d(nn.Module):
    """k x k conv with circular (periodic) padding and bias."""

    def __init__(self, cin, cout, k):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, bias=True)
        self.k = k

    def forward(self, x):
        p = self.k // 2
        if self.k % 2 == 1:
            x = F.pad(x, (p, p, p, p), mode="circular")
        else:  # not used by the frozen spec; kept explicit and fail-loud
            raise NotImplementedError("even kernels not in spec")
        return self.conv(x)


def _dsc_level(cin, cout=32):
    """3x3 -> 3x3 block with ReLU between (level body)."""
    return nn.Sequential(
        CircConv2d(cin, cout, 3),
        nn.ReLU(inplace=True),
        CircConv2d(cout, cout, 3),
        nn.ReLU(inplace=True),
    )


def _ms_branch(k):
    """One multi-scale branch: 2->16->8->8, kernels k x k, ReLU between."""
    return nn.Sequential(
        CircConv2d(2, 16, k),
        nn.ReLU(inplace=True),
        CircConv2d(16, 8, k),
        nn.ReLU(inplace=True),
        CircConv2d(8, 8, k),
        nn.ReLU(inplace=True),
    )


class DSCMS(nn.Module):
    """DSC/MS network with circular boundaries.

    DSC: 4 levels; level i input is the nearest-upsampled previous level
    concatenated with the input max-pooled by 8/4/2/1 respectively; each
    level body is conv3 34->32->32 (first level 2->32->32) + nearest x2 up
    (except level 4).
    MS:  three branches k=5,9,13 (2->16->8->8), fusion cat(3x8 + input 2)
    -> conv7 26->8 -> conv5 8->3.
    Out: cat(DSC 32, MS 3) -> conv3 35->2 linear.
    """

    def __init__(self):
        super().__init__()
        self.l1 = _dsc_level(2, 32)
        self.l2 = _dsc_level(34, 32)
        self.l3 = _dsc_level(34, 32)
        self.l4 = _dsc_level(34, 32)
        self.ms5 = _ms_branch(5)
        self.ms9 = _ms_branch(9)
        self.ms13 = _ms_branch(13)
        self.ms_fuse1 = nn.Sequential(CircConv2d(26, 8, 7), nn.ReLU(inplace=True))
        self.ms_fuse2 = nn.Sequential(CircConv2d(8, 3, 5), nn.ReLU(inplace=True))
        self.head = CircConv2d(35, 2, 3)

    def forward(self, x):
        # x: (B, 2, Ns, Nx) normalized LR replicated to HR size
        d = self.l1(F.max_pool2d(x, 8))
        d = F.interpolate(d, scale_factor=2, mode="nearest")
        d = self.l2(torch.cat([d, F.max_pool2d(x, 4)], dim=1))
        d = F.interpolate(d, scale_factor=2, mode="nearest")
        d = self.l3(torch.cat([d, F.max_pool2d(x, 2)], dim=1))
        d = F.interpolate(d, scale_factor=2, mode="nearest")
        d = self.l4(torch.cat([d, x], dim=1))                     # (B,32,Ns,Nx)
        m = self.ms_fuse1(torch.cat([self.ms5(x), self.ms9(x), self.ms13(x), x], dim=1))
        m = self.ms_fuse2(m)                                       # (B,3,Ns,Nx)
        return self.head(torch.cat([d, m], dim=1))                 # (B,2,Ns,Nx) linear

    def param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def replicate_lr_to_hr(y_lr_t, r):
    """Region-constant replication of an LR tensor to the HR size (torch).

    y_lr_t: (B, 2, Ns/r, Nx/r) -> (B, 2, Ns, Nx); matches C_r exactly.
    """
    return torch.repeat_interleave(torch.repeat_interleave(y_lr_t, r, dim=2), r, dim=3)
