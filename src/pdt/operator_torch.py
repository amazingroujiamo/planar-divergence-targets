"""Differentiable in-plane spectral divergence (plan section 4.1).

Convention (frozen from the archived periodic_fft.py): periodic FFT with
physical spacings, per-axis Nyquist derivative zeroed by INTEGER index
(k[N//2] = 0 for even N), all other wavenumbers retained.  torch.fft uses the
same norm="backward" convention as numpy.fft, so spectral multipliers match
exactly.

The operator acts on velocity normalized by U_g (dimensionless); multiplying
by ell_g = sqrt(Lx*Ls)/(2*pi) makes the output a dimensionless divergence:
    D_nd z' = ell_g * D_physical(z / U_g)
D_physical uses the true in-plane spacings dx = Lx/Nx, ds = Ls/Ns.

Tensor layout: (B, 2, Ns, Nx); x is the LAST axis, s the middle axis.
Both numpy and torch versions implement the identical operator; the numpy
FP64 version is the acceptance reference.
"""
import numpy as np
import torch
import torch.nn as nn


def effective_wavenumbers(Nx, Ns, Lx, Ls, dtype=np.float64):
    """Angular wavenumbers kx (Nx,), ks (Ns,) with integer-index Nyquist zero."""
    dx, ds = Lx / Nx, Ls / Ns
    kx = 2.0 * np.pi * np.fft.fftfreq(Nx, d=dx)
    ks = 2.0 * np.pi * np.fft.fftfreq(Ns, d=ds)
    kx = kx.copy().astype(dtype)
    ks = ks.copy().astype(dtype)
    if Nx % 2 == 0:
        kx[Nx // 2] = 0.0
    if Ns % 2 == 0:
        ks[Ns // 2] = 0.0
    return kx, ks


def divergence_numpy(q, Nx, Ns, Lx, Ls):
    """FP64 reference: q (2, Ns, Nx) -> (Ns, Nx), d = dx q0 + ds q1."""
    kx, ks = effective_wavenumbers(Nx, Ns, Lx, Ls, np.float64)
    U = np.fft.fft2(q[0])
    V = np.fft.fft2(q[1])
    Dhat = 1j * (kx[None, :] * U + ks[:, None] * V)
    return np.real(np.fft.ifft2(Dhat))


def gradient_numpy(q, Nx, Ns, Lx, Ls):
    """All four in-plane gradients for chi^2: returns dict of (Ns, Nx)."""
    kx, ks = effective_wavenumbers(Nx, Ns, Lx, Ls, np.float64)
    out = {}
    for c in range(2):
        F = np.fft.fft2(q[c])
        out["d{}_x".format(c)] = np.real(np.fft.ifft2(1j * kx[None, :] * F))
        out["d{}_s".format(c)] = np.real(np.fft.ifft2(1j * ks[:, None] * F))
    return out


class SpectralDivergence(nn.Module):
    """Differentiable dimensionless in-plane divergence ell_g * D(z/U_g).

    Registers kx, ks (with Nyquist zeroed) and ell_g as buffers; dtype is
    switchable between float32 (training) and float64 (evaluation).  Works on
    real inputs; the FFT path is always the real-axis full complex fft2 so
    torch and numpy agree mode-by-mode (no rfft packing asymmetries).
    """

    def __init__(self, Nx, Ns, Lx, Ls, ell_g, dtype=torch.float32):
        super().__init__()
        kx, ks = effective_wavenumbers(Nx, Ns, Lx, Ls, np.float64)
        self.register_buffer("kx", torch.tensor(kx, dtype=dtype))
        self.register_buffer("ks", torch.tensor(ks, dtype=dtype))
        self.register_buffer("ell_g", torch.tensor(float(ell_g), dtype=dtype))
        self.Nx, self.Ns, self.Lx, self.Ls = Nx, Ns, Lx, Ls

    def set_dtype(self, dtype):
        self.kx = self.kx.to(dtype)
        self.ks = self.ks.to(dtype)
        self.ell_g = self.ell_g.to(dtype)
        return self

    def forward(self, z):
        """z (B, 2, Ns, Nx) normalized velocity -> (B, Ns, Nx) dimensionless div."""
        U = torch.fft.fft2(z[:, 0])
        V = torch.fft.fft2(z[:, 1])
        Dhat = 1j * (self.kx[None, None, :] * U + self.ks[None, :, None] * V)
        d = torch.fft.ifft2(Dhat).real
        return self.ell_g * d
