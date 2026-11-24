"""Native plane reading, observation and normalized plane stores.

read_center_plane resolves the CENTER layer from the ACTUAL file layout
(shape[normal]==1 -> single layer; ==9 -> slab, center at index 4), never
from the legacy audit-m rule (plan 2.4: new single-layer prospective files
must not be misread as slabs).  The native center coordinate is verified
against the dataset spec when the file exposes it.

PlaneStore preloads for one (group, factor, role):
    q64      (2, Ns, Nx) float64 original HR velocity (never projected)
    y64      (2, Ns/r, Nx/r) float64 block-average LR observation
    mu64     (2,) per-component LR plane mean == exact HR mean
    q32/y32  normalized by the frozen train-set scale U_g (float32)
    dref32   ell_g*D(q32) reference dimensionless divergence map, computed
             once with the SAME torch operator under no_grad (plan 4.2);
             cache identity (file hashes, operator id, U_g, ell_g, dtype)
             is stored alongside for the acceptance check
    denom64  sum_c (q* - mu)^2 in physical units (E_v denominator)
"""
import hashlib
import os

import h5py
import numpy as np
import torch

from . import spec as S
from .operator_torch import SpectralDivergence, divergence_numpy

OPERATOR_ID = "pdt.spectral_divergence.v1"


def block_average(q, f):
    """Area average over f x f blocks -> (2, Ns//f, Nx//f) float64."""
    C, Ns, Nx = q.shape
    assert Ns % f == 0 and Nx % f == 0
    Ns2, Nx2 = Ns // f, Nx // f
    out = np.empty((C, Ns2, Nx2), dtype=np.float64)
    for c in range(C):
        blk = q[c].reshape(Ns2, f, Nx2, f)
        out[c] = blk.mean(axis=(1, 3))
    return out


def sha256_file(path, chunk=8 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def read_center_plane(data_root, dataset, time_index_1based, center_1based,
                      prospective=False):
    """Return (q64, info). q64 is (2, Ns, Nx) float64 reconstruction comps.

    The center layer is chosen from the actual normal-axis extent stored in
    the file: 1 layer -> index 0; 9 layers -> index 4 (slab center); anything
    else is a structural error.  The native center coordinate is checked
    against the dataset spec when present in the file.
    """
    sp = S.DATASETS[dataset]
    rel = S.plane_relpath(dataset, time_index_1based, center_1based, prospective)
    path = os.path.join(data_root, rel)
    with h5py.File(path, "r") as f:
        vel = f["velocity"]
        ax = sp["normal_axis"]
        if ax == "z":
            nz, ny, nx, nc = vel.shape
        else:
            nz, ny, nx, nc = vel.shape
        if ax == "z":
            n_normal = nz
        else:
            n_normal = ny
        if n_normal == 1:
            ci = 0
        elif n_normal == 9:
            ci = 4
        else:
            raise ValueError("unexpected normal extent {} in {}".format(n_normal, path))
        if ax == "z":
            arr = vel[ci, :, :, list(sp["comp_idx"])]        # (Ns, Nx, 2) = (y, x, c)
            coords = np.asarray(f["z"][:], dtype=np.float64) if "z" in f else None
        else:
            arr = vel[:, ci, :, list(sp["comp_idx"])]        # (Ns, Nx, 2) = (z, x, c)
            coords = np.asarray(f["y"][:], dtype=np.float64) if "y" in f else None
        q = np.ascontiguousarray(arr.transpose(2, 0, 1)).astype(np.float64)
    if q.shape != (2, sp["Ns"], sp["Nx"]):
        raise ValueError("bad plane shape {} for {} {}".format(q.shape, dataset, rel))
    if not np.isfinite(q).all():
        raise ValueError("non-finite values in {}".format(rel))
    if coords is not None and len(coords) == n_normal and n_normal > 1:
        # verify the slab center sits at the requested native coordinate
        expected = None
        for g, gd in S.GROUPS.items():
            if gd["dataset"] == dataset and center_1based in gd["centers_1based"]:
                expected = gd["center_coords"][gd["centers_1based"].index(center_1based)]
        if expected is not None and abs(coords[ci] - expected) > 1e-9:
            raise ValueError("center coord {} != expected {} in {}".format(
                coords[ci], expected, rel))
    info = {
        "dataset": dataset,
        "time_index_1based": int(time_index_1based),
        "time_physical": S.physical_time(dataset, time_index_1based),
        "center_1based": int(center_1based),
        "n_normal_layers": int(n_normal),
        "prospective": bool(prospective),
        "source_file": rel,
        "shape_2csx": list(q.shape),
    }
    return q, info


def load_channel_y_table(data_root):
    path = os.path.join(data_root, "coordinates", "channel_y_native.csv")
    raw = np.loadtxt(path, delimiter=",", comments="#", skiprows=1)
    y = raw[:, 1].astype(np.float64)
    if raw.shape != (512, 2) or not np.all(np.diff(y) > 0):
        raise ValueError("bad channel y table")
    return y


class PlaneStore:
    """Preloaded normalized planes of one (group, factor, role)."""

    def __init__(self, data_root, group, factor, samples, U_g, ell_g,
                 grid=None, device_for_prefetch="cpu", with_dref=True):
        gd = S.GROUPS[group]
        dataset = gd["dataset"]
        sp = S.DATASETS[dataset]
        g = grid or S.group_grid(group)
        self.group, self.factor, self.U_g, self.ell_g = group, factor, U_g, ell_g
        self.grid = g
        self.items = []
        div = SpectralDivergence(g["Nx"], g["Ns"], g["Lx"], g["Ls"], ell_g,
                                 dtype=torch.float32).to(device_for_prefetch)
        hashes = []
        for (m, c) in samples:
            ti = S.legacy_time_index(dataset, m)
            q64, info = read_center_plane(data_root, dataset, ti, c)
            y64 = block_average(q64, factor)
            mu64 = y64.mean(axis=(1, 2))
            q32 = ((q64 - mu64[:, None, None]) / U_g).astype(np.float32)
            y32 = ((y64 - mu64[:, None, None]) / U_g).astype(np.float32)
            denom = float(np.sum((q64 - mu64[:, None, None]) ** 2))
            dref32 = None
            if with_dref:
                with torch.no_grad():
                    z = torch.from_numpy(q32)[None].to(device_for_prefetch)
                    dref32 = div(z)[0].cpu().numpy().astype(np.float32)
            # label divergence energy in normalized units (for checks)
            dref_energy = float(np.mean(dref32.astype(np.float64) ** 2)) if dref32 is not None else None
            self.items.append({
                "m": m, "center": c, "info": info,
                "q64": q64, "mu64": mu64, "denom64": denom,
                "q32": q32, "y32": y32, "dref32": dref32,
                "dref_energy64": dref_energy,
            })
        self.cache_identity = {
            "operator_id": OPERATOR_ID,
            "U_g": float(U_g), "ell_g": float(ell_g),
            "dtype": "float32", "n_planes": len(self.items),
            "source_files": [it["info"]["source_file"] for it in self.items],
        }

    def __len__(self):
        return len(self.items)


def compute_U_g(data_root, group):
    """U_g^2 = mean over TRAIN planes of mean_{c,i,j} (q* - mu_LR)^2 (plan 3).

    mu_LR is the per-component block-average plane mean with the exact-
    arithmetic convention r_probe=8 (factor-independent); computed here in
    float64 from the original HR planes.
    """
    gd = S.GROUPS[group]
    dataset = gd["dataset"]
    vals = []
    for (m, c) in S.group_samples(group, "train"):
        ti = S.legacy_time_index(dataset, m)
        q64, _ = read_center_plane(data_root, dataset, ti, c)
        mu = q64.mean(axis=(1, 2))       # exact == block-average mean
        qf = q64 - mu[:, None, None]
        vals.append(float(np.sum(qf * qf)) / (2.0 * q64.shape[1] * q64.shape[2]))
    return float(np.sqrt(np.mean(vals))), len(vals)
