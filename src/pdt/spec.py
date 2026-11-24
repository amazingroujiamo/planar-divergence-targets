"""Planar divergence targets v1: dataset specs, splits and length scales.

Implements EXPERIMENT_PLAN_20261013.md sections 2-4.  All JHTDB indices are
1-based; array indices are 0-based.  Tensor layout everywhere is
(C=2, Ns, Nx): axis 1 is the in-plane 's' axis (iso: y; channel: z), axis 2
is x.  In-plane divergence:
    iso_xy      (u, v)  ->  dx u + dy v   = -dz w
    channel_*   (u, w)  ->  dx u + dz w   = -dy v

Old stored frames use the legacy sampling index m with time_index = 1 + step*m
(iso step 78, channel step 62).  The NEW prospective test frames (plan 2.4)
use time_index = offset + step*m with m = 48..63 (iso offset 40, channel
offset 32); these are native stored frames never downloaded before and are
read by explicit time index, never by the old audit-m slab rule.
"""
import math

ARTIFACTS = "artifacts/planar_divergence_targets_v1"

TWO_PI = 2.0 * math.pi

DATASETS = {
    "isotropic1024coarse": {
        "normal_axis": "z",
        "grid_xyz": (1024, 1024, 1024),
        "Nx": 1024, "Ns": 1024,          # axis1 = s = y, axis2 = x
        "Lx": TWO_PI, "Ls": TWO_PI,
        "components": ("u", "v"),        # reconstruction components (idx 0, 1)
        "comp_idx": (0, 1),
        "time_step": 78, "native_dt": 0.002,
        "normal_spacing": TWO_PI / 1024.0,
    },
    "channel": {
        "normal_axis": "y",
        "grid_xyz": (2048, 512, 1536),
        "Nx": 2048, "Ns": 1536,          # axis1 = s = z, axis2 = x
        "Lx": 8.0 * math.pi, "Ls": 3.0 * math.pi,
        "components": ("u", "w"),        # idx 0, 2
        "comp_idx": (0, 2),
        "time_step": 62, "native_dt": 0.0065,
        "y75_native": -0.8991894845941382,
        "y256_native": -0.003079524345704338,
        "y30_native": -0.9846274440829048,
    },
}

# formal groups (plan 2.1); channel_y030 deliberately excluded from the matrix
GROUPS = {
    "iso_xy": {
        "dataset": "isotropic1024coarse",
        "centers_1based": [129, 385, 641, 897],
        "center_coords": [math.pi / 4, 3 * math.pi / 4, 5 * math.pi / 4, 7 * math.pi / 4],
    },
    "channel_y075": {
        "dataset": "channel",
        "centers_1based": [75],
        "center_coords": [DATASETS["channel"]["y75_native"]],
    },
    "channel_y256": {
        "dataset": "channel",
        "centers_1based": [256],
        "center_coords": [DATASETS["channel"]["y256_native"]],
    },
}

FORMAL_GROUPS = ["iso_xy", "channel_y075", "channel_y256"]
FACTORS = [4, 8]
FORMAL_SEEDS = [11, 22, 33]
PILOT_SEED = 900

# Time splits on the legacy sampling index m (plan 2.3; identical ranges to
# the archived g=6 (iso) / g=2 (channel) layout whose gap evidence is reused).
SPLITS = {
    "isotropic1024coarse": {
        "train": list(range(0, 28)),
        "gap_train_val": list(range(28, 34)),
        "val": list(range(34, 42)),
        "gap_val_oldtest": list(range(42, 48)),
        "old_test": list(range(48, 64)),
    },
    "channel": {
        "train": list(range(0, 36)),
        "gap_train_val": list(range(36, 38)),
        "val": list(range(38, 46)),
        "gap_val_oldtest": list(range(46, 48)),
        "old_test": list(range(48, 64)),
    },
}

# Prospective test cohort (plan 2.4): native stored time indices that were
# never part of the legacy 1+step*m sampling.
PROSPECTIVE = {
    "isotropic1024coarse": {
        "m_range": list(range(48, 64)),
        "time_index": {m: 40 + 78 * m for m in range(48, 64)},
        "centers_1based": [129, 385, 641, 897],
    },
    "channel": {
        "m_range": list(range(48, 64)),
        "time_index": {m: 32 + 62 * m for m in range(48, 64)},
        "centers_1based": [75, 256],
    },
}


def legacy_time_index(dataset, m):
    return 1 + DATASETS[dataset]["time_step"] * m


def physical_time(dataset, time_index_1based):
    return (time_index_1based - 1) * DATASETS[dataset]["native_dt"]


def length_scale(group):
    """ell_g = sqrt(Lx*Ls)/(2*pi); iso 1.0, channel sqrt(6)."""
    spec = DATASETS[GROUPS[group]["dataset"]]
    return math.sqrt(spec["Lx"] * spec["Ls"]) / TWO_PI


def group_grid(group):
    spec = DATASETS[GROUPS[group]["dataset"]]
    return {"Nx": spec["Nx"], "Ns": spec["Ns"],
            "Lx": spec["Lx"], "Ls": spec["Ls"],
            "dx": spec["Lx"] / spec["Nx"], "ds": spec["Ls"] / spec["Ns"],
            "ell_g": length_scale(group)}


def group_samples(group, role):
    """(m, center) pairs of one split role for one formal group."""
    ds = GROUPS[group]["dataset"]
    ms = SPLITS[ds][role]
    return [(m, c) for m in ms for c in GROUPS[group]["centers_1based"]]


def plane_relpath(dataset, time_index_1based, center_1based, prospective=False):
    ax = DATASETS[dataset]["normal_axis"]
    fname = "t{:06d}_{}{:04d}.h5".format(time_index_1based, ax, center_1based)
    sub = "raw_prospective" if prospective else "raw"
    return "{}/{}/{}".format(sub, dataset, fname)
