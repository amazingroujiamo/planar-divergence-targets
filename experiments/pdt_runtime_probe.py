#!/usr/bin/env python3
"""Runtime / memory probe (plan section 11.2): warm up 5 steps then time at
least 20 full training steps and one full validation pass, per data type and
per loss class, on the pilot conditions (iso_xy r=4, channel_y075 r=4).
Writes reports/runtime_pilot.json.  No checkpoints are kept.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.pdt import spec as S                                    # noqa: E402
from src.pdt.data import PlaneStore                              # noqa: E402
from src.pdt.losses import loss_terms                            # noqa: E402
from src.pdt.model_dsc_ms import DSCMS, replicate_lr_to_hr       # noqa: E402
from src.pdt.operator_torch import SpectralDivergence            # noqa: E402
from src.pdt.train import _validate, load_train_stats, ARTIFACTS  # noqa: E402


def probe(project_root, data_root, group, factor, target, lam, device="cuda",
          warmup=5, timed=20):
    stats = load_train_stats(project_root)
    g = S.group_grid(group)
    U_g = stats["groups"][group]["U_g"]
    ell_g = stats["groups"][group]["ell_g"]
    ds = S.GROUPS[group]["dataset"]
    # subsample the train planes for the probe: first `warmup+timed` samples
    samples = S.group_samples(group, "train")[: warmup + timed]
    store = PlaneStore(data_root, group, factor, samples, U_g, ell_g,
                       device_for_prefetch=device)
    val_samples = S.group_samples(group, "val")
    val_store = PlaneStore(data_root, group, factor, val_samples, U_g, ell_g,
                           device_for_prefetch=device)
    div = SpectralDivergence(g["Nx"], g["Ns"], g["Lx"], g["Ls"], ell_g,
                             dtype=torch.float32).to(device)
    model = DSCMS().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    if device.startswith("cuda"):
        torch.backends.cudnn.benchmark = True
        torch.cuda.reset_peak_memory_stats()
    step_times = []
    t_all0 = time.time()
    n = warmup + timed
    for k in range(n):
        it = store.items[k % len(store.items)]
        x = torch.from_numpy(np.ascontiguousarray(it["y32"]))[None].to(device)
        x = replicate_lr_to_hr(x, factor)
        q = torch.from_numpy(np.ascontiguousarray(it["q32"]))[None].to(device)
        dref = (torch.from_numpy(np.ascontiguousarray(it["dref32"]))[None].to(device)
                if target == "T" else None)
        t0 = time.time()
        opt.zero_grad(set_to_none=True)
        out = model(x)
        total, L_v, div_raw, _, _ = loss_terms(out, q, dref, div, lam, target)
        total.backward()
        opt.step()
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        dt = time.time() - t0
        if k >= warmup:
            step_times.append(dt)
    t_val0 = time.time()
    val, diag = _validate(model, val_store, div, g, device, ds, U_g)
    t_val = time.time() - t_val0
    peak = (float(torch.cuda.max_memory_allocated()) / 2**30
            if device.startswith("cuda") else 0.0)
    peak_reserved = (float(torch.cuda.max_memory_reserved()) / 2**30
                     if device.startswith("cuda") else 0.0)
    return {
        "group": group, "factor": factor, "target": target, "lam": lam,
        "n_timed_steps": len(step_times),
        "step_seconds_mean": float(np.mean(step_times)),
        "step_seconds_std": float(np.std(step_times)),
        "step_seconds_max": float(np.max(step_times)),
        "val_planes": len(val_store),
        "val_seconds": t_val, "val_E_v_probe": val,
        "peak_mem_alloc_gib": peak, "peak_mem_reserved_gib": peak_reserved,
        "n_train_planes_full": len(S.group_samples(group, "train")),
        "projected_epoch_seconds": float(
            np.mean(step_times) * len(S.group_samples(group, "train")) + t_val),
        "device": torch.cuda.get_device_name(0) if device.startswith("cuda") else "cpu",
        "torch": torch.__version__,
        "probe_total_seconds": time.time() - t_all0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=os.getcwd())
    ap.add_argument("--data-root", required=True)
    args = ap.parse_args()
    out = {"study": "planar_divergence_targets_v1",
           "checked_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "probes": []}
    for group, factor in (("iso_xy", 4), ("channel_y075", 4)):
        out["probes"].append(probe(args.project_root, args.data_root, group,
                                   factor, "A", 0.0))
        out["probes"].append(probe(args.project_root, args.data_root, group,
                                   factor, "Z", 1e-3))
        out["probes"].append(probe(args.project_root, args.data_root, group,
                                   factor, "T", 1e-3))
    path = os.path.join(args.project_root, ARTIFACTS, "reports",
                        "runtime_pilot.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    for p in out["probes"]:
        print("{} r{} {} lam={:g}: {:.3f}s/step, val {:.1f}s, peak {:.2f} GiB, "
              "epoch~{:.1f}s".format(p["group"], p["factor"], p["target"], p["lam"],
                                     p["step_seconds_mean"], p["val_seconds"],
                                     p["peak_mem_alloc_gib"],
                                     p["projected_epoch_seconds"]))


if __name__ == "__main__":
    main()
