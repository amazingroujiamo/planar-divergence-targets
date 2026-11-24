#!/usr/bin/env python3
"""P0 data preparation: source inventory, split manifest, train_stats (U_g,
ell_g) and train-set divergence diagnostics (plan sections 2-3, 7.3).

Outputs under artifacts/planar_divergence_targets_v1/manifests/:
  source_inventory.csv   one row per native source file actually used
                         (train+val roles here; prospective rows appended by
                         the acquisition step), with sha256
  split_manifest.csv     one row per (dataset, m, center) with its role
  train_stats.json       U_g and ell_g per formal group + provenance
  data_diagnostics.json  per-group reference divergence amplitude A_d* and
                         gradient share chi^2 over train and val planes
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.pdt import spec as S                                    # noqa: E402
from src.pdt.data import (read_center_plane, sha256_file,         # noqa: E402
                          block_average)
from src.pdt.metrics import divergence_metrics, chi_squared       # noqa: E402

ART = "artifacts/planar_divergence_targets_v1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=os.getcwd())
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--legacy-stats", default=None,
                    help="archived dns_v1 train_stats.json for cross-check")
    args = ap.parse_args()
    t0 = time.time()
    man = os.path.join(args.project_root, ART, "manifests")
    os.makedirs(man, exist_ok=True)

    # ---- split manifest (legacy m-based roles) ----
    rows = []
    for group in S.FORMAL_GROUPS:
        gd = S.GROUPS[group]
        ds = gd["dataset"]
        roles = S.SPLITS[ds]
        for role, ms in roles.items():
            for m in ms:
                for c in gd["centers_1based"]:
                    ti = S.legacy_time_index(ds, m)
                    rows.append({
                        "group": group, "dataset": ds, "m": m,
                        "time_index_1based": ti,
                        "time_physical": "{:.6g}".format(S.physical_time(ds, ti)),
                        "center_1based": c,
                        "role": role,
                        "relpath": S.plane_relpath(ds, ti, c),
                    })
    with open(os.path.join(man, "split_manifest.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("split_manifest.csv:", len(rows), "rows")

    # ---- source inventory: files actually used by train/val ----
    used = {}
    for r in rows:
        if r["role"] in ("train", "val"):
            used.setdefault(r["relpath"], {"roles": set(), "dataset": r["dataset"],
                                           "groups": set()})
            used[r["relpath"]]["roles"].add(r["role"])
            used[r["relpath"]]["groups"].add(r["group"])
    inv = []
    for rel in sorted(used):
        path = os.path.join(args.data_root, rel)
        size = os.path.getsize(path)
        h = sha256_file(path)
        inv.append({"relpath": rel, "dataset": used[rel]["dataset"],
                    "sha256": h, "bytes": size,
                    "roles": "+".join(sorted(used[rel]["roles"])),
                    "groups": "+".join(sorted(used[rel]["groups"]))})
        print("hashed", rel, h[:12])
    with open(os.path.join(man, "source_inventory.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(inv[0].keys()))
        w.writeheader()
        w.writerows(inv)
    print("source_inventory.csv:", len(inv), "files")

    # ---- train stats: U_g and ell_g per group ----
    groups_out = {}
    for group in S.FORMAL_GROUPS:
        gd = S.GROUPS[group]
        ds = gd["dataset"]
        g = S.group_grid(group)
        vals = []
        diags = {"train": [], "val": []}
        for role in ("train", "val"):
            for (m, c) in S.group_samples(group, role):
                ti = S.legacy_time_index(ds, m)
                q64, info = read_center_plane(args.data_root, ds, ti, c)
                mu = q64.mean(axis=(1, 2))
                qf = q64 - mu[:, None, None]
                if role == "train":
                    vals.append(float(np.sum(qf * qf)) / (2.0 * q64.shape[1] * q64.shape[2]))
                U_probe = 1.0  # chi2 is scale-free; A_d* uses the final U_g
                dm = divergence_metrics(q64, q64, g, 1.0, g["ell_g"])
                chi2, _, _ = chi_squared(q64, g)
                diags[role].append({
                    "m": m, "center": c,
                    "A_d_star_unitU": dm["A_d_star"], "chi2": chi2,
                })
        U2 = float(np.mean(vals))
        U = float(np.sqrt(U2))
        # recompute A_d* in final normalization for reporting
        A_train = []
        for (m, c) in S.group_samples(group, "train"):
            ti = S.legacy_time_index(ds, m)
            q64, _ = read_center_plane(args.data_root, ds, ti, c)
            dm = divergence_metrics(q64, q64, g, U, g["ell_g"])
            A_train.append(dm["A_d_star"])
        groups_out[group] = {
            "dataset": ds,
            "centers_1based": gd["centers_1based"],
            "n_train_planes": len(vals),
            "U_g": U, "U_g_squared": U2,
            "U_g_definition": ("sqrt(mean over train planes of "
                               "mean_{c,i,j}(q* - mu_LR)^2)"),
            "ell_g": g["ell_g"],
            "ell_g_definition": "sqrt(Lx*Ls)/(2*pi)",
            "grid": {k: g[k] for k in ("Nx", "Ns", "Lx", "Ls", "dx", "ds")},
            "train_A_d_star_mean": float(np.mean(A_train)),
            "train_chi2_mean": float(np.mean([d["chi2"] for d in diags["train"]])),
            "val_A_d_star_mean_unitU": float(np.mean(
                [d["A_d_star_unitU"] for d in diags["val"]])),
        }
        print(group, "U_g={:.6g} ell={:.6g} A_d*={:.6g} chi2={:.6g}".format(
            U, g["ell_g"], groups_out[group]["train_A_d_star_mean"],
            groups_out[group]["train_chi2_mean"]))
    stats = {
        "study": "planar_divergence_targets_v1",
        "computed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "data_root_recorded_at_runtime": True,
        "source_inventory": "source_inventory.csv",
        "split_manifest": "split_manifest.csv",
        "groups": groups_out,
    }
    if args.legacy_stats and os.path.exists(args.legacy_stats):
        with open(args.legacy_stats) as f:
            legacy = json.load(f)["groups"]
        x = {}
        for gname, rec in groups_out.items():
            if gname in legacy:
                old = float(legacy[gname]["s_g"])
                x[gname] = {"legacy_s_g": old, "U_g": rec["U_g"],
                            "rel_diff": abs(old - rec["U_g"]) / old}
        stats["legacy_crosscheck"] = x
        print("legacy crosscheck:", json.dumps(x))
    with open(os.path.join(man, "train_stats.json"), "w") as f:
        json.dump(stats, f, indent=1)
    with open(os.path.join(man, "data_diagnostics.json"), "w") as f:
        json.dump({g: {"train": groups_out[g]["train_A_d_star_mean"],
                       "chi2": groups_out[g]["train_chi2_mean"]}
                   for g in groups_out}, f, indent=1)
    print("elapsed {:.1f}s".format(time.time() - t0))


if __name__ == "__main__":
    main()
