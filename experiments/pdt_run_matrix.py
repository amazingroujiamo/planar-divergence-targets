#!/usr/bin/env python3
"""Run the frozen formal matrix (plan section 6 / P3) with sharded workers.

  python pdt_run_matrix.py --prebuild            # caches + pair inits first
  python pdt_run_matrix.py --worker --shard i --nshards N [--stage main|continuation]

Workers claim the rows of experiment_matrix.csv assigned to their shard,
skipping rows whose run record already exists.  Every run writes its own
curve/best/last/summary; technical failures keep failure.json and do not
abort the worker.
"""
import argparse
import csv
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.pdt import spec as S                                    # noqa: E402
from src.pdt.train import (train_run, continue_run, get_store,       # noqa: E402
                           load_train_stats, ARTIFACTS)


def read_matrix(project_root):
    path = os.path.join(project_root, ARTIFACTS, "manifests",
                        "experiment_matrix.csv")
    with open(path) as f:
        return list(csv.DictReader(f))


def prebuild(project_root, data_root):
    stats = load_train_stats(project_root)
    import torch
    from src.pdt.model_dsc_ms import DSCMS
    for group in S.FORMAL_GROUPS:
        for factor in S.FACTORS:
            for role in ("train", "val"):
                t0 = time.time()
                s = get_store(project_root, data_root, group, factor, role,
                              stats, device="cuda")
                print("cache {} r{} {}: {} planes ({:.0f}s)".format(
                    group, factor, role, len(s), time.time() - t0), flush=True)
            cond = "{}_r{}".format(group, factor)
            for seed in S.FORMAL_SEEDS:
                p = os.path.join(project_root, ARTIFACTS, "runs", "formal",
                                 cond, "pair_seeds", "s{}".format(seed), "init.pt")
                os.makedirs(os.path.dirname(p), exist_ok=True)
                if not os.path.exists(p):
                    torch.manual_seed(seed)
                    torch.save(DSCMS().state_dict(), p)
    print("PREBUILD DONE")


def run_dir_abs(project_root, rel):
    return os.path.join(project_root, ARTIFACTS, rel)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=os.getcwd())
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--prebuild", action="store_true")
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--stage", default="main",
                    choices=["main", "continuation"])
    args = ap.parse_args()
    if args.prebuild:
        prebuild(args.project_root, args.data_root)
        return
    if not args.worker:
        print("use --prebuild or --worker")
        return
    rows = [r for r in read_matrix(args.project_root) if r["stage"] == args.stage]
    mine = [r for i, r in enumerate(rows) if i % args.nshards == args.shard]
    print("shard {}/{} stage {}: {} rows".format(
        args.shard, args.nshards, args.stage, len(mine)), flush=True)
    for r in mine:
        rd = run_dir_abs(args.project_root, r["run_dir"])
        if os.path.exists(os.path.join(rd, "summary.json")) or \
                os.path.exists(os.path.join(rd, "failure.json")):
            print("skip existing", r["run_dir"], flush=True)
            continue
        group = r["group"]
        factor = int(r["factor"])
        seed = int(r["seed"])
        target = r["target"]
        lam = float(r["lam"])
        print("RUN", r["run_dir"], flush=True)
        try:
            if args.stage == "main":
                s = train_run(args.project_root, args.data_root, group,
                              factor, seed, target, lam, max_epochs=300,
                              device="cuda")
            else:
                s = continue_run(args.project_root, args.data_root, group,
                                 factor, seed, target, lam, to_epoch=600,
                                 lr=1e-4, device="cuda")
            print("done {} epochs={} best={:.6g} reason={}".format(
                r["run_dir"], s.get("epochs_run"), s.get("best_val_E_v",
                                                         s.get("final_val_E_v")),
                s["stop_reason"]), flush=True)
        except Exception as e:  # noqa: BLE001
            print("TASK ERROR", r["run_dir"], repr(e), flush=True)


if __name__ == "__main__":
    main()
