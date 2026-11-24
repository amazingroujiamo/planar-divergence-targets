#!/usr/bin/env python3
"""Freeze the formal experiment matrix (plan sections 6, 11.3 step P2).

Reads the pilot-selected three weights from
manifests/pilot_weight_selection.json and writes manifests/experiment_matrix.csv
covering the 126 main runs plus the 18 planned continuation stages, with the
code snapshot hash, lr schedule, seeds and run-directory conventions.  The
file is written ONCE; rerunning refuses to overwrite.
"""
import argparse
import csv
import hashlib
import io
import json
import os
import sys
import tarfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.pdt import spec as S                                    # noqa: E402
from src.pdt.train import ARTIFACTS, LR_SCHEDULE                 # noqa: E402


def code_snapshot_hash(project_root):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for sub in ("src/pdt", "experiments"):
            base = os.path.join(project_root, sub)
            if not os.path.isdir(base):
                continue
            for fn in sorted(os.listdir(base)):
                if fn.endswith(".py"):
                    p = os.path.join(base, fn)
                    tar.add(p, arcname="{}/{}".format(sub, fn))
    return hashlib.sha256(buf.getvalue()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=os.getcwd())
    args = ap.parse_args()
    man = os.path.join(args.project_root, ARTIFACTS, "manifests")
    sel_path = os.path.join(man, "pilot_weight_selection.json")
    with open(sel_path) as f:
        sel = json.load(f)
    lams = [float(x) for x in sel["chosen"]]
    assert len(lams) == 3 and lams[0] < lams[1] < lams[2], lams
    out_path = os.path.join(man, "experiment_matrix.csv")
    if os.path.exists(out_path):
        raise SystemExit("refusing to overwrite existing " + out_path)

    code_sha = code_snapshot_hash(args.project_root)
    rows = []
    configs = [("A", 0.0)] + [("Z", l) for l in lams] + [("T", l) for l in lams]
    for group in S.FORMAL_GROUPS:
        for factor in S.FACTORS:
            cond = "{}_r{}".format(group, factor)
            for seed in S.FORMAL_SEEDS:
                for target, lam in configs:
                    w = "shared" if target == "A" else "lam_" + format(lam, ".0e")
                    rd = os.path.join("runs", "formal", cond, "s{}".format(seed),
                                      target, w)
                    rows.append({
                        "condition": cond, "group": group, "factor": factor,
                        "seed": seed, "target": target, "lam": format(lam, ".17g"),
                        "stage": "main", "epochs": 300,
                        "lr_schedule": "|".join("{:.0e}".format(l) for _, _, l in LR_SCHEDULE),
                        "run_dir": rd, "status": "pending",
                    })
    for group in S.FORMAL_GROUPS:            # continuation: seed 11, lam_high
        for factor in S.FACTORS:
            cond = "{}_r{}".format(group, factor)
            for target in ("A", "Z", "T"):
                lam = 0.0 if target == "A" else lams[2]
                w = "shared" if target == "A" else "lam_" + format(lam, ".0e")
                rd = os.path.join("runs", "formal", cond, "s11", target, w,
                                  "continuation")
                rows.append({
                    "condition": cond, "group": group, "factor": factor,
                    "seed": 11, "target": target,
                    "lam": format(lam, ".17g"),
                    "stage": "continuation", "epochs": "301-600",
                    "lr_schedule": "1e-4 fixed",
                    "run_dir": rd, "status": "pending",
                })
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    meta = {
        "study": "planar_divergence_targets_v1",
        "frozen_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "lambda_low": lams[0], "lambda_mid": lams[1], "lambda_high": lams[2],
        "selection_source": "pilot_weight_selection.json",
        "code_snapshot_sha256": code_sha,
        "formal_seeds": S.FORMAL_SEEDS,
        "main_runs": sum(1 for r in rows if r["stage"] == "main"),
        "continuation_stages": sum(1 for r in rows if r["stage"] == "continuation"),
        "formal_epochs": 300,
        "batch": "one full native plane",
        "optimizer": "Adam betas(0.9,0.999) eps1e-8 wd0",
        "precision": "fp32 training + fp32 FFT; fp64 evaluation",
    }
    with open(os.path.join(man, "matrix_freeze.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print("frozen:", json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
