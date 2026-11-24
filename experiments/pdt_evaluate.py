#!/usr/bin/env python3
"""Prospective test evaluation (plan sections 6.3, 7; P4).

Two comparison families, both evaluated on the SAME prospective native test
frames (never used for any selection):
  1. same-weight comparison: every formal model (7 configs x 3 seeds);
  2. application comparison: per condition and target, the weight chosen on
     the three-seed mean VALIDATION E_v among {0, lam_low, lam_mid, lam_high}
     (ties or <1% relative difference -> smaller weight; zero allowed).

Outputs:
  metrics/per_field.csv   one row per (model, test field), all plan-7 metrics
  manifests/formal_weight_selection.json
Test truth enters only here; predictions are restored to physical units and
compared against the ORIGINAL, unprojected HR velocity in float64.
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.pdt import spec as S                                    # noqa: E402
from src.pdt.data import read_center_plane, block_average, PlaneStore  # noqa: E402
from src.pdt.metrics import plane_metrics                        # noqa: E402
from src.pdt.model_dsc_ms import DSCMS, replicate_lr_to_hr       # noqa: E402
from src.pdt.train import ARTIFACTS, load_train_stats           # noqa: E402


def matrix_rows(project_root):
    with open(os.path.join(project_root, ARTIFACTS, "manifests",
                           "experiment_matrix.csv")) as f:
        return [r for r in csv.DictReader(f) if r["stage"] == "main"]


def validation_selection(project_root, rows):
    """Plan 6.3 rule 2.  Returns {(cond, target): (lam, reason)}.  For target
    'A' the only candidate is the shared baseline (lam 0)."""
    by_key = {}
    for r in rows:
        by_key.setdefault((r["condition"], r["target"], float(r["lam"])), []).append(r)
    sel = {}
    conds = sorted({r["condition"] for r in rows})
    for cond in conds:
        for target in ("A", "Z", "T"):
            cands = []
            base = by_key.get((cond, "A", 0.0))
            base_val = (np.mean([float(
                json.load(open(os.path.join(
                    project_root, ARTIFACTS, r["run_dir"], "summary.json"))
                )["best_val_E_v"]) for r in base])
                if base else None)
            if target == "A":
                sel[(cond, target)] = (0.0, "baseline only", base_val)
                continue
            lams = sorted({l for (c, t, l) in by_key if c == cond and t == target})
            best = None
            for lam in [0.0] + lams:
                if lam == 0.0:
                    v = base_val
                else:
                    rr = by_key[(cond, target, lam)]
                    if len(rr) != 3:
                        continue
                    v = float(np.mean([
                        json.load(open(os.path.join(
                            project_root, ARTIFACTS, r["run_dir"], "summary.json"))
                        )["best_val_E_v"] for r in rr]))
                if v is None:
                    continue
                # within 1% of the running best -> keep the SMALLER weight
                if best is None or v < best[1] * 0.99:
                    best = (lam, v)
                elif v < best[1] and lam < best[0]:
                    best = (lam, v)
            sel[(cond, target)] = (best[0], "min 3-seed mean val E_v; <1% ties to smaller",
                                   best[1])
    return sel


def load_model(run_dir_rel, project_root, device):
    p = os.path.join(project_root, ARTIFACTS, run_dir_rel, "best.pt")
    ck = torch.load(p, map_location="cpu", weights_only=False)
    model = DSCMS().to(device)
    model.load_state_dict(ck["model_state"])
    model.eval()
    return model, ck


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=os.getcwd())
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cohort", default="prospective",
                    choices=["prospective", "old_test"])
    args = ap.parse_args()
    t0 = time.time()
    stats = load_train_stats(args.project_root)
    rows = matrix_rows(args.project_root)

    sel_path = os.path.join(args.project_root, ARTIFACTS, "manifests",
                            "formal_weight_selection.json")
    if not os.path.exists(sel_path):
        sel = validation_selection(args.project_root, rows)
        with open(sel_path, "w") as f:
            json.dump({ "{}|{}".format(k[0], k[1]): {
                "lam": v[0], "rule": v[1], "mean_val_E_v": v[2]}
                for k, v in sel.items()}, f, indent=1)
    with open(sel_path) as f:
        sel_raw = json.load(f)

    out_dir = os.path.join(args.project_root, ARTIFACTS, "metrics")
    os.makedirs(out_dir, exist_ok=True)
    fieldnames = ["condition", "group", "factor", "seed", "target", "lam",
                  "dataset", "cohort", "m", "time_index_1based", "center_1based",
                  "E_v", "sqrt_E_v", "E_v_c0", "E_v_c1",
                  "E_d", "A_d", "A_d_star", "gain_a", "residual_energy",
                  "ref_energy", "identity_check", "chi2", "obs_consistency",
                  "e_v_den"]
    out_path = os.path.join(out_dir, "per_field_{}.csv".format(args.cohort))
    done = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            for r in csv.DictReader(f):
                done.add((r["condition"], r["seed"], r["target"], r["lam"],
                          r["m"], r["center_1based"]))
        fout = open(out_path, "a", newline="")
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
    else:
        fout = open(out_path, "w", newline="")
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()

    n_eval = 0
    try:
        for r in rows:
            group, factor = r["group"], int(r["factor"])
            dataset = S.GROUPS[group]["dataset"]
            g = S.group_grid(group)
            U_g = stats["groups"][group]["U_g"]
            ell_g = stats["groups"][group]["ell_g"]
            rd = r["run_dir"]
            summ_path = os.path.join(args.project_root, ARTIFACTS, rd, "summary.json")
            if not os.path.exists(summ_path):
                print("MISSING summary, skip", rd)
                continue
            with open(summ_path) as f:
                summ = json.load(f)
            if summ.get("stop_reason") == "technical_fail" and summ.get("best_epoch") is None:
                print("FAILED run without best checkpoint, skip", rd)
                continue
            # test fields for this condition
            if args.cohort == "prospective":
                pc = S.PROSPECTIVE[dataset]
                fields = [(m, c, pc["time_index"][m], True)
                          for m in pc["m_range"]
                          for c in S.GROUPS[group]["centers_1based"]]
            else:
                fields = [(m, c, S.legacy_time_index(dataset, m), False)
                          for m in S.SPLITS[dataset]["old_test"]
                          for c in S.GROUPS[group]["centers_1based"]]
            todo = [(m, c, ti, pro) for (m, c, ti, pro) in fields
                    if (r["condition"], r["seed"], r["target"], r["lam"],
                        str(m), str(c)) not in done]
            if not todo:
                continue
            model, ck = load_model(rd, args.project_root, args.device)
            for (m, c, ti, pro) in todo:
                q64, info = read_center_plane(args.data_root, dataset, ti, c,
                                              prospective=pro)
                y64 = block_average(q64, factor)
                mu64 = y64.mean(axis=(1, 2))
                y32 = ((y64 - mu64[:, None, None]) / U_g).astype(np.float32)
                x = torch.from_numpy(np.ascontiguousarray(y32))[None].to(args.device)
                x = replicate_lr_to_hr(x, factor)
                with torch.no_grad():
                    z32 = model(x)[0].float().cpu().numpy()
                z64 = U_g * z32.astype(np.float64) + mu64[:, None, None]
                mt = plane_metrics(z64, q64, y64, mu64, g, U_g, ell_g)
                writer.writerow({
                    "condition": r["condition"], "group": group, "factor": factor,
                    "seed": r["seed"], "target": r["target"], "lam": r["lam"],
                    "dataset": dataset, "cohort": args.cohort,
                    "m": m, "time_index_1based": ti, "center_1based": c,
                    "E_v": mt["E_v"], "sqrt_E_v": mt["sqrt_E_v"],
                    "E_v_c0": mt["E_v_c0"], "E_v_c1": mt["E_v_c1"],
                    "E_d": mt["E_d"], "A_d": mt["A_d"], "A_d_star": mt["A_d_star"],
                    "gain_a": mt["gain_a"],
                    "residual_energy": mt["residual_energy"],
                    "ref_energy": mt["ref_energy"],
                    "identity_check": mt["identity_check"],
                    "chi2": mt["chi2"],
                    "obs_consistency": mt["obs_consistency"],
                    "e_v_den": mt["e_v_den"],
                })
                n_eval += 1
            fout.flush()
            print("evaluated", rd, "({} fields, total {})".format(len(todo), n_eval),
                  flush=True)
    finally:
        fout.close()
    print("cohort {} done: {} field rows, {:.0f}s".format(
        args.cohort, n_eval, time.time() - t0))


if __name__ == "__main__":
    main()
