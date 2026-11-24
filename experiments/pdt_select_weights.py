#!/usr/bin/env python3
"""Select the three formal weights from the pilot (plan section 5.2).

S(lambda) = 1 - median_g A_{Z,g}(lambda) / A_{A,g}, with A measured at each
run's best-validation-E_v checkpoint epoch (never by test performance).
Valid candidates require BOTH targets to have completed (no technical
failure) in BOTH pilot conditions.  Three strictly increasing weights
minimize |S-0.10| + |S-0.30| + |S-0.60|; ties break to the lexicographically
smaller triple.  Writes manifests/pilot_weight_selection.json with every
pilot number, the response curve, the selection and (if needed) a range-
coverage verdict.
"""
import argparse
import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.pdt import spec as S                                    # noqa: E402
from src.pdt.train import ARTIFACTS                              # noqa: E402

TARGETS = [0.10, 0.30, 0.60]
PILOT_GROUPS = ["iso_xy_r4", "channel_y075_r4"]
LAMBDA_GRID = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]


def read_run(project_root, cond, target, lam):
    w = "shared" if target == "A" else "lam_" + format(lam, ".0e")
    rd = os.path.join(project_root, ARTIFACTS, "runs", "pilot", cond,
                      "s900", target, w)
    summ_path = os.path.join(rd, "summary.json")
    fail_path = os.path.join(rd, "failure.json")
    if os.path.exists(fail_path):
        return {"status": "technical_fail", "dir": rd}
    if not os.path.exists(summ_path):
        return {"status": "missing", "dir": rd}
    with open(summ_path) as f:
        summ = json.load(f)
    best_epoch = summ.get("best_epoch")
    row = None
    if best_epoch is not None:
        with open(os.path.join(rd, "curve.jsonl")) as f:
            for line in f:
                r = json.loads(line)
                if r["epoch"] == best_epoch:
                    row = r
                    break
    return {"status": "ok", "summary": summ, "best_row": row, "dir": rd}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=os.getcwd())
    ap.add_argument("--boundary", action="store_true",
                    help="include 1e-7 / 10 extension runs in the grid")
    args = ap.parse_args()
    grid = LAMBDA_GRID + ([1e-7, 10.0] if args.boundary else [])
    grid = sorted(grid)

    runs = {}
    for cond in PILOT_GROUPS:
        runs[(cond, "A", 0.0)] = read_run(args.project_root, cond, "A", 0.0)
        for lam in grid:
            runs[(cond, "Z", lam)] = read_run(args.project_root, cond, "Z", lam)
            runs[(cond, "T", lam)] = read_run(args.project_root, cond, "T", lam)

    # baseline divergence amplitudes
    A_base = {}
    for cond in PILOT_GROUPS:
        r = runs[(cond, "A", 0.0)]
        if r["status"] != "ok" or r["best_row"] is None:
            raise SystemExit("baseline run unavailable for {}".format(cond))
        A_base[cond] = r["best_row"]["val_A_d"]
    print("baseline A_d:", A_base)

    S_of = {}
    status_of = {}
    for lam in grid:
        ratios = []
        ok = True
        detail = {}
        for cond in PILOT_GROUPS:
            rz = runs[(cond, "Z", lam)]
            rt = runs[(cond, "T", lam)]
            if rz["status"] != "ok" or rz["best_row"] is None:
                ok = False
                detail[cond] = "Z {}".format(rz["status"])
                continue
            if rt["status"] != "ok":
                detail[cond] = "T {}".format(rt["status"])
                ok = False
                continue
            ratios.append(rz["best_row"]["val_A_d"] / A_base[cond])
        status_of[lam] = {"valid": ok, "detail": detail}
        S_of[lam] = 1.0 - float(np.median(ratios)) if ratios else None
        print("lam={:.0e} S={} valid={} {}".format(
            lam, S_of[lam], ok, detail))

    valid = [lam for lam in grid if status_of[lam]["valid"]
             and S_of[lam] is not None and np.isfinite(S_of[lam])]
    best = None
    for combo in itertools.combinations(sorted(valid), 3):
        s = [S_of[c] for c in combo]
        cost = (abs(s[0] - TARGETS[0]) + abs(s[1] - TARGETS[1])
                + abs(s[2] - TARGETS[2]))
        key = (cost, combo)
        if best is None or key < best[0]:
            best = (key, combo, cost)
    if best is None:
        raise SystemExit("no valid weight triple")
    (cost, combo), chosen, cost_val = best

    # range coverage verdict (plan 5.2: extension only if clearly insufficient)
    s_vals = [S_of[c] for c in sorted(valid)]
    coverage = {
        "max_S": max(s_vals) if s_vals else None,
        "min_S": min(s_vals) if s_vals else None,
        "upper_extension_recommended": (max(s_vals) < 0.35) if s_vals else True,
        "lower_extension_recommended": (min(s_vals) > 0.15) if s_vals else False,
    }

    out = {
        "study": "planar_divergence_targets_v1",
        "rule": ("S(lambda)=1-median_g A_Z,g(lambda)/A_A,g at each run's "
                 "best val E_v epoch; triple minimizing |S-.1|+|S-.3|+|S-.6|; "
                 "lexicographic smaller-weight tie break"),
        "targets": TARGETS,
        "grid": grid,
        "baseline_A_d": A_base,
        "S_of_lambda": {format(k, ".0e"): v for k, v in S_of.items()},
        "status_of_lambda": {format(k, ".0e"): v for k, v in status_of.items()},
        "chosen": [format(c, ".0e") for c in chosen],
        "chosen_S": [S_of[c] for c in chosen],
        "cost": cost_val,
        "coverage": coverage,
        "runs": {"{}_{}_{}".format(c, t, format(l, ".0e")): {
            "status": r["status"],
            **({"best_epoch": r["summary"].get("best_epoch"),
                "best_val_E_v": r["summary"].get("best_val_E_v"),
                "val_A_d_at_best": (r["best_row"] or {}).get("val_A_d"),
                "val_E_d_at_best": (r["best_row"] or {}).get("val_E_d"),
                "stop_reason": r["summary"].get("stop_reason")}
               if r["status"] == "ok" else {})}
            for (c, t, l), r in runs.items()},
    }
    path = os.path.join(args.project_root, ARTIFACTS, "manifests",
                        "pilot_weight_selection.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print("\nchosen:", [format(c, ".0e") for c in chosen],
          "S:", [round(S_of[c], 4) for c in chosen], "cost", cost_val)
    print("coverage:", coverage)
    print("written", path)


if __name__ == "__main__":
    main()
