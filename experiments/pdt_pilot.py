#!/usr/bin/env python3
"""Weight pilot (plan section 5): 30 short runs, optional boundary extension
(+8).  Conditions iso_xy r=4 and channel_y075 r=4, seed 900 (not a formal
replicate), max 50 epochs, lr 1e-3 (the formal schedule's initial rate).

Usage:
  python pdt_pilot.py --plan                 # write the task manifest
  python pdt_pilot.py --worker --shard i --nshards N   # run one shard
Records every run under runs/pilot/; selection happens in
pdt_select_weights.py, never by test performance.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.pdt import spec as S                                    # noqa: E402
from src.pdt.train import train_run, ARTIFACTS                   # noqa: E402

PILOT_GROUPS = [("iso_xy", 4), ("channel_y075", 4)]
LAMBDA_GRID = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
BOUNDARY = [1e-7, 10.0]
PILOT_EPOCHS = 50


def build_tasks(boundary=False):
    tasks = []
    for g, r in PILOT_GROUPS:
        tasks.append({"group": g, "factor": r, "target": "A", "lam": 0.0})
        for lam in LAMBDA_GRID:
            tasks.append({"group": g, "factor": r, "target": "Z", "lam": lam})
            tasks.append({"group": g, "factor": r, "target": "T", "lam": lam})
    if boundary:
        for g, r in PILOT_GROUPS:
            for lam in BOUNDARY:
                tasks.append({"group": g, "factor": r, "target": "Z", "lam": lam})
                tasks.append({"group": g, "factor": r, "target": "T", "lam": lam})
    return tasks


def run_dir_for(t, project_root):
    from src.pdt.train import condition_of
    cond = condition_of(t["group"], t["factor"])
    w = "shared" if t["target"] == "A" else "lam_" + format(t["lam"], ".0e")
    return os.path.join(project_root, ARTIFACTS, "runs", "pilot", cond,
                        "s900", t["target"], w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=os.getcwd())
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--boundary", action="store_true",
                    help="include the 1e-7 / 10 boundary extension runs")
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    args = ap.parse_args()
    tasks = build_tasks(args.boundary)
    man_path = os.path.join(args.project_root, ARTIFACTS, "manifests",
                            "pilot_tasks{}.json".format(
                                "_boundary" if args.boundary else ""))
    if args.plan:
        os.makedirs(os.path.dirname(man_path), exist_ok=True)
        with open(man_path, "w") as f:
            json.dump({"tasks": tasks, "seed": S.PILOT_SEED,
                       "max_epochs": PILOT_EPOCHS,
                       "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                    time.gmtime())}, f, indent=1)
        print("wrote", man_path, len(tasks), "tasks")
        return
    if not args.worker:
        print("nothing to do: use --plan or --worker")
        return
    mine = [t for i, t in enumerate(tasks) if i % args.nshards == args.shard]
    print("shard {} of {} -> {} tasks".format(args.shard, args.nshards, len(mine)))
    results = []
    for t in mine:
        rd = run_dir_for(t, args.project_root)
        if os.path.exists(os.path.join(rd, "summary.json")) or \
                os.path.exists(os.path.join(rd, "failure.json")):
            print("skip existing", rd)
            continue
        print("RUN", t, flush=True)
        try:
            s = train_run(args.project_root, args.data_root, t["group"],
                          t["factor"], S.PILOT_SEED, t["target"], t["lam"],
                          max_epochs=PILOT_EPOCHS, device="cuda", pilot=True)
            results.append(s)
            print("done", t, "best_val_E_v={:.6g} epochs={} reason={}".format(
                s["best_val_E_v"], s["epochs_run"], s["stop_reason"]))
        except Exception as e:  # noqa: BLE001
            print("TASK ERROR", t, repr(e), flush=True)
            results.append({"task": t, "error": repr(e)})


if __name__ == "__main__":
    main()
