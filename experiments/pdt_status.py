#!/usr/bin/env python3
"""One-glance status of all PDT runs (pilot/formal/continuation)."""
import glob
import json
import os
import sys

ART = "artifacts/planar_divergence_targets_v1"


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    n_ok = n_fail = n_live = 0
    for stage in ("pilot", "formal"):
        for summ in sorted(glob.glob(os.path.join(
                root, ART, "runs", stage, "**", "summary.json"),
                recursive=True)):
            with open(summ) as f:
                s = json.load(f)
            n_ok += 1
            tag = s.get("tag") or os.path.relpath(summ, os.path.join(root, ART, "runs"))
            print("[{}] {} ep={}/{} best={:.6g} @{} reason={}".format(
                stage, os.path.dirname(os.path.relpath(
                    summ, os.path.join(root, ART, "runs"))),
                s.get("epochs_run"), s.get("max_epochs") or 300,
                s.get("best_val_E_v", float("nan")),
                s.get("best_epoch"), s.get("stop_reason")))
        for fail in glob.glob(os.path.join(
                root, ART, "runs", stage, "**", "failure.json"),
                recursive=True):
            n_fail += 1
            print("[{}] FAIL {}".format(stage, os.path.relpath(
                fail, os.path.join(root, ART, "runs"))))
    # in-flight: run dirs with curve.jsonl but no summary yet
    for curve in glob.glob(os.path.join(
            root, ART, "runs", "**", "curve.jsonl"), recursive=True):
        d = os.path.dirname(curve)
        if os.path.exists(os.path.join(d, "summary.json")):
            continue
        n_live += 1
        try:
            lines = open(curve).read().strip().splitlines()
            last = json.loads(lines[-1])
            print("[LIVE] {} epoch {}".format(os.path.relpath(
                d, os.path.join(root, ART, "runs")), last["epoch"]))
        except Exception:
            pass
    print("\ntotals: ok={} fail={} live={}".format(n_ok, n_fail, n_live))


if __name__ == "__main__":
    main()
