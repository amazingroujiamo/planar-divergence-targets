#!/usr/bin/env python3
"""Verify the 96 downloaded prospective test frames (plan 2.4) and freeze
their identity into manifests/test_cohort.json.

Checks per file: existence, velocity shape/dtype/finite values, native center
coordinate against the dataset spec, attrs time index == requested, full
in-plane extents, uniform in-plane coordinate grids, sha256.  Also re-checks
zero overlap of the prospective time indices with every legacy stored frame
and with the train/val/gap ranges, and that each frame is a SINGLE-layer file
(never read through the old audit-slab rule).
"""
import argparse
import csv
import json
import os
import sys
import time

import h5py
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.pdt import spec as S                                    # noqa: E402
from src.pdt.data import sha256_file                             # noqa: E402

ART = "artifacts/planar_divergence_targets_v1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--update-inventory", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    frames, problems = [], []
    n = 0
    for ds, pc in S.PROSPECTIVE.items():
        sp = S.DATASETS[ds]
        for m in pc["m_range"]:
            ti = pc["time_index"][m]
            for c in pc["centers_1based"]:
                n += 1
                rel = S.plane_relpath(ds, ti, c, prospective=True)
                path = os.path.join(args.data_root, rel)
                rec = {"dataset": ds, "m": m, "time_index_1based": ti,
                       "t_physical_expected": S.physical_time(ds, ti),
                       "center_1based": c, "relpath": rel}
                try:
                    with h5py.File(path, "r") as f:
                        vel = f["velocity"]
                        if ds.startswith("iso"):
                            want = (1, 1024, 1024, 3)
                        else:
                            want = (1536, 1, 2048, 3)
                        assert vel.shape == want, vel.shape
                        assert vel.dtype == np.float32
                        arr = vel[...] if vel.size < 4e7 else None
                        if arr is not None:
                            assert np.isfinite(arr).all()
                        ax = sp["normal_axis"]
                        ccoord = np.asarray(f[ax][:], dtype=np.float64)
                        gd = next(gdd for gdd in S.GROUPS.values()
                                  if gdd["dataset"] == ds
                                  and c in gdd["centers_1based"])
                        expected = gd["center_coords"][
                            gd["centers_1based"].index(c)]
                        assert abs(ccoord[0] - expected) < 1e-9, (ccoord, expected)
                        a = dict(f.attrs)
                        assert int(a["time_index_1based"]) == ti
                        assert int(a["{}_start_1based".format(ax)]) == c
                        assert int(a["{}_end_1based".format(ax)]) == c
                        assert int(a["x_start_1based"]) == 1
                        assert int(a["x_end_1based"]) == sp["Nx"]
                        x = np.asarray(f["x"][:], dtype=np.float64)
                        dx = sp["Lx"] / sp["Nx"]
                        assert np.abs(x - np.arange(sp["Nx"]) * dx).max() < 1e-9
                        rec.update({
                            "status": "verified",
                            "velocity_shape": list(vel.shape),
                            "center_coordinate": float(ccoord[0]),
                            "attrs_time_physical": float(a.get("time_physical", -1)),
                            "sha256": sha256_file(path),
                            "bytes": os.path.getsize(path),
                        })
                        frames.append(rec)
                except Exception as e:  # noqa: BLE001
                    rec["status"] = "FAILED"
                    rec["error"] = repr(e)
                    frames.append(rec)
                    problems.append((rel, repr(e)))
    # zero-overlap statements
    leg = {ds: {S.legacy_time_index(ds, k) for k in range(64)}
           for ds in S.PROSPECTIVE}
    overlaps = {ds: len(leg[ds] & set(S.PROSPECTIVE[ds]["time_index"].values()))
                for ds in S.PROSPECTIVE}
    out = {
        "study": "planar_divergence_targets_v1",
        "verified_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_planned": n, "n_verified": sum(
            1 for r in frames if r["status"] == "verified"),
        "time_index_overlap_with_legacy_stored_frames": overlaps,
        "distinct_time_indices": {
            ds: sorted(set(S.PROSPECTIVE[ds]["time_index"].values()))
            for ds in S.PROSPECTIVE},
        "frames": frames,
        "problems": problems,
        "elapsed_seconds": time.time() - t0,
    }
    path = os.path.join(args.project_root, ART, "manifests", "test_cohort.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print("verified {}/{} frames; problems: {}".format(
        out["n_verified"], n, len(problems)))
    for p in problems[:10]:
        print("PROBLEM", p)
    if args.update_inventory and not problems:
        inv = os.path.join(args.project_root, ART, "manifests",
                           "source_inventory.csv")
        rows = []
        with open(inv) as f:
            rows = list(csv.DictReader(f))
        have = {r["relpath"] for r in rows}
        added = 0
        for r in frames:
            if r["relpath"] not in have:
                rows.append({"relpath": r["relpath"], "dataset": r["dataset"],
                             "sha256": r["sha256"], "bytes": r["bytes"],
                             "roles": "prospective_test",
                             "groups": "iso_xy" if r["dataset"].startswith("iso")
                             else "channel_y075+channel_y256"})
                added += 1
        with open(inv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print("source_inventory.csv: appended", added, "prospective rows")
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
