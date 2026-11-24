#!/usr/bin/env python3
"""Acquire the 96 prospective native test frames (plan section 2.4).

Downloads single-layer native stored frames via the official JHTDB GetCutout
HTTP endpoint (no interpolation, no time averaging):
    isotropic1024coarse  time_index = 40 + 78*m, m = 48..63, z in {129,385,641,897}
    channel              time_index = 32 + 62*m, m = 48..63, y in {75,256}
Files land in <data_root>/raw_prospective/<dataset>/tXXXXXX_<axis><center>.h5
with the same HDF5 layout as the legacy single-layer files.  Every frame is
verified (shape, native coords, finite values, stored time coordinate) and
recorded with sha256 into manifests/test_cohort.json.  Resume-safe: completed
files are skipped; partial files are redownloaded.

The token is read from the environment (JHTDB_TOKEN) or ~/.env and is never
logged or written into any output file.
"""
import argparse
import json
import os
import sys
import time
import urllib.request

import h5py
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.pdt import spec as S                                    # noqa: E402
from src.pdt.data import sha256_file                             # noqa: E402

API = "https://web.idies.jhu.edu/turbulence-svc/cutout/api/local"
ART = "artifacts/planar_divergence_targets_v1"


def load_token():
    tok = os.environ.get("JHTDB_TOKEN", "").strip()
    if tok:
        return tok
    for path in (os.path.expanduser("~/.env"),):
        if os.path.exists(path):
            for line in open(path):
                if line.strip().startswith(("JHTDB_TOKEN", "FR_MSC_JHTDB_TOKEN")):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("JHTDB token not found (env JHTDB_TOKEN or ~/.env)")


def get_cutout(token, dataset, ts, ranges, max_tries=5):
    params = dict(token=token, function="velocity", dataset=dataset,
                  ts=ts, te=ts, stridet=1, stridex=1, stridey=1, stridez=1,
                  filter_width=1)
    for a in "xyz":
        params[a + "s"], params[a + "e"] = ranges[a]
    q = "&".join("{}={}".format(k, v) for k, v in params.items())
    url = API + "?" + q
    last = None
    for attempt in range(max_tries):
        try:
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=600) as r:
                return json.loads(r.read().decode())
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(min(60, 5 * (2 ** attempt)))
    raise RuntimeError("get_cutout failed after {} tries: {}".format(max_tries, last))


def parse_cutout(result):
    fields = result.get("data_vars", {})
    if len(fields) != 1:
        raise ValueError("expected one velocity field")
    da = next(iter(fields.values()))
    dims = list(da["dims"])
    ix = next(i for i, d in enumerate(dims) if d.strip().lower().endswith("x"))
    iy = next(i for i, d in enumerate(dims) if d.strip().lower().endswith("y"))
    iz = next(i for i, d in enumerate(dims) if d.strip().lower().endswith("z"))
    it = next((i for i, d in enumerate(dims) if "time" in d.lower()), None)
    ic = ({0, 1, 2, 3} - {ix, iy, iz, it}).pop() if it is not None else \
        ({0, 1, 2, 3} - {ix, iy, iz}).pop()
    arr = np.asarray(da["data"], dtype=np.float32)
    if it is not None:
        arr = np.take(arr, 0, axis=it)
    arr = arr.transpose(iz, iy, ix, ic)
    coords = {}
    for a, i in (("x", ix), ("y", iy), ("z", iz)):
        dim = dims[i]
        coords[a] = np.asarray(result["coords"][dim]["data"], dtype=np.float64)
    if it is not None:
        coords["time"] = np.asarray(
            result["coords"][dims[it]]["data"], dtype=np.float64)
    return arr, coords


def download_frame(token, data_root, dataset, time_index, center, log):
    sp = S.DATASETS[dataset]
    ax = sp["normal_axis"]
    rel = S.plane_relpath(dataset, time_index, center, prospective=True)
    out_path = os.path.join(data_root, rel)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp = out_path + ".part"
    ranges = {"x": (1, sp["Nx"])}
    # in-plane s axis spans the full native extent
    if ax == "z":
        ranges["y"] = (1, 1024)
        ranges["z"] = (center, center)
    else:
        ranges["y"] = (center, center)
        ranges["z"] = (1, 1536)
    # split large in-plane requests into z(x) tiles to bound response size
    s_axis = "z" if ax == "y" else "y"
    n_tile = 2 if sp["Ns"] * sp["Nx"] >= 1536 * 1024 else 1
    tiles = []
    for i in range(n_tile):
        rr = {k: list(v) for k, v in ranges.items()}
        lo = rr[s_axis]
        seg = (lo[1] - lo[0] + 1) // n_tile
        rr[s_axis] = (lo[0] + i * seg, lo[0] + (i + 1) * seg - 1)
        tiles.append(rr)
    full = None
    coords_all = None
    t_coord = None
    for rr in tiles:
        result = get_cutout(token, dataset, time_index, rr)
        arr, coords = parse_cutout(result)
        if "time" in coords:
            t_coord = float(coords["time"][0])
        full = arr if full is None else np.concatenate([full, arr], axis=(
            1 if ax == "y" else 0))
        coords_all = coords
    # expected final layout
    if ax == "z":
        want = (1, sp["Ns"], sp["Nx"], 3)
    else:
        want = (sp["Ns"], 1, sp["Nx"], 3)
    if full.shape != want:
        raise ValueError("shape {} != {}".format(full.shape, want))
    if not np.isfinite(full).all():
        raise ValueError("non-finite values")
    # verify native center coordinate
    if ax == "z":
        cvals = coords_all.get("z")
    else:
        cvals = coords_all.get("y")
    gd = None
    for gname, gdd in S.GROUPS.items():
        if gdd["dataset"] == dataset and center in gdd["centers_1based"]:
            gd = gdd
    if cvals is not None and gd is not None:
        expected = gd["center_coords"][gd["centers_1based"].index(center)]
        if abs(float(cvals[0]) - expected) > 1e-9:
            raise ValueError("center coord {} != {}".format(cvals[0], expected))
    with h5py.File(tmp, "w") as f:
        f.create_dataset("velocity", data=full)
        if ax == "z" and "y" in coords_all:
            f.create_dataset("y", data=coords_all["y"])
            if cvals is not None:
                f.create_dataset("z", data=cvals)
        elif ax == "y" and "z" in coords_all:
            f.create_dataset("z", data=coords_all["z"])
            if cvals is not None:
                f.create_dataset("y", data=cvals)
        f.attrs["dataset"] = dataset
        f.attrs["time_index_1based"] = time_index
        f.attrs["center_index_1based"] = center
        f.attrs["prospective"] = True
        f.attrs["time_coordinate"] = t_coord if t_coord is not None else -1.0
    os.replace(tmp, out_path)
    return out_path, t_coord


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=os.getcwd())
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--limit", type=int, default=0, help="debug: first N frames")
    args = ap.parse_args()
    token = load_token()
    man = os.path.join(args.project_root, ART, "manifests")
    os.makedirs(man, exist_ok=True)
    cohort_path = os.path.join(man, "test_cohort.json")
    cohort = {"study": "planar_divergence_targets_v1",
              "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "endpoint": API, "frames": []}
    if os.path.exists(cohort_path):
        with open(cohort_path) as f:
            old = json.load(f)
        done = {fr["relpath"]: fr for fr in old.get("frames", [])}
    else:
        done = {}
    n_planned = 0
    for ds, pc in S.PROSPECTIVE.items():
        n_planned += len(pc["m_range"]) * len(pc["centers_1based"])
    n_done = 0
    t0 = time.time()
    for ds, pc in S.PROSPECTIVE.items():
        sp = S.DATASETS[ds]
        for m in pc["m_range"]:
            ti = pc["time_index"][m]
            for c in pc["centers_1based"]:
                rel = S.plane_relpath(ds, ti, c, prospective=True)
                n_done += 1
                if args.limit and n_done > args.limit:
                    break
                path = os.path.join(args.data_root, rel)
                entry = {
                    "dataset": ds, "m": m,
                    "time_index_1based": ti,
                    "t_physical_expected": (ti - 1) * sp["native_dt"],
                    "center_1based": c, "relpath": rel,
                }
                if rel in done and os.path.exists(path) and \
                        done[rel].get("status") == "verified":
                    entry.update(done[rel])
                    cohort["frames"].append(entry)
                    print("skip", rel)
                    continue
                print("download", rel, flush=True)
                path, t_coord = download_frame(token, args.data_root, ds, ti, c, None)
                entry.update({
                    "status": "verified",
                    "sha256": sha256_file(path),
                    "bytes": os.path.getsize(path),
                    "time_coordinate": t_coord,
                    "time_matches_native_dt": (
                        abs(t_coord - entry["t_physical_expected"]) < 1e-6
                        if t_coord is not None else None),
                })
                cohort["frames"].append(entry)
                with open(cohort_path, "w") as f:
                    json.dump(cohort, f, indent=1)
    cohort["n_frames"] = len(cohort["frames"])
    cohort["n_planned"] = n_planned
    cohort["elapsed_seconds"] = time.time() - t0
    with open(cohort_path, "w") as f:
        json.dump(cohort, f, indent=1)
    print("cohort:", cohort["n_frames"], "/", n_planned,
          "elapsed {:.1f}s".format(cohort["elapsed_seconds"]))


if __name__ == "__main__":
    main()
