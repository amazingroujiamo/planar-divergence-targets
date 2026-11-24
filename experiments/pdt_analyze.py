#!/usr/bin/env python3
"""Paired analysis of the prospective test evaluation (plan sections 7-8).

Reads metrics/per_field_prospective.csv (+ optional old_test cohort) and the
formal run records, then writes:
  metrics/per_time.csv, metrics/per_seed.csv, metrics/condition_summary.csv
  metrics/effects.csv            per-condition effects with seed values
  reports/bootstrap.json         block-bootstrap intervals (len 4; sens 2, 8)
  reports/classification.json    plan-8.3 four-way classification per comparison
  reports/convergence_check.json plan-6.4 continuation sensitivity verdicts
  reports/results.md             the internal results report (no paper prose)

Aggregation and interval rules:
  - per-time means: iso averages its four positions first, then times;
    channel positions are separate conditions averaged over times;
  - effects are PAIRED on (seed, time): dE uses the same reference field;
  - R uses condition-level three-seed means;
  - circular block bootstrap over the 16 ordered test times, block length 4,
    5000 replicates, seed 20261013; the SAME time indices are reused for
    every configuration of the same DNS within one replicate (paired);
  - lambda_mid six-condition R_target family gets Bonferroni (alpha/6)
    two-sided percentile intervals; everything else descriptive 95%;
  - classification band: |R| <= 1% with seed-direction agreement rules.
"""
import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.pdt import spec as S                                    # noqa: E402
from src.pdt.spec import ARTIFACTS                               # noqa: E402

BOOT_K = 5000
BOOT_SEED = 20261013
BLOCK = 4
BLOCK_SENS = [2, 8]
BAND = 0.01
BAND_SENS = [0.005, 0.02]
ALPHA = 0.05
PRIMARY_FAMILY = 6


def load_fields(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def per_time_rows(fields, cohort):
    """iso: mean over centers within each time first. channel: each center is
    its own condition; rows are one center per time already."""
    agg = defaultdict(lambda: defaultdict(list))
    for r in fields:
        key = (r["condition"], r["seed"], r["target"], r["lam"])
        agg[key][int(r["m"])].append(r)
    out = []
    for (cond, seed, target, lam), by_m in sorted(agg.items()):
        group = cond.rsplit("_r", 1)[0]
        ds = S.GROUPS[group]["dataset"]
        for m in sorted(by_m):
            rows = by_m[m]
            rec = {"condition": cond, "seed": seed, "target": target,
                   "lam": lam, "m": m, "cohort": cohort}
            for f in ("E_v", "E_d", "A_d", "A_d_star", "gain_a", "chi2",
                      "obs_consistency", "E_v_c0", "E_v_c1"):
                vals = [float(x[f]) for x in rows]
                rec[f] = float(np.mean(vals))
            out.append(rec)
    return out


def mean_over_times(recs, keys=("E_v", "E_d", "A_d", "gain_a")):
    out = {}
    for k in keys:
        out[k] = float(np.mean([r[k] for r in recs])) if recs else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=os.getcwd())
    args = ap.parse_args()
    t0 = time.time()
    art = os.path.join(args.project_root, ARTIFACTS)
    fields = load_fields(os.path.join(art, "metrics", "per_field_prospective.csv"))
    times = per_time_rows(fields, "prospective")
    FLOAT_FIELDS = ("E_v", "E_d", "A_d", "A_d_star", "gain_a", "chi2",
                    "obs_consistency", "E_v_c0", "E_v_c1")

    # ---- per_time / per_seed csv ----
    os.makedirs(os.path.join(art, "metrics"), exist_ok=True)
    with open(os.path.join(art, "metrics", "per_time.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(times[0].keys()))
        w.writeheader()
        w.writerows(times)

    for r in times:
        for f in FLOAT_FIELDS:
            r[f] = float(r[f])
    by_seed = defaultdict(list)
    for r in times:
        by_seed[(r["condition"], r["seed"], r["target"], r["lam"])].append(r)
    per_seed = []
    for k, recs in sorted(by_seed.items()):
        rec = {"condition": k[0], "seed": k[1], "target": k[2], "lam": k[3]}
        rec.update(mean_over_times(recs))
        rec["n_times"] = len(recs)
        per_seed.append(rec)
    with open(os.path.join(art, "metrics", "per_seed.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_seed[0].keys()))
        w.writeheader()
        w.writerows(per_seed)

    # condition-level (mean over seeds) + effects --------------------------
    lams_by_cond = defaultdict(set)
    for r in per_seed:
        lams_by_cond[r["condition"]].add(float(r["lam"]))
    sel_path = os.path.join(art, "manifests", "pilot_weight_selection.json")
    with open(sel_path) as f:
        lams3 = [float(x) for x in json.load(f)["chosen"]]

    Ev = {(r["condition"], r["seed"], r["target"], float(r["lam"])): r["E_v"]
          for r in per_seed}
    Ed = {(r["condition"], r["seed"], r["target"], float(r["lam"])): r["E_d"]
          for r in per_seed}
    Ad = {(r["condition"], r["seed"], r["target"], float(r["lam"])): r["A_d"]
          for r in per_seed}

    effects = []
    conds = sorted({r["condition"] for r in per_seed})
    for cond in conds:
        seeds = sorted({r["seed"] for r in per_seed if r["condition"] == cond})
        base = {s: Ev[(cond, s, "A", 0.0)] for s in seeds}
        base_mean = float(np.mean(list(base.values())))
        row_base = {"condition": cond, "comparison": "baseline",
                    "E_A": base_mean,
                    "seed_E_A": json.dumps(base)}
        effects.append(row_base)
        for lam in sorted(lams_by_cond[cond]):
            if lam == 0.0:
                continue
            for tgt in ("Z", "T"):
                e = {s: Ev[(cond, s, tgt, lam)] for s in seeds}
                emean = float(np.mean(list(e.values())))
                eff = {"condition": cond, "lam": lam, "target": tgt,
                       "E_{}".format(tgt): emean,
                       "seed_E_{}".format(tgt): json.dumps(e),
                       "E_A": base_mean, "seed_E_A": json.dumps(base)}
                eff["Delta_{}".format(tgt)] = emean - base_mean
                eff["R_{}".format(tgt)] = (emean - base_mean) / base_mean
                eff["per_seed_Delta_{}".format(tgt)] = json.dumps(
                    {s: e[s] - base[s] for s in seeds})
                eff["per_seed_R_{}".format(tgt)] = json.dumps(
                    {s: (e[s] - base[s]) / base[s] for s in seeds})
                effects.append(eff)
            ez = {s: Ev[(cond, s, "Z", lam)] for s in seeds}
            et = {s: Ev[(cond, s, "T", lam)] for s in seeds}
            eff = {"condition": cond, "lam": lam, "target": "T_vs_Z"}
            eff["E_T_minus_E_Z"] = float(np.mean([et[s] - ez[s] for s in seeds]))
            eff["per_seed"] = json.dumps({s: et[s] - ez[s] for s in seeds})
            # 保留旧 per_seed 绝对差值列；相对阈值必须使用各自基线归一化后的值。
            eff["per_seed_Delta_target"] = eff["per_seed"]
            eff["per_seed_R_target"] = json.dumps(
                {s: (et[s] - ez[s]) / base[s] for s in seeds})
            eff["R_target"] = eff["E_T_minus_E_Z"] / base_mean
            effects.append(eff)
    with open(os.path.join(art, "metrics", "effects.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted(
            {k for e in effects for k in e}))
        w.writeheader()
        w.writerows(effects)

    # ---- paired block bootstrap ------------------------------------------
    rng = np.random.default_rng(BOOT_SEED)
    rng_state = rng.bit_generator.state

    def series_for(cond, target, lam):
        """(n_seed, n_time) E_v array, times in ascending order."""
        seeds = sorted({r["seed"] for r in times if r["condition"] == cond})
        mat = []
        for s in seeds:
            rows = sorted([r for r in times
                           if r["condition"] == cond and r["seed"] == s
                           and r["target"] == target
                           and float(r["lam"]) == lam],
                          key=lambda r: r["m"])
            mat.append([r["E_v"] for r in rows])
        return np.array(mat), seeds

    def block_indices(n_times, block, k, rng):
        n_blocks = int(np.ceil(n_times / block))
        starts = rng.integers(0, n_times, size=(k, n_blocks))
        idx = []
        for row in starts:
            ii = [(s + j) % n_times for s in row for j in range(block)]
            idx.append(ii[:n_times])
        return np.asarray(idx, dtype=np.int64)

    def bootstrap_condition(cond, lam, block=BLOCK, k=BOOT_K):
        A, _ = series_for(cond, "A", 0.0)
        Z, _ = series_for(cond, "Z", lam)
        T, _ = series_for(cond, "T", lam)
        n_times = A.shape[1]
        rng2 = np.random.default_rng(BOOT_SEED)
        idx = block_indices(n_times, block, k, rng2)
        A_b = A[:, idx]                      # (seed, k, t)
        Z_b = Z[:, idx]
        T_b = T[:, idx]
        A_m = A_b.mean(axis=(0, 2))
        Z_m = Z_b.mean(axis=(0, 2))
        T_m = T_b.mean(axis=(0, 2))
        R_Z = (Z_m - A_m) / A_m
        R_T = (T_m - A_m) / A_m
        R_tgt = (T_m - Z_m) / A_m
        dZ = (Z_b - A_b).mean(axis=(0, 2))
        dT = (T_b - A_b).mean(axis=(0, 2))
        dtgt = (T_b - Z_b).mean(axis=(0, 2))
        return {"R_Z": R_Z, "R_T": R_T, "R_target": R_tgt,
                "Delta_Z": dZ, "Delta_T": dT, "Delta_target": dtgt}

    boot = {"settings": {"replicates": BOOT_K, "seed": BOOT_SEED,
                         "block": BLOCK, "block_sensitivity": BLOCK_SENS,
                         "band": BAND, "band_sensitivity": BAND_SENS,
                         "level": 0.95, "primary_family": PRIMARY_FAMILY},
            "conditions": {}}
    for cond in conds:
        boot["conditions"][cond] = {}
        for lam in [lams3[1]] + sorted(lams_by_cond[cond] - {0.0, lams3[1]}):
            b = bootstrap_condition(cond, lam)
            entry = {}
            for key in ("R_Z", "R_T", "R_target", "Delta_Z", "Delta_T",
                        "Delta_target"):
                q = np.percentile(b[key], [2.5, 50, 97.5])
                # Bonferroni family (6 conditions, lambda_mid R_target is the
                # primary family): per-interval two-sided level 1-alpha/6
                tail = 100.0 * (ALPHA / PRIMARY_FAMILY) / 2.0
                entry[key] = {
                    "desc_lo95": float(q[0]), "median": float(q[1]),
                    "desc_hi95": float(q[2]),
                    "bonf_lo": float(np.percentile(b[key], tail)),
                    "bonf_hi": float(np.percentile(b[key], 100.0 - tail)),
                }
            for bsens in BLOCK_SENS:
                b2 = bootstrap_condition(cond, lam, block=bsens)
                entry["block{}_R_target_ci".format(bsens)] = [
                    float(np.percentile(b2["R_target"], 2.5)),
                    float(np.percentile(b2["R_target"], 97.5))]
            boot["conditions"][cond][format(lam, ".0e")] = entry

    # classification (plan 8.3) --------------------------------------------
    # Interval convention (plan 8.2): the PRIMARY family — lambda_mid
    # R_target over the six conditions — uses Bonferroni-adjusted intervals;
    # every other comparison uses descriptive 95% intervals.  Seed stability
    # (plan 8.3 "especially seed-direction conflict"): a within-band interval
    # is only called practical equivalence when no seed deviates beyond the
    # band in conflicting directions; otherwise the comparison stays
    # uncertain_or_unstable (narrow time intervals do not by themselves
    # certify stability across training randomness).
    # “正负方向混合且至少一种子超带”是验收阶段对计划稳定性原则的细化，
    # 不是原计划已经逐字预先规定的定量规则。
    classification = {}
    for e in effects:
        if "lam" not in e:
            continue
        cond, lam = e["condition"], e["lam"]
        key = None
        if "R_Z" in e:
            key = "Z"
        elif "R_T" in e:
            key = "T"
        elif "R_target" in e:
            key = "target"
        if key is None:
            continue
        primary = (key == "target" and lam == lams3[1])
        if key in ("Z", "T"):
            r = e["R_{}".format(key)]
            seed_rs = {k: v for k, v in
                       json.loads(e["per_seed_R_{}".format(key)]).items()}
            ci = boot["conditions"][cond][format(lam, ".0e")]
            c = ci["R_{}".format(key)]
        else:
            r = e["R_target"]
            seed_rs = json.loads(e["per_seed_R_target"])
            ci = boot["conditions"][cond][format(lam, ".0e")]
            c = ci["R_target"]
        lo, hi = ((c["bonf_lo"], c["bonf_hi"]) if primary
                  else (c["desc_lo95"], c["desc_hi95"]))
        seeds_dir = [np.sign(v) for v in seed_rs.values()]
        same_dir = len(set(seeds_dir)) == 1 and seeds_dir[0] != 0
        seed_conflict = (len({d for d in seeds_dir if d != 0}) > 1
                         and max(abs(v) for v in seed_rs.values()) > BAND)
        if hi < -BAND and same_dir and r < 0:
            cls = "improvement"
        elif lo > BAND and same_dir and r > 0:
            cls = "degradation"
        elif -BAND <= lo and hi <= BAND and not seed_conflict:
            cls = "practical_equivalence"
        else:
            cls = "uncertain_or_unstable"
        classification["{}|{}|{}".format(cond, key, format(lam, ".0e"))] = {
            "R": r, "ci95": [lo, hi],
            "interval_type": "bonferroni_95" if primary else "descriptive_95",
            "seed_R": seed_rs, "seed_directions": seeds_dir,
            "seed_conflict_beyond_band": bool(seed_conflict),
            "class": cls}

    # convergence continuation check (plan 6.4) -----------------------------
    conv = {}
    for group in S.FORMAL_GROUPS:
        for factor in S.FACTORS:
            cond = "{}_r{}".format(group, factor)
            entry = {}
            for target, lamtag in (("A", "0"), ("Z", format(lams3[2], ".17g")),
                                   ("T", format(lams3[2], ".17g"))):
                wdir = ("shared" if target == "A"
                        else "lam_" + format(lams3[2], ".0e"))
                base_curve = os.path.join(art, "runs", "formal", cond, "s11",
                                          target, wdir, "curve.jsonl")
                cont_curve = os.path.join(art, "runs", "formal", cond, "s11",
                                          target, wdir, "continuation",
                                          "curve.jsonl")
                if not (os.path.exists(base_curve) and os.path.exists(cont_curve)):
                    continue
                v300 = None
                with open(base_curve) as f:
                    for line in f:
                        r = json.loads(line)
                        v300 = r["val_E_v"]
                v600 = None
                with open(cont_curve) as f:
                    for line in f:
                        r = json.loads(line)
                        v600 = r["val_E_v"]
                entry[target] = {"val_E_v_300": v300, "val_E_v_600": v600,
                                 "rel_change": (v600 - v300) / v300 if v300 else None}
            if "Z" in entry and "T" in entry:
                r300 = entry["T"]["val_E_v_300"] - entry["Z"]["val_E_v_300"]
                r600 = entry["T"]["val_E_v_600"] - entry["Z"]["val_E_v_600"]
                entry["order_flip"] = bool(r300 * r600 < 0)
                entry["sensitive"] = bool(
                    entry["order_flip"]
                    or any(abs(entry[t]["rel_change"]) > 0.01
                           for t in ("A", "Z", "T") if t in entry))
            conv[cond] = entry

    for name, obj in (("bootstrap", boot), ("classification", classification),
                      ("convergence_check", conv)):
        with open(os.path.join(art, "reports", name + ".json"), "w") as f:
            json.dump(obj, f, indent=1)
    print("analysis done in {:.0f}s".format(time.time() - t0))
    print("conditions:", conds)
    for k, v in list(classification.items())[:8]:
        print(k, v["class"], round(v["R"], 4))


if __name__ == "__main__":
    main()
