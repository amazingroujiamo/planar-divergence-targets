"""Training entry point for the planar divergence targets experiment.

Implements plan sections 3-6:
  - U_g / ell_g frozen from manifests/train_stats.json (never re-estimated
    silently); mu(y) from the LR observation per plane;
  - paired initialization: one init.pt per (condition, seed) shared by all
    seven configurations, plus identical per-epoch sample order via an
    independent numpy RNG stream seeded once per (condition, seed);
  - fixed LR schedule; no early stopping in the formal budget; best
    checkpoint = strict minimum epoch-wise validation E_v (equal-time rule),
    ties resolved to the earlier epoch;
  - per-epoch curve records all loss components, grad norms, divergence
    diagnostics, timing and peak device memory;
  - non-finite losses / metrics / parameters are technical failures with a
    failure.json record; the best checkpoint so far is preserved;
  - continuation runs load last.pt (model + optimizer + RNG state) and write
    into a continuation/ subdirectory, leaving the 300-epoch records intact.
"""
import json
import os
import time

import numpy as np
import torch

from . import spec as S
from .data import PlaneStore, sha256_file, read_center_plane
from .losses import loss_terms
from .metrics import equal_time_mean, divergence_metrics
from .operator_torch import SpectralDivergence
from .model_dsc_ms import DSCMS, replicate_lr_to_hr, EXPECTED_PARAM_COUNT

ARTIFACTS = S.ARTIFACTS

LR_SCHEDULE = [(1, 150, 1e-3), (151, 250, 3e-4), (251, 300, 1e-4)]


def lr_at(epoch, schedule=None):
    for lo, hi, lr in (schedule or LR_SCHEDULE):
        if lo <= epoch <= hi:
            return lr
    return (schedule or LR_SCHEDULE)[-1][2]


def load_train_stats(project_root):
    with open(os.path.join(project_root, ARTIFACTS, "manifests",
                           "train_stats.json")) as f:
        return json.load(f)


def condition_of(group, factor):
    return "{}_r{}".format(group, factor)


def _cache_path(project_root, group, factor, role, stage):
    return os.path.join(project_root, ARTIFACTS, "cache",
                        "{}_r{}_{}_{}.npz".format(group, factor, role, stage))


def get_store(project_root, data_root, group, factor, role, stats, device="cpu",
              with_dref=True):
    """PlaneStore with an on-disk cache; identity = source file sha256 list
    + U_g + ell_g + operator id, verified on every load."""
    gd = S.GROUPS[group]
    ds = gd["dataset"]
    U_g = stats["groups"][group]["U_g"]
    ell_g = stats["groups"][group]["ell_g"]
    samples = S.group_samples(group, role)
    src_files = []
    for (m, c) in samples:
        ti = S.legacy_time_index(ds, m)
        src_files.append(S.plane_relpath(ds, ti, c))
    cp = _cache_path(project_root, group, factor, role, "v1")
    identity = {
        "group": group, "factor": factor, "role": role,
        "source_files": src_files,
        "source_sha256": None,   # filled on first build
        "U_g": U_g, "ell_g": ell_g,
        "operator_id": "pdt.spectral_divergence.v1",
    }
    if os.path.exists(cp):
        try:
            with np.load(cp, allow_pickle=False) as z:
                meta = json.loads(str(z["meta"][()]))
                if (meta["source_files"] == src_files and meta["U_g"] == U_g
                        and meta["ell_g"] == ell_g and meta["factor"] == factor
                        and meta["with_dref"] == with_dref):
                    store = PlaneStore.__new__(PlaneStore)
                    store.group, store.factor = group, factor
                    store.U_g, store.ell_g = U_g, ell_g
                    store.grid = S.group_grid(group)
                    store.items = []
                    n = int(z["n"])
                    for i in range(n):
                        item = {
                            "m": int(z["m_{}".format(i)]),
                            "center": int(z["c_{}".format(i)]),
                            "info": json.loads(str(z["info_{}".format(i)][()])),
                            "q64": z["q64_{}".format(i)],
                            "mu64": z["mu64_{}".format(i)],
                            "denom64": float(z["denom_{}".format(i)]),
                            "q32": z["q32_{}".format(i)],
                            "y32": z["y32_{}".format(i)],
                            "dref32": z["dref32_{}".format(i)] if with_dref else None,
                        }
                        store.items.append(item)
                    store.cache_identity = meta
                    return store
        except Exception:
            pass  # fall through and rebuild
    store = PlaneStore(data_root, group, factor, samples, U_g, ell_g,
                       device_for_prefetch=device, with_dref=with_dref)
    os.makedirs(os.path.dirname(cp), exist_ok=True)
    payload = {"n": len(store.items), "meta": json.dumps(
        dict(identity, source_sha256="see_source_inventory", factor=factor,
             with_dref=with_dref))}
    for i, it in enumerate(store.items):
        payload["m_{}".format(i)] = it["m"]
        payload["c_{}".format(i)] = it["center"]
        payload["info_{}".format(i)] = json.dumps(it["info"])
        payload["q64_{}".format(i)] = it["q64"]
        payload["mu64_{}".format(i)] = it["mu64"]
        payload["denom_{}".format(i)] = it["denom64"]
        payload["q32_{}".format(i)] = it["q32"]
        payload["y32_{}".format(i)] = it["y32"]
        if with_dref:
            payload["dref32_{}".format(i)] = it["dref32"]
    np.savez_compressed(cp, **payload)
    return store


def _param_grad_norm(model):
    s = 0.0
    for p in model.parameters():
        if p.grad is not None:
            s += float(p.grad.detach().pow(2).sum())
    return float(np.sqrt(s))


def _init_params_finite(model):
    return all(bool(torch.isfinite(p).all()) for p in model.parameters())


def _validate(model, val_store, div32, div_grid, device, dataset, U_g):
    """Equal-time validation E_v plus divergence diagnostics (plan 7).

    Fast in-training path: E_v via float64 GPU reduction (the checkpoint
    metric; numerically the same sum as the numpy version up to summation
    order).  Divergence diagnostics use the SAME float32 spectral operator as
    the training loss in the dimensionless domain, where A_d = rms(D_nd z'),
    A_d* = rms(D_nd q*'), E_d = mean((D_nd z' - D_nd q*')^2).  The FP64 numpy
    operator stays the reference for acceptance checks and the frozen P4
    test evaluation (pdt_evaluate)."""
    model.eval()
    per_plane = {}
    diag = {"E_d": [], "A_d": [], "A_d_star": []}
    with torch.no_grad():
        for it in val_store.items:
            x = torch.from_numpy(np.ascontiguousarray(it["y32"]))[None].to(device)
            x = replicate_lr_to_hr(x, val_store.factor)
            out = model(x)[0].float()
            mu = torch.tensor(it["mu64"], device=out.device,
                              dtype=torch.float64)[:, None, None]
            q64 = torch.tensor(it["q64"], device=out.device,
                               dtype=torch.float64)
            z64 = float(U_g) * out.double() + mu
            num = float((z64 - q64).pow(2).sum())
            ev = num / it["denom64"]
            per_plane[(it["m"], it["center"])] = ev
            dz = div32(out[None])[0]
            dref = torch.from_numpy(np.ascontiguousarray(it["dref32"])).to(device)
            diag["A_d_star"].append(float(dref.double().pow(2).mean().sqrt()))
            diag["A_d"].append(float(dz.double().pow(2).mean().sqrt()))
            diag["E_d"].append(float((dz - dref).double().pow(2).mean()))
    val = equal_time_mean(per_plane, dataset)
    return val, {k: float(np.mean(v)) for k, v in diag.items()}


def _write_failure(out_dir, reason, epoch, detail, config):
    rec = {"status": "technical_fail", "reason": reason, "epoch": epoch,
           "detail": detail, "config": config}
    with open(os.path.join(out_dir, "failure.json"), "w") as f:
        json.dump(rec, f, indent=1)
    return rec


def train_run(project_root, data_root, group, factor, seed, target, lam,
              max_epochs=300, device="cuda", pilot=False,
              schedule=None, tag=None, max_hours=None, seed_for_pairs=None):
    """Train one configuration.  Returns the summary dict.

    Paired identity: init.pt and the per-epoch sample order are keyed by
    (condition, seed) via seed_for_pairs (defaults to seed) so the seven
    configurations of one (condition, seed) share both.
    """
    stats = load_train_stats(project_root)
    gd = S.GROUPS[group]
    dataset = gd["dataset"]
    g = S.group_grid(group)
    U_g = stats["groups"][group]["U_g"]
    ell_g = stats["groups"][group]["ell_g"]
    cond = condition_of(group, factor)
    if target == "A":
        assert lam == 0.0, "baseline must have lambda=0"
    stage = "pilot" if pilot else "formal"
    weight_dir = "shared" if target == "A" else "lam_" + format(float(lam), ".0e")
    if tag is not None:
        run_dir = os.path.join(project_root, ARTIFACTS, "runs", stage, tag)
    else:
        run_dir = os.path.join(project_root, ARTIFACTS, "runs", stage,
                               cond, "s{}".format(seed), target, weight_dir)
    if os.path.exists(os.path.join(run_dir, "summary.json")) or \
            os.path.exists(os.path.join(run_dir, "failure.json")):
        raise RuntimeError("run record already exists: {}".format(run_dir))
    os.makedirs(run_dir, exist_ok=True)

    pair_seed = seed_for_pairs if seed_for_pairs is not None else seed
    init_path = os.path.join(project_root, ARTIFACTS, "runs", stage, cond,
                             "pair_seeds", "s{}".format(pair_seed), "init.pt")
    config = {
        "study": "planar_divergence_targets_v1", "stage": stage,
        "group": group, "factor": factor, "condition": cond,
        "seed": seed, "pair_seed": pair_seed, "target": target, "lam": float(lam),
        "max_epochs": max_epochs, "device": device,
        "lr_schedule": schedule or LR_SCHEDULE,
        "U_g": U_g, "ell_g": ell_g, "dataset": dataset,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(os.path.join(run_dir, "run_meta.json"), "w") as f:
        json.dump(config, f, indent=1)

    torch.manual_seed(seed)
    if not os.path.exists(init_path):
        os.makedirs(os.path.dirname(init_path), exist_ok=True)
        model = DSCMS()
        torch.save(model.state_dict(), init_path)
    state = torch.load(init_path, map_location="cpu", weights_only=True)
    init_sha = sha256_file(init_path)
    model = DSCMS().to(device)
    model.load_state_dict(state)
    n_params = model.param_count()
    if n_params != EXPECTED_PARAM_COUNT:
        raise RuntimeError("param count {} != {}".format(n_params, EXPECTED_PARAM_COUNT))

    t0 = time.time()
    train_store = get_store(project_root, data_root, group, factor, "train",
                            stats, device=device)
    val_store = get_store(project_root, data_root, group, factor, "val",
                          stats, device=device)
    load_seconds = time.time() - t0

    div32 = SpectralDivergence(g["Nx"], g["Ns"], g["Lx"], g["Ls"], ell_g,
                               dtype=torch.float32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_at(1, schedule),
                                 betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
    perm_rng = np.random.default_rng(1000 + pair_seed)   # shared sample order

    if device.startswith("cuda"):
        torch.backends.cudnn.benchmark = True

    curve_path = os.path.join(run_dir, "curve.jsonl")
    best_path = os.path.join(run_dir, "best.pt")
    last_path = os.path.join(run_dir, "last.pt")
    best_val = float("inf")
    best_epoch = None
    stop_reason = "max_epochs"
    failure = None
    epoch = 0
    val = float("nan")
    curve_f = open(curve_path, "w")
    deadline = time.time() + max_hours * 3600 if max_hours else None
    try:
        for epoch in range(1, max_epochs + 1):
            lr = lr_at(epoch, schedule)
            for pg in optimizer.param_groups:
                pg["lr"] = lr
            model.train()
            order = perm_rng.permutation(len(train_store))
            stats_ep = {"L_v": [], "div_raw": [], "div_w": [], "gnorm": []}
            t_ep = time.time()
            if device.startswith("cuda"):
                torch.cuda.reset_peak_memory_stats()
            for i in order:
                it = train_store.items[i]
                x = torch.from_numpy(np.ascontiguousarray(it["y32"]))[None].to(device)
                x = replicate_lr_to_hr(x, factor)
                q = torch.from_numpy(np.ascontiguousarray(it["q32"]))[None].to(device)
                dref = None
                if target == "T":
                    dref = torch.from_numpy(np.ascontiguousarray(it["dref32"]))[None].to(device)
                optimizer.zero_grad(set_to_none=True)
                out = model(x)
                total, L_v, div_raw, div_w, _dz = loss_terms(
                    out, q, dref, div32, lam, target)
                lv = float(L_v.detach())
                if not np.isfinite(lv):
                    failure = _write_failure(run_dir, "nonfinite_train_loss", epoch,
                                             {"plane": int(i), "L_v": lv}, config)
                    break
                total.backward()
                gn = _param_grad_norm(model)
                optimizer.step()
                stats_ep["L_v"].append(lv)
                stats_ep["gnorm"].append(gn)
                if div_raw is not None:
                    dv = float(div_raw.detach())
                    if not np.isfinite(dv):
                        failure = _write_failure(run_dir, "nonfinite_divergence_term",
                                                 epoch, {"plane": int(i), "div": dv}, config)
                        break
                    stats_ep["div_raw"].append(dv)
                    stats_ep["div_w"].append(dv * lam)
            if failure is not None:
                stop_reason = "technical_fail"
                break
            if deadline is not None and time.time() > deadline:
                stop_reason = "time_budget"
                break
            t_train = time.time() - t_ep
            t_val0 = time.time()
            val, ddiag = _validate(model, val_store, div32, g, device, dataset, U_g)
            t_val = time.time() - t_val0
            if not np.isfinite(val):
                failure = _write_failure(run_dir, "nonfinite_validation_metric", epoch,
                                         {"value": val}, config)
                stop_reason = "technical_fail"
                break
            is_best = val < best_val
            if is_best:
                if not _init_params_finite(model):
                    failure = _write_failure(run_dir, "nonfinite_parameters", epoch,
                                             {"best_candidate": val}, config)
                    stop_reason = "technical_fail"
                    break
                best_val, best_epoch = val, epoch
                torch.save({"epoch": epoch, "model_state": model.state_dict(),
                            "val_E_v": val, "seed": seed, "target": target,
                            "lam": float(lam), "group": group, "factor": factor,
                            "U_g": U_g, "ell_g": ell_g, "stage": stage},
                           best_path)
            peak = (float(torch.cuda.max_memory_allocated()) / 2**30
                    if device.startswith("cuda") else 0.0)
            rec = {
                "epoch": epoch, "lr": lr,
                "train_L_v": float(np.mean(stats_ep["L_v"])),
                "train_div_raw": (float(np.mean(stats_ep["div_raw"]))
                                  if stats_ep["div_raw"] else None),
                "train_div_weighted": (float(np.mean(stats_ep["div_w"]))
                                       if stats_ep["div_w"] else None),
                "grad_norm_mean": float(np.mean(stats_ep["gnorm"])),
                "grad_norm_max": float(np.max(stats_ep["gnorm"])),
                "val_E_v": val, "val_E_d": ddiag["E_d"],
                "val_A_d": ddiag["A_d"], "val_A_d_star": ddiag["A_d_star"],
                "best_val_E_v": best_val, "is_best": bool(is_best),
                "epoch_seconds": t_train + t_val,
                "train_seconds": t_train, "val_seconds": t_val,
                "peak_mem_gib": peak,
            }
            curve_f.write(json.dumps(rec) + "\n")
            curve_f.flush()
    finally:
        curve_f.close()

    if failure is None:
        torch.save({"epoch": epoch, "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "perm_rng_state": perm_rng.bit_generator.state,
                    "val_E_v": val, "seed": seed, "target": target,
                    "lam": float(lam), "group": group, "factor": factor,
                    "U_g": U_g, "ell_g": ell_g, "stage": stage,
                    "stop_reason": stop_reason}, last_path)
    summary = {
        "stage": stage, "group": group, "factor": factor, "condition": cond,
        "seed": seed, "target": target, "lam": float(lam),
        "dataset": dataset, "n_params": n_params, "U_g": U_g, "ell_g": ell_g,
        "init_sha256": init_sha, "pair_seed": pair_seed,
        "n_train_planes": len(train_store), "n_val_planes": len(val_store),
        "epochs_run": epoch, "best_val_E_v": best_val, "best_epoch": best_epoch,
        "stop_reason": stop_reason, "device": device,
        "load_seconds": load_seconds, "total_seconds": time.time() - t0,
        "out_dir": run_dir,
    }
    if failure is not None:
        summary["failure"] = failure["reason"]
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    return summary


def continue_run(project_root, data_root, group, factor, seed, target, lam,
                 to_epoch=600, lr=1e-4, device="cuda"):
    """Continue a finished 300-epoch run to `to_epoch` at fixed lr.

    Loads last.pt (model + optimizer + sample-order RNG state); writes
    continuation/curve.jsonl and continuation/last.pt inside the original run
    directory; the 300-epoch records are never modified.
    """
    stats = load_train_stats(project_root)
    gd = S.GROUPS[group]
    dataset = gd["dataset"]
    g = S.group_grid(group)
    U_g = stats["groups"][group]["U_g"]
    ell_g = stats["groups"][group]["ell_g"]
    cond = condition_of(group, factor)
    weight_dir = "shared" if target == "A" else "lam_" + format(float(lam), ".0e")
    base = os.path.join(project_root, ARTIFACTS, "runs", "formal",
                        cond, "s{}".format(seed), target, weight_dir)
    src = os.path.join(base, "last.pt")
    out_dir = os.path.join(base, "continuation")
    if os.path.exists(os.path.join(out_dir, "summary.json")):
        raise RuntimeError("continuation record already exists: {}".format(out_dir))
    os.makedirs(out_dir, exist_ok=True)
    ck = torch.load(src, map_location="cpu", weights_only=False)
    start_epoch = int(ck["epoch"])

    model = DSCMS().to(device)
    model.load_state_dict(ck["model_state"])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                 betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
    optimizer.load_state_dict(ck["optimizer_state"])
    perm_rng = np.random.default_rng(1)
    perm_rng.bit_generator.state = ck["perm_rng_state"]

    train_store = get_store(project_root, data_root, group, factor, "train",
                            stats, device=device)
    val_store = get_store(project_root, data_root, group, factor, "val",
                          stats, device=device)
    div32 = SpectralDivergence(g["Nx"], g["Ns"], g["Lx"], g["Ls"], ell_g,
                               dtype=torch.float32).to(device)
    config = {"type": "continuation", "from_epoch": start_epoch,
              "to_epoch": to_epoch, "lr": lr, "group": group, "factor": factor,
              "seed": seed, "target": target, "lam": float(lam),
              "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    with open(os.path.join(out_dir, "run_meta.json"), "w") as f:
        json.dump(config, f, indent=1)
    if device.startswith("cuda"):
        torch.backends.cudnn.benchmark = True

    curve_f = open(os.path.join(out_dir, "curve.jsonl"), "w")
    failure = None
    stop_reason = "max_epochs"
    val = float("nan")
    epoch = start_epoch
    try:
        for epoch in range(start_epoch + 1, to_epoch + 1):
            model.train()
            order = perm_rng.permutation(len(train_store))
            lv_list, dv_list = [], []
            for i in order:
                it = train_store.items[i]
                x = torch.from_numpy(np.ascontiguousarray(it["y32"]))[None].to(device)
                x = replicate_lr_to_hr(x, factor)
                q = torch.from_numpy(np.ascontiguousarray(it["q32"]))[None].to(device)
                dref = None
                if target == "T":
                    dref = torch.from_numpy(np.ascontiguousarray(it["dref32"]))[None].to(device)
                optimizer.zero_grad(set_to_none=True)
                out = model(x)
                total, L_v, div_raw, _, _ = loss_terms(out, q, dref, div32, lam, target)
                lv = float(L_v.detach())
                if not np.isfinite(lv):
                    failure = _write_failure(out_dir, "nonfinite_train_loss", epoch,
                                             {"plane": int(i)}, config)
                    break
                total.backward()
                optimizer.step()
                lv_list.append(lv)
                if div_raw is not None:
                    dv_list.append(float(div_raw.detach()))
            if failure is not None:
                stop_reason = "technical_fail"
                break
            val, ddiag = _validate(model, val_store, div32, g, device, dataset, U_g)
            if not np.isfinite(val):
                failure = _write_failure(out_dir, "nonfinite_validation_metric", epoch,
                                         {"value": val}, config)
                stop_reason = "technical_fail"
                break
            rec = {"epoch": epoch, "lr": lr,
                   "train_L_v": float(np.mean(lv_list)),
                   "train_div_raw": float(np.mean(dv_list)) if dv_list else None,
                   "val_E_v": val, "val_E_d": ddiag["E_d"],
                   "val_A_d": ddiag["A_d"], "val_A_d_star": ddiag["A_d_star"]}
            curve_f.write(json.dumps(rec) + "\n")
            curve_f.flush()
    finally:
        curve_f.close()
    if failure is None:
        torch.save({"epoch": epoch, "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "perm_rng_state": perm_rng.bit_generator.state,
                    "val_E_v": val, "group": group, "factor": factor,
                    "seed": seed, "target": target, "lam": float(lam),
                    "stage": "continuation"}, os.path.join(out_dir, "last.pt"))
    summary = {"group": group, "factor": factor, "seed": seed, "target": target,
               "lam": float(lam), "from_epoch": start_epoch, "to_epoch": to_epoch,
               "epochs_run": epoch, "stop_reason": stop_reason,
               "final_val_E_v": val, "out_dir": out_dir}
    if failure is not None:
        summary["failure"] = failure["reason"]
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    return summary
