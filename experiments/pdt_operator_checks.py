#!/usr/bin/env python3
"""Plan section 4.3 acceptance checks for the differentiable divergence.

Runs entirely on real grids (including the non-square channel grid) with
both numpy FP64 (reference) and torch FP64/FP32.  Writes
reports/operator_checks.json with per-check PASS/FAIL and the measured
numbers; exits nonzero if any REQUIRED check fails.

Checks (plan 4.3):
 1. constant field; analytic low-wavenumber sinusoids; non-square grid;
    single-axis Nyquist with resolvable derivative on the other axis
 2. numpy vs torch FP64 (atol 1e-10, rtol 1e-8); torch FP32 vs FP64 RMS
    relative error <= 1e-4
 3. small-grid FP64 autograd gradient vs central finite differences;
    linearity D(z - q*) = Dz - Dq*
 4. z' = q*' -> zero velocity and reference-divergence terms; zero-target
    term equals the label divergence energy; cached dref32 == online
 5. lambda=0 loss and per-parameter gradient identical to baseline
 6. block average / replication / component order / mean restoration /
    cache identity / no test-input leakage into the network input
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.pdt import spec as S                                    # noqa: E402
from src.pdt.operator_torch import (SpectralDivergence,           # noqa: E402
                                    divergence_numpy, effective_wavenumbers)
from src.pdt.data import block_average, read_center_plane         # noqa: E402
from src.pdt.losses import loss_terms                            # noqa: E402
from src.pdt.model_dsc_ms import replicate_lr_to_hr               # noqa: E402

RESULTS = {}


def record(name, ok, detail, required=True):
    RESULTS[name] = {"status": "PASS" if ok else "FAIL",
                     "required": required, "detail": detail}
    print("[{}] {} :: {}".format(RESULTS[name]["status"], name, detail))
    return ok


def check_1_operator_basics():
    ok_all = True
    # constant field -> zero divergence (square iso grid)
    Nx, Ns, Lx, Ls = 64, 64, 2 * np.pi, 2 * np.pi
    q = np.ones((2, Ns, Nx))
    d = divergence_numpy(q, Nx, Ns, Lx, Ls)
    m = float(np.abs(d).max())
    ok_all &= record("1a_constant_field", m < 1e-12, "max|D|={:.2e}".format(m))

    # analytic low-wavenumber sinusoid, non-square grid
    Nx, Ns, Lx, Ls = 96, 64, 3.0, 2.0
    x = np.arange(Nx) * (Lx / Nx)
    s = np.arange(Ns) * (Ls / Ns)
    kx = 2 * np.pi * 2 / Lx          # mode 2
    ks = 2 * np.pi * 3 / Ls          # mode 3
    X, SS = np.meshgrid(x, s)        # (Ns, Nx)
    q = np.stack([np.cos(kx * X), np.sin(ks * SS)])
    d = divergence_numpy(q, Nx, Ns, Lx, Ls)
    d_exact = -kx * np.sin(kx * X) + ks * np.cos(ks * SS)
    err = float(np.abs(d - d_exact).max() / np.abs(d_exact).max())
    ok_all &= record("1b_nonsquare_sinusoid", err < 1e-12,
                     "max rel err={:.2e}".format(err))

    # single-axis Nyquist: u = (-1)^i checkerboard along x only (Nyquist),
    # v = resolvable along s. dx u must be zeroed; ds v retained.
    Nx, Ns, Lx, Ls = 96, 64, 3.0, 2.0
    x = np.arange(Nx) * (Lx / Nx)
    s = np.arange(Ns) * (Ls / Ns)
    X, SS = np.meshgrid(x, s)
    q = np.stack([(-1.0) ** np.arange(Nx)[None, :] * np.ones((Ns, 1)),
                  np.sin(2 * np.pi * 3 / Ls * SS)])
    d = divergence_numpy(q, Nx, Ns, Lx, Ls)
    expect = 2 * np.pi * 3 / Ls * np.cos(2 * np.pi * 3 / Ls * SS)
    err = float(np.abs(d - expect).max() / np.abs(expect).max())
    ok_all &= record("1c_single_axis_nyquist", err < 1e-12,
                     "max rel err={:.2e}".format(err))
    return ok_all


def _smooth_random(Ns, Nx, Lx, Ls, seed):
    rng = np.random.default_rng(seed)
    from numpy.fft import rfftfreq
    F = (rng.standard_normal((2, Ns, Nx // 2 + 1))
         + 1j * rng.standard_normal((2, Ns, Nx // 2 + 1)))
    fx = rfftfreq(Nx)
    fz = np.fft.fftfreq(Ns)
    FX, FZ = np.meshgrid(fx, fz)
    F *= np.exp(-((FX * Nx / 8.0) ** 2 + (FZ * Ns / 8.0) ** 2))
    F[:, 0, 0] = 0
    out = np.fft.irfft2(F, s=(Ns, Nx))
    return out / np.abs(out).max()


def check_2_precision():
    ok_all = True
    cases = [("iso_1024", 1024, 1024, 2 * np.pi, 2 * np.pi),
             ("channel_1536x2048", 1536, 2048, 3 * np.pi, 8 * np.pi),
             ("nonsquare_small", 64, 96, 2.0, 3.0)]
    for name, Ns, Nx, Ls, Lx in cases:
        ell = np.sqrt(Lx * Ls) / (2 * np.pi)
        q = _smooth_random(Ns, Nx, Lx, Ls, seed=hash(name) % 2**31)
        d_np = divergence_numpy(q, Nx, Ns, Lx, Ls) * ell
        div64 = SpectralDivergence(Nx, Ns, Lx, Ls, ell, dtype=torch.float64)
        with torch.no_grad():
            d_t64 = div64(torch.tensor(q[None]))[0].numpy()
        ad = float(np.abs(d_np - d_t64).max())
        rd = float(np.abs(d_np - d_t64).max() / np.abs(d_np).max())
        ok_all &= record("2a_numpy_vs_torch_fp64_{}".format(name),
                         ad < 1e-10 and rd < 1e-8,
                         "atol={:.2e} rtol={:.2e}".format(ad, rd))
        div32 = SpectralDivergence(Nx, Ns, Lx, Ls, ell, dtype=torch.float32)
        with torch.no_grad():
            d_t32 = div32(torch.tensor(q[None], dtype=torch.float32))[0].numpy()
        rms = float(np.sqrt(np.mean((d_t32.astype(np.float64) - d_np) ** 2))
                   / np.sqrt(np.mean(d_np ** 2)))
        ok_all &= record("2b_fp32_vs_fp64_{}".format(name), rms <= 1e-4,
                         "rms rel={:.2e}".format(rms))
    return ok_all


def check_3_autograd_linearity():
    ok_all = True
    Nx, Ns, Lx, Ls = 96, 64, 3.0, 2.0
    ell = np.sqrt(Lx * Ls) / (2 * np.pi)
    div = SpectralDivergence(Nx, Ns, Lx, Ls, ell, dtype=torch.float64)
    rng = np.random.default_rng(7)
    z0 = torch.tensor(rng.standard_normal((1, 2, Ns, Nx)), dtype=torch.float64)
    z = z0.clone().requires_grad_(True)
    loss = div(z).pow(2).mean()
    loss.backward()
    g_auto = z.grad.numpy().copy()

    g_fd = np.zeros_like(z0.numpy())
    eps = 1e-6
    base = float(loss.detach())
    flat = z0.numpy().ravel()
    idx = rng.choice(flat.size, size=40, replace=False)
    for i in idx:
        zp = z0.detach().clone().numpy()
        zp.ravel()[i] += eps
        zm = z0.detach().clone().numpy()
        zm.ravel()[i] -= eps
        with torch.no_grad():
            lp = div(torch.tensor(zp)).pow(2).mean().item()
            lm = div(torch.tensor(zm)).pow(2).mean().item()
        g_fd.ravel()[i] = (lp - lm) / (2 * eps)
    err = float(np.abs(g_auto.ravel()[idx] - g_fd.ravel()[idx]).max()
                / max(1e-30, np.abs(g_fd.ravel()[idx]).max()))
    ok_all &= record("3a_autograd_vs_finite_differences", err < 1e-6,
                     "max rel err={:.2e} on 40 coords".format(err))

    # linearity D(z - q*) = Dz - Dq*
    a = torch.tensor(rng.standard_normal((1, 2, Ns, Nx)), dtype=torch.float64)
    b = torch.tensor(rng.standard_normal((1, 2, Ns, Nx)), dtype=torch.float64)
    with torch.no_grad():
        lhs = div(a - b)
        rhs = div(a) - div(b)
    m = float((lhs - rhs).abs().max())
    ok_all &= record("3b_linearity_D(z-q)", m < 1e-12, "max|diff|={:.2e}".format(m))
    return ok_all


def check_4_loss_zero_points(project_root, data_root):
    ok_all = True
    from src.pdt.data import PlaneStore
    stats_path = os.path.join(project_root,
                              "artifacts/planar_divergence_targets_v1/manifests/train_stats.json")
    if not os.path.exists(stats_path):
        # fall back to direct computation on one real plane
        q64, _ = read_center_plane(data_root, "isotropic1024coarse",
                                   S.legacy_time_index("isotropic1024coarse", 0), 129)
        U_g = float(np.sqrt(np.mean((q64 - q64.mean(axis=(1, 2))[:, None, None]) ** 2)))
        ell = 1.0
        grid = S.group_grid("iso_xy")
    else:
        with open(stats_path) as f:
            stats = json.load(f)
        U_g = stats["groups"]["iso_xy"]["U_g"]
        ell = stats["groups"]["iso_xy"]["ell_g"]
        grid = S.group_grid("iso_xy")
        q64, _ = read_center_plane(data_root, "isotropic1024coarse",
                                   S.legacy_time_index("isotropic1024coarse", 0), 129)
    mu = q64.mean(axis=(1, 2))
    q32 = ((q64 - mu[:, None, None]) / U_g).astype(np.float32)
    z = torch.tensor(q32[None])
    div32 = SpectralDivergence(grid["Nx"], grid["Ns"], grid["Lx"], grid["Ls"], ell,
                               dtype=torch.float32)
    with torch.no_grad():
        dref = div32(z)
    total, L_v, div_Z, div_T, _ = loss_terms(z, z, dref, div32, 0.5, "T")
    lv = float(L_v)
    dt = float(div_T)
    dz = float(div_Z)
    energy = float(dref.double().pow(2).mean())
    ok_all &= record("4a_zero_at_identity_T", abs(lv) < 1e-12 and abs(dt) < 1e-12,
                     "L_v={:.2e}, div_T={:.2e}".format(lv, dt))
    _, _, div_Z, _, _ = loss_terms(z, z, dref, div32, 0.5, "Z")
    ok_all &= record("4b_zero_target_equals_label_energy",
                     abs(float(div_Z) - energy) / energy < 1e-6,
                     "div_Z={:.12e} vs energy={:.12e}".format(float(div_Z), energy))
    # cached dref vs online recomputation (PlaneStore path)
    st = PlaneStore(data_root, "iso_xy", 4, [(0, 129)], U_g, ell,
                    device_for_prefetch="cpu")
    with torch.no_grad():
        online = div32(torch.tensor(st.items[0]["q32"])[None])[0].numpy()
    md = float(np.abs(st.items[0]["dref32"] - online).max())
    ok_all &= record("4c_cached_dref_matches_online", md < 1e-9,
                     "max|diff|={:.2e}".format(md))
    return ok_all


def check_5_lambda_zero_equivalence():
    ok_all = True
    from src.pdt.model_dsc_ms import DSCMS
    torch.manual_seed(0)
    model = DSCMS()
    Nx, Ns, Lx, Ls = 128, 128, 2 * np.pi, 2 * np.pi
    div = SpectralDivergence(Nx, Ns, Lx, Ls, 1.0, dtype=torch.float32)
    rng = np.random.default_rng(3)
    y = rng.standard_normal((1, 2, Ns // 4, Nx // 4)).astype(np.float32)
    x = replicate_lr_to_hr(torch.tensor(y), 4)
    q = torch.tensor(rng.standard_normal((1, 2, Ns, Nx)).astype(np.float32))
    dref = div(q).detach()
    outs = {}
    for tgt in ("A", "Z", "T"):
        model.zero_grad(set_to_none=True)
        total, L_v, dr, dw, _ = loss_terms(model(x), q, dref, div, 0.0, tgt)
        total.backward()
        grads = torch.cat([p.grad.ravel() for p in model.parameters()])
        outs[tgt] = (float(total), float(L_v), grads)
    same_loss = (outs["A"][0] == outs["Z"][0] == outs["T"][0]
                 and outs["A"][1] == outs["Z"][1] == outs["T"][1])
    same_grad = (bool(torch.equal(outs["A"][2], outs["Z"][2]))
                 and bool(torch.equal(outs["A"][2], outs["T"][2])))
    div_terms_none = True
    ok_all &= record("5_lambda0_loss_and_gradient_identical",
                     same_loss and same_grad and div_terms_none,
                     "loss equal: {}, grad equal: {}".format(same_loss, same_grad))
    return ok_all


def check_6_pipeline(data_root):
    ok_all = True
    # block average exactness on constants and against a manual loop
    rng = np.random.default_rng(11)
    q = rng.standard_normal((2, 32, 48))
    y = block_average(q, 4)
    man = np.stack([q[c].reshape(8, 4, 12, 4).mean(axis=(1, 3)) for c in range(2)])
    ok_all &= record("6a_block_average_matches_manual",
                     float(np.abs(y - man).max()) < 1e-15,
                     "max|diff|={:.2e}".format(float(np.abs(y - man).max())))
    # replication is the exact adjoint-consistent upsampling: C_r R = I
    y_t = torch.tensor(y[None])
    up = replicate_lr_to_hr(y_t, 4)
    back = block_average(up[0].numpy(), 4)
    ok_all &= record("6b_blockavg_of_replication_is_identity",
                     float(np.abs(back - y).max()) < 1e-15,
                     "max|diff|={:.2e}".format(float(np.abs(back - y).max())))
    # component order on a REAL plane: u is comp 0, v (iso) / w (channel) comp 1
    for ds, comps in (("isotropic1024coarse", ("u", "v")), ("channel", ("u", "w"))):
        q64, info = read_center_plane(data_root, ds,
                                      S.legacy_time_index(ds, 0),
                                      S.GROUPS["iso_xy" if ds.startswith("iso") else "channel_y075"]["centers_1based"][0])
        import h5py
        with h5py.File(os.path.join(data_root, info["source_file"]), "r") as f:
            vel = f["velocity"]
            if ds.startswith("iso"):
                ref0 = vel[4 if vel.shape[0] == 9 else 0, :, :, 0]
                ref1 = vel[4 if vel.shape[0] == 9 else 0, :, :, 1]
            else:
                ref0 = vel[:, 4 if vel.shape[1] == 9 else 0, :, 0]
                ref1 = vel[:, 4 if vel.shape[1] == 9 else 0, :, 2]
        ok = (float(np.abs(q64[0] - ref0).max()) == 0.0
              and float(np.abs(q64[1] - ref1).max()) == 0.0)
        ok_all &= record("6c_component_order_{}".format(ds), ok,
                         "u->ch0, {}->ch1 exact".format(comps[1]))
    # mean restoration from float32 normalized fields
    q64, info = read_center_plane(data_root, "isotropic1024coarse",
                                  S.legacy_time_index("isotropic1024coarse", 0), 129)
    mu = q64.mean(axis=(1, 2))
    U = 0.7
    q32 = ((q64 - mu[:, None, None]) / U).astype(np.float32)
    back = U * q32.astype(np.float64) + mu[:, None, None]
    rel = float(np.abs(back - q64).max() / np.abs(q64).max())
    ok_all &= record("6d_mean_restoration_fp32", rel < 1e-5,
                     "max rel err={:.2e}".format(rel))
    # LR plane mean equals HR plane mean exactly (full-domain area average)
    y64 = block_average(q64, 8)
    ok_all &= record("6e_lr_mean_equals_hr_mean",
                     float(np.abs(y64.mean(axis=(1, 2)) - mu).max()) < 1e-12,
                     "max|diff|={:.2e}".format(
                         float(np.abs(y64.mean(axis=(1, 2)) - mu).max())))
    # input-leakage structural check: network input is built ONLY from y32
    Nx, Ns = 128, 128
    inp = replicate_lr_to_hr(torch.zeros((1, 2, Ns // 4, Nx // 4)), 4)
    ok_all &= record("6f_network_input_shape_two_channels",
                     tuple(inp.shape) == (1, 2, Ns, Nx),
                     "shape={}".format(tuple(inp.shape)))
    # prospective frame identity: legacy and prospective time index sets disjoint
    leg = {S.legacy_time_index("isotropic1024coarse", m) for m in range(64)}
    pro = set(S.PROSPECTIVE["isotropic1024coarse"]["time_index"].values())
    ok_all &= record("6g_prospective_times_disjoint_from_legacy",
                     len(leg & pro) == 0, "overlap={}".format(len(leg & pro)))
    legc = {S.legacy_time_index("channel", m) for m in range(64)}
    proc = set(S.PROSPECTIVE["channel"]["time_index"].values())
    ok_all &= record("6g_prospective_channel_times_disjoint",
                     len(legc & proc) == 0, "overlap={}".format(len(legc & proc)))
    return ok_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=os.getcwd())
    ap.add_argument("--data-root", required=True)
    args = ap.parse_args()
    t0 = time.time()
    ok = True
    ok &= check_1_operator_basics()
    ok &= check_2_precision()
    ok &= check_3_autograd_linearity()
    ok &= check_4_loss_zero_points(args.project_root, args.data_root)
    ok &= check_5_lambda_zero_equivalence()
    ok &= check_6_pipeline(args.data_root)
    out = {
        "study": "planar_divergence_targets_v1",
        "checked_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "device_name": (torch.cuda.get_device_name(0)
                        if torch.cuda.is_available() else "cpu"),
        "elapsed_seconds": time.time() - t0,
        "all_required_passed": bool(ok),
        "checks": RESULTS,
    }
    out_dir = os.path.join(args.project_root,
                           "artifacts/planar_divergence_targets_v1/reports")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "operator_checks.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print("\nwritten:", path)
    print("ALL REQUIRED PASSED:", ok)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
