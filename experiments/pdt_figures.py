#!/usr/bin/env python3
"""Figures for the internal report (plan section 10; NOT paper submission).

fig1: E_v vs lambda per condition (both targets + shared baseline), seeds shown
fig2: divergence action (A_d, E_d, gain a) vs speed change
fig3: one common reference field: truth, LR input, three predictions, errors
      (channel_y075 r4 seed 11 lambda_mid; test time chosen by the BASELINE
      per-time E_v median — display only, statistics untouched)
Outputs figures/*.png + figures/figure_data/*.csv.
"""
import argparse
import csv
import json
import os
import sys
from collections import defaultdict

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.pdt import spec as S                                    # noqa: E402
from src.pdt.data import read_center_plane, block_average        # noqa: E402
from src.pdt.model_dsc_ms import DSCMS, replicate_lr_to_hr       # noqa: E402
from src.pdt.train import ARTIFACTS, load_train_stats           # noqa: E402


def load_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=os.getcwd())
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    art = os.path.join(args.project_root, ARTIFACTS)
    figd = os.path.join(art, "figures")
    fdat = os.path.join(figd, "figure_data")
    os.makedirs(fdat, exist_ok=True)

    per_seed = load_csv(os.path.join(art, "metrics", "per_seed.csv"))
    with open(os.path.join(art, "manifests", "pilot_weight_selection.json")) as f:
        lams3 = [float(x) for x in json.load(f)["chosen"]]

    # ---------------- fig 1: E_v vs weight ----------------
    conds = sorted({r["condition"] for r in per_seed})
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    f1rows = []
    for ax, cond in zip(axes.ravel(), conds):
        rows = [r for r in per_seed if r["condition"] == cond]
        base = [r for r in rows if r["target"] == "A"]
        for r in base:
            f1rows.append({"condition": cond, "lam": 0, "target": "A",
                           "seed": r["seed"], "E_v": r["E_v"]})
            ax.scatter([0.9], [float(r["E_v"])], color="k", marker="x", zorder=5)
        base_mean = np.mean([float(r["E_v"]) for r in base])
        ax.axhline(base_mean, color="k", lw=0.8, ls="--", alpha=0.6)
        for tgt, color in (("Z", "tab:red"), ("T", "tab:blue")):
            lams = sorted({float(r["lam"]) for r in rows
                           if r["target"] == tgt})
            xs, ys, ylo, yhi = [], [], [], []
            for lam in lams:
                vals = [float(r["E_v"]) for r in rows
                        if r["target"] == tgt and float(r["lam"]) == lam]
                for v, s in zip(vals, [r["seed"] for r in rows
                                       if r["target"] == tgt
                                       and float(r["lam"]) == lam]):
                    f1rows.append({"condition": cond, "lam": lam,
                                   "target": tgt, "seed": s, "E_v": v})
                xs.append(lam)
                ys.append(np.mean(vals))
                ylo.append(np.min(vals))
                yhi.append(np.max(vals))
            ax.plot(xs, ys, "-o", color=color, label={"Z": "zero", "T": "ref"}[tgt])
            ax.fill_between(xs, ylo, yhi, color=color, alpha=0.15)
        ax.set_xscale("symlog", linthresh=1e-7)
        ax.set_title(cond)
        ax.set_xlabel(r"$\lambda$")
        ax.set_ylabel(r"$E_v$")
    axes[0, 0].legend()
    fig.suptitle("Speed error vs divergence weight (x marks the shared lambda=0 baseline)")
    fig.tight_layout()
    fig.savefig(os.path.join(figd, "fig1_Ev_vs_lambda.png"), dpi=160)
    plt.close(fig)
    with open(os.path.join(fdat, "fig1_data.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(f1rows[0].keys()))
        w.writeheader()
        w.writerows(f1rows)

    # ---------------- fig 2: divergence action vs speed change ------------
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    rows_out = []
    for cond in conds:
        rows = [r for r in per_seed if r["condition"] == cond]
        base_Ev = np.mean([float(r["E_v"]) for r in rows if r["target"] == "A"])
        base_Ad = np.mean([float(r["A_d"]) for r in rows if r["target"] == "A"])
        base_Ed = np.mean([float(r["E_d"]) for r in rows if r["target"] == "A"])
        for lam in sorted({float(r["lam"]) for r in rows} - {0.0}):
            for tgt in ("Z", "T"):
                sel = [r for r in rows if r["target"] == tgt
                       and float(r["lam"]) == lam]
                if not sel:
                    continue
                rec = {"condition": cond, "target": tgt, "lam": lam,
                       "R_Ev": np.mean([float(r["E_v"]) for r in sel]) / base_Ev - 1,
                       "A_d_ratio": np.mean([float(r["A_d"]) for r in sel]) / base_Ad,
                       "E_d_ratio": np.mean([float(r["E_d"]) for r in sel]) / base_Ed,
                       "gain_a": np.mean([float(r["gain_a"]) for r in sel])}
                rows_out.append(rec)
                axes[0].scatter(rec["A_d_ratio"], rec["R_Ev"] * 100,
                                12 if tgt == "Z" else 22, marker="o" if tgt == "Z" else "^",
                                alpha=0.7)
                axes[1].scatter(rec["E_d_ratio"], rec["R_Ev"] * 100,
                                12 if tgt == "Z" else 22, marker="o" if tgt == "Z" else "^",
                                alpha=0.7)
                axes[2].scatter(rec["gain_a"], rec["R_Ev"] * 100,
                                12 if tgt == "Z" else 22, marker="o" if tgt == "Z" else "^",
                                alpha=0.7)
    axes[0].set_xlabel(r"$A_d / A_{d,\mathrm{base}}$")
    axes[1].set_xlabel(r"$E_d / E_{d,\mathrm{base}}$")
    axes[2].set_xlabel(r"divergence-map gain $a$")
    for ax in axes:
        ax.set_ylabel(r"$R$ of $E_v$ (%)")
        ax.axhline(0, color="k", lw=0.6)
    axes[2].axvline(1.0, color="k", lw=0.6, ls=":")
    fig.suptitle("Divergence action vs speed change (o zero target, ^ reference target)")
    fig.tight_layout()
    fig.savefig(os.path.join(figd, "fig2_divergence_action.png"), dpi=160)
    plt.close(fig)
    with open(os.path.join(fdat, "fig2_data.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)

    # ---------------- fig 3: common field display -------------------------
    group, factor, seed = "channel_y075", 4, 11
    lam_mid = lams3[1]
    cond = "{}_r{}".format(group, factor)
    stats = load_train_stats(args.project_root)
    U_g = stats["groups"][group]["U_g"]
    per_time = load_csv(os.path.join(art, "metrics", "per_time_prospective.csv")
                        if os.path.exists(os.path.join(art, "metrics",
                                                       "per_time_prospective.csv"))
                        else os.path.join(art, "metrics", "per_time.csv"))
    btimes = [r for r in per_time if r["condition"] == cond
              and r["target"] == "A" and r["seed"] == str(seed)]
    if btimes:
        m_star = int(sorted(btimes, key=lambda r: abs(
            float(r["E_v"]) - np.median([float(x["E_v"]) for x in btimes])))[0]["m"])
    else:
        m_star = 48
    ti = S.PROSPECTIVE["channel"]["time_index"][m_star]
    q64, _ = read_center_plane(args.data_root, "channel", ti, 75, prospective=True)
    y64 = block_average(q64, factor)
    mu = y64.mean(axis=(1, 2))
    y32 = ((y64 - mu[:, None, None]) / U_g).astype(np.float32)
    x = torch.from_numpy(np.ascontiguousarray(y32))[None].to(args.device)
    x = replicate_lr_to_hr(x, factor)
    models = {"A": ("shared", "baseline"),
              "Z": ("lam_" + format(lam_mid, ".0e"), "zero target"),
              "T": ("lam_" + format(lam_mid, ".0e"), "reference target")}
    fig, axes = plt.subplots(2, 5, figsize=(21, 6))
    lr_disp = y64[0, ::8, ::8]
    axes[0, 0].imshow(lr_disp, origin="lower", cmap="viridis")
    axes[0, 0].set_title("LR input u (x8 stride)")
    axes[0, 1].imshow(q64[0], origin="lower", cmap="viridis")
    axes[0, 1].set_title("HR truth u")
    pc = None
    for col, (tgt, (wdir, label)) in enumerate(models.items()):
        p = os.path.join(art, "runs", "formal", cond, "s{}".format(seed),
                         tgt, wdir, "best.pt")
        if not os.path.exists(p):
            axes[0, 2 + col].set_title("{} (missing)".format(label))
            axes[0, 2 + col].axis("off")
            axes[1, 2 + col].axis("off")
            continue
        ck = torch.load(p, map_location="cpu", weights_only=False)
        model = DSCMS().to(args.device)
        model.load_state_dict(ck["model_state"])
        model.eval()
        with torch.no_grad():
            z32 = model(x)[0].float().cpu().numpy()
        z64 = U_g * z32.astype(np.float64) + mu[:, None, None]
        err = z64[0] - q64[0]
        axes[0, 2 + col].imshow(z64[0], origin="lower", cmap="viridis")
        axes[0, 2 + col].set_title("{} u".format(label))
        vmax = np.percentile(np.abs(err), 99.5)
        im = axes[1, 2 + col].imshow(err, origin="lower", cmap="RdBu_r",
                                     vmin=-vmax, vmax=vmax)
        plt.colorbar(im, ax=axes[1, 2 + col], shrink=0.8)
        axes[1, 2 + col].set_title("error u ({})".format(label))
    d_true = q64[0] * 0  # placeholder to keep layout; truth error panel below
    err_lr = np.repeat(np.repeat(y64[0], factor, 0), factor, 1) - q64[0]
    vmax = np.percentile(np.abs(err_lr), 99.5)
    im = axes[1, 0].imshow(err_lr, origin="lower", cmap="RdBu_r",
                           vmin=-vmax, vmax=vmax)
    plt.colorbar(im, ax=axes[1, 0], shrink=0.8)
    axes[1, 0].set_title("error u (LR replication)")
    axes[1, 1].axis("off")
    axes[1, 1].text(0.02, 0.5, "test m={} t_idx={}\nseed {} lambda_mid={:.0e}\n"
                    "display rule: baseline-median E_v time".format(
                        m_star, ti, seed, lam_mid), fontsize=9,
                    va="center", transform=axes[1, 1].transAxes)
    fig.tight_layout()
    fig.savefig(os.path.join(figd, "fig3_common_field.png"), dpi=160)
    plt.close(fig)
    print("figures written to", figd)


if __name__ == "__main__":
    main()
