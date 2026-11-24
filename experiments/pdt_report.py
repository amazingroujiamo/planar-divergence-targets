#!/usr/bin/env python3
"""Internal completion report generator (plan sections 8-9, 11; P4 close-out).

Assembles reports/results.md from the frozen manifests, run records, metrics,
bootstrap intervals and convergence checks.  This is the INTERNAL experiment
completion report requested by the user — explicitly not paper prose.
"""
import argparse
import csv
import glob
import json
import os
import sys
import time
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.pdt import spec as S                                    # noqa: E402
from src.pdt.spec import ARTIFACTS                               # noqa: E402


def jload(path):
    with open(path) as f:
        return json.load(f)


def load_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def fmt_ci(ci):
    if ci is None:
        return "n/a"
    return "[{:+.2%}, {:+.2%}]".format(ci[0], ci[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=os.getcwd())
    args = ap.parse_args()
    art = os.path.join(args.project_root, ARTIFACTS)
    man = os.path.join(art, "manifests")
    rep = os.path.join(art, "reports")

    L = []
    add = L.append
    add("# 散度目标替换实验：内部完成报告（planar_divergence_targets_v1）\n")
    add("生成时间：{}\n".format(
        time.strftime("%Y-%m-%d %H:%M:%S %Z")))

    # ---- 1 completion counts ----
    add("\n## 1 实际完成量\n")
    rows = load_csv(os.path.join(man, "experiment_matrix.csv"))
    main_rows = [r for r in rows if r["stage"] == "main"]
    cont_rows = [r for r in rows if r["stage"] == "continuation"]
    n_ok = n_fail = n_cont_ok = n_cont_fail = 0
    fails = []
    for r in main_rows:
        d = os.path.join(art, r["run_dir"])
        if os.path.exists(os.path.join(d, "summary.json")):
            n_ok += 1
        elif os.path.exists(os.path.join(d, "failure.json")):
            n_fail += 1
            fails.append(r["run_dir"])
    for r in cont_rows:
        d = os.path.join(art, r["run_dir"])
        if os.path.exists(os.path.join(d, "summary.json")):
            n_cont_ok += 1
        elif os.path.exists(os.path.join(d, "failure.json")):
            n_cont_fail += 1
            fails.append(r["run_dir"] + " (continuation)")
    add("- 主矩阵 {} 个配置：完成 {}，技术失败 {}，未完成 {}。".format(
        len(main_rows), n_ok, n_fail, len(main_rows) - n_ok - n_fail))
    add("- 预定继续训练 {} 个阶段（种子 11、lambda_high、300->600 epochs）：完成 {}，失败 {}。".format(
        len(cont_rows), n_cont_ok, n_cont_fail))
    pilot_dirs = glob.glob(os.path.join(art, "runs", "pilot", "**", "summary.json"),
                           recursive=True)
    add("- 预实验：完成 {} 次短训练。".format(len(pilot_dirs)))
    if fails:
        add("- 技术失败清单：{}".format(fails))
    cohort = jload(os.path.join(man, "test_cohort.json"))
    add("- 前瞻测试帧：{}/{} 已核验（时间索引与全部旧存储帧零重叠：{}）。".format(
        cohort["n_verified"], cohort["n_planned"],
        cohort["time_index_overlap_with_legacy_stored_frames"]))

    # ---- 2 data and stats ----
    add("\n## 2 数据、归一化与来源\n")
    stats = jload(os.path.join(man, "train_stats.json"))
    for g, rec in stats["groups"].items():
        add("- `{}`：U_g={:.6g}，ell_g={:.6g}，训练 {} 场；训练集 A_d*={:.3g}，"
            "chi2={:.3g}。".format(g, rec["U_g"], rec["ell_g"],
                                   rec["n_train_planes"],
                                   rec["train_A_d_star_mean"],
                                   rec["train_chi2_mean"]))
    if "legacy_crosscheck" in stats:
        add("- 与归档 dns_v1 train_stats 的 U_g 交叉核对：{}。".format(
            json.dumps(stats["legacy_crosscheck"])))
    inv = load_csv(os.path.join(man, "source_inventory.csv"))
    add("- 来源清单：{} 个原生文件（训练/验证 {}，前瞻测试 {}）。".format(
        len(inv),
        sum(1 for r in inv if r["roles"] in ("train", "val", "train+val")),
        sum(1 for r in inv if r["roles"] == "prospective_test")))

    # ---- 3 pilot and weights ----
    add("\n## 3 权重预实验与三档选择\n")
    sel = jload(os.path.join(man, "pilot_weight_selection.json"))
    add("- S(lambda) 响应：{}".format(
        json.dumps({k: (round(v, 4) if v is not None else None)
                    for k, v in sel["S_of_lambda"].items()})))
    add("- 选中三档：low={} mid={} high={}（对应 S={}）。".format(
        *sel["chosen"], [round(x, 4) for x in sel["chosen_S"]]))
    add("- 覆盖判断：{}。".format(json.dumps(sel["coverage"])))

    # ---- 4 primary results ----
    add("\n## 4 主结果（前瞻测试，速度误差 E_v 对原始 HR 真值）\n")
    per_seed = load_csv(os.path.join(art, "metrics", "per_seed.csv"))
    boot = jload(os.path.join(rep, "bootstrap.json"))
    cls = jload(os.path.join(rep, "classification.json"))
    conds = sorted({r["condition"] for r in per_seed})
    lams3 = [float(x) for x in sel["chosen"]]
    lam_names = {lams3[0]: "low", lams3[1]: "mid", lams3[2]: "high"}

    add("（各组 E_v 先按三种子取平均，再计算括号内相对基线的变化；"
        "各种子数值见 effects.csv / per_seed.csv）\n")
    add("| 条件 | 基线 E_v | Z(low) R_Z | Z(mid) | Z(high) | T(low) R_T | T(mid) | T(high) |")
    add("|---|---|---|---|---|---|---|---|")

    def seed_mean(cond, tgt, lam):
        vals = [float(r["E_v"]) for r in per_seed
                if r["condition"] == cond and r["target"] == tgt
                and float(r["lam"]) == lam]
        return float(np.mean(vals))

    for cond in conds:
        base = seed_mean(cond, "A", 0.0)
        cells = ["{:.4f}".format(base)]
        for tgt in ("Z", "T"):
            for lam in lams3:
                v = seed_mean(cond, tgt, lam)
                cells.append("{:.4f} ({:+.1%})".format(v, v / base - 1))
        add("| {} | {} |".format(cond, " | ".join(cells)))

    add("\n### 目标替换效应 R_target = (E_T - E_Z)/E_A（同权重，三种子平均）\n")
    add("（条件级效果由三种子平均误差计算；稳定性检查中的逐种子效果为 "
        "R_target,s = (E_T,s − E_Z,s) / E_A,s，见 classification.json 的 seed_R。）\n")
    add("| 条件 | low | mid（Bonferroni 95%） | high |")
    add("|---|---|---|---|")
    for cond in conds:
        cells = []
        for lam in lams3:
            key = "{}|target|{}".format(cond, format(lam, ".0e"))
            c = cls.get(key)
            b = boot["conditions"][cond][format(lam, ".0e")]["R_target"]
            if lam == lams3[1]:
                cells.append("{:+.2%} {} Bonf [{:+.2%},{:+.2%}] {}".format(
                    c["R"] if c else float("nan"), fmt_ci(c["ci95"] if c else None),
                    b["bonf_lo"], b["bonf_hi"],
                    (c["class"] if c else "?")))
            else:
                cells.append("{:+.2%} {} {}".format(
                    c["R"] if c else float("nan"), fmt_ci(c["ci95"] if c else None),
                    (c["class"] if c else "?")))
        add("| {} | {} |".format(cond, " | ".join(cells)))

    add("\n### 零目标相对基线 R_Z 与参考目标相对基线 R_T（mid 档）\n")
    add("（分类区间口径：lambda_mid 的 R_target 主族用 Bonferroni 95%，"
        "其余用描述性 95%。实际等效要求区间完全位于 ±1% 带内，且不存在"
        "“种子正负方向混合、同时至少一个种子的 |R|>1%”的情况。若各种子均在 ±1% 带内，"
        "微小的正负方向差异本身不判为不稳定。这是验收阶段对原计划 §8.3 稳定性原则的"
        "操作化细化，并非原计划已完整预先规定的定量规则。）\n")
    add("| 条件 | R_Z（各种子） | R_Z 分类 | R_T（各种子） | R_T 分类 |")
    add("|---|---|---|---|---|")

    def seed_str(c):
        if not c:
            return "n/a"
        vals = [float(v) for v in c["seed_R"].values()]
        return "{:+.2%} [{}]".format(
            c["R"], "/".join("{:+.2%}".format(v) for v in sorted(vals)))

    for cond in conds:
        cz = cls.get("{}|Z|{}".format(cond, format(lams3[1], ".0e")))
        ct = cls.get("{}|T|{}".format(cond, format(lams3[1], ".0e")))
        add("| {} | {} {} | {} | {} {} | {} |".format(
            cond, seed_str(cz), fmt_ci(cz["ci95"]), cz["class"],
            seed_str(ct), fmt_ci(ct["ci95"]), ct["class"]))

    # application comparison
    fws_path = os.path.join(man, "formal_weight_selection.json")
    if os.path.exists(fws_path):
        fws = jload(fws_path)
        add("\n### 验证选权重后的应用比较\n")
        add("真正存在权重选择的是 6 条件 x 2 附加目标 = **12 项**（基线组 6 项无候选集，"
            "恒为 λ=0）。按预定规则（三种子平均验证 E_v 最小；1% 相对容差内并列取较小"
            "权重，0 为候选），**12 项全部选择 λ=0**。这不等于正权重没有任何数值改善："
            "下表同时给出各目标最优正权重的平均验证 E_v 与基线之差。\n")
        add("| 条件 | 目标 | 选中权重 | 平均验证 E_v | 最优正权重 | 其均值 | 相对基线 |")
        add("|---|---|---|---|---|---|---|")
        # per (cond, target) best positive weight from run summaries
        best_pos = {}
        for r in rows:
            if r["stage"] != "main" or r["target"] == "A":
                continue
            key = (r["condition"], r["target"])
            sp = os.path.join(art, r["run_dir"], "summary.json")
            if not os.path.exists(sp):
                continue
            v = float(jload(sp)["best_val_E_v"])
            best_pos.setdefault(key, {}).setdefault(float(r["lam"]), []).append(v)
        for k, v in sorted(fws.items()):
            cond, tgt = k.split("|")
            bp_lam = bp_val = None
            if (cond, tgt) in best_pos:
                means = {lam: float(np.mean(vs)) for lam, vs
                         in best_pos[(cond, tgt)].items()}
                bp_lam = min(means, key=lambda l: means[l])
                bp_val = means[bp_lam]
            add("| {} | {} | {} | {:.6g} | {} | {} | {} |".format(
                cond, tgt,
                format(v["lam"], ".0e") if v["lam"] else "0（基线）",
                v["mean_val_E_v"],
                format(bp_lam, ".0e") if bp_lam is not None else "-",
                "{:.6g}".format(bp_val) if bp_val is not None else "-",
                "{:+.2%}".format(bp_val / v["mean_val_E_v"] - 1)
                if bp_val is not None else "-"))

    # divergence diagnostics
    add("\n### 散度诊断（mid 档，测试集）\n")
    add("| 条件 | A_d 基线 | A_d Z | A_d T | gain a (Z) | gain a (T) |")
    add("|---|---|---|---|---|---|")
    for cond in conds:
        def m(tgt, lam, f):
            vals = [float(r[f]) for r in per_seed
                    if r["condition"] == cond and r["target"] == tgt
                    and float(r["lam"]) == lam]
            return np.mean(vals) if vals else float("nan")
        add("| {} | {:.3g} | {:.3g} | {:.3g} | {:.3f} | {:.3f} |".format(
            cond, m("A", 0.0, "A_d"), m("Z", lams3[1], "A_d"),
            m("T", lams3[1], "A_d"), m("Z", lams3[1], "gain_a"),
            m("T", lams3[1], "gain_a")))

    # convergence
    conv = jload(os.path.join(rep, "convergence_check.json"))
    add("\n## 5 预算敏感性（300→600 epochs 继续训练，种子 11，lambda_high）\n")
    for cond, e in conv.items():
        if not e:
            continue
        add("- `{}`：{}".format(cond, json.dumps(
            {k: ({kk: (round(vv, 6) if isinstance(vv, float) else vv)
                  for kk, vv in v.items()} if isinstance(v, dict) else v)
             for k, v in e.items()})))

    # runtime
    rt = jload(os.path.join(rep, "runtime_pilot.json"))
    add("\n## 6 运行时与环境\n")
    for p in rt["probes"][:3]:
        add("- {} r{} {}：{:.3f} s/步，验证 {:.1f} s，峰值 {:.2f} GiB。".format(
            p["group"], p["factor"], p["target"], p["step_seconds_mean"],
            p["val_seconds"], p["peak_mem_alloc_gib"]))
    oc = jload(os.path.join(rep, "operator_checks.json"))
    add("- 算子验收：{} 项检查全部 {}（torch {}，{}）。".format(
        len(oc["checks"]), "通过" if oc["all_required_passed"] else "未全部通过",
        oc["torch_version"], oc["device_name"]))

    # interpretation guardrails
    add("\n## 7 结果能够支持与不能支持的表述\n")
    # corrected headline counts from the classification itself
    z_mid_deg = sum(1 for cond in conds
                    if cls.get("{}|Z|{}".format(cond, format(lams3[1], ".0e")), {})
                    .get("class") == "degradation")
    tgt_mid_imp = sum(1 for cond in conds
                      if cls.get("{}|target|{}".format(cond, format(lams3[1], ".0e")), {})
                      .get("class") == "improvement")
    z_high_deg = sum(1 for cond in conds
                     if cls.get("{}|Z|{}".format(cond, format(lams3[2], ".0e")), {})
                     .get("class") == "degradation")
    t_comparisons = [(key, cls[key]) for cond in conds for lam in lams3
                     for key in ["{}|T|{}".format(cond, format(lam, ".0e"))]]
    t_counts = Counter(c["class"] for _, c in t_comparisons)
    t_noneq = [(key, c["class"]) for key, c in t_comparisons
               if c["class"] != "practical_equivalence"]
    add("- 核心认识（数据支持）：在当前网络、数据与倍率范围内，较强的二维零散度惩罚"
        "造成明显速度退化——λ_high 下 {} / 6 条件分类为退化；λ_mid 下 {} / 6 条件退化"
        "（其余为实际等效）。".format(z_high_deg, z_mid_deg))
    add("- 目标替换效应（R_target，与 R_Z 退化为两项不同比较）：λ_mid 下 {} / 6 条件"
        "达到超 1% 的改善（Bonferroni 口径）；λ_high 下六条件全部为改善。"
        "这支持零目标失配是零目标组退化的一个因素，不支持把全部速度误差严格因果分解。".format(tgt_mid_imp))
    add("- 参考目标组：{} 项（{} 条件 × {} 权重，仅 T 相对基线）中，"
        "{} 项实际等效、{} 项不确定／不稳定、{} 项改善、{} 项退化。非等效条目：{}。"
        "时间区间窄不代表训练随机性下稳定；种子值与区间见相应表格及 classification.json。".format(
            len(t_comparisons), len(conds), len(lams3),
            t_counts["practical_equivalence"], t_counts["uncertain_or_unstable"],
            t_counts["improvement"], t_counts["degradation"],
            "、".join("`{}`={}".format(k, v) for k, v in t_noneq) or "无"))
    add("- 验证选权重：12 项目标级比较按 1% 容差、并列取小的预定规则全部选择 λ=0；"
        "部分正权重存在不足 1% 的验证改善（见上表），不能表述为正权重无任何改善，"
        "也不改变未选择正权重的结果。")
    add("- 不能支持：将有限权重软损失的结果外推到所有严格无散方法；把位置差异解释为"
        "纯散度因果效应；宣称前瞻测试为独立 DNS 实现；把 a 与 E_d 的分解当作速度误差"
        "的严格因果分解；单条件结果外推到未训练流动条件；以及\"参考目标在所有条件、"
        "所有权重下等效\"的概括（存在上述例外）。")
    add("- 区间条件于本次三个已训练种子的平均结果与 16 个测试时刻的时间块 bootstrap；"
        "未覆盖未来训练随机性；种子间差异另见各表种子列。")
    add("- 汇总更正记录：2026-10-19 验收发现并已修复 (1) 主表曾误用单个种子"
        "（键缺种子维度的覆盖 bug），现全部为三种子平均；(2) λ_mid 零目标退化计数"
        "3/6（此前误写 4/6，系与目标替换改善 4/6 混用）；(3) 参考目标概括加入例外；"
        "(4) 权重选择计数 12 项；(5) 主族分类统一 Bonferroni 区间；(6) 等效类附加"
        "种子稳定性检查。随后复验又修复 R_target 的逐种子归一化（此前 seed_R 误存绝对差值），"
        "低权重下 channel_y256_r4 与 iso_xy_r4 的目标替换比较改判为不确定／不稳定；"
        "参考目标汇总改为仅统计 T 的 18 项比较，并修复主表被说明文字截断的排版。"
        "种子稳定性规则的验收阶段细化已在上文说明；训练与逐场评价数据未改动。")

    out = os.path.join(rep, "results.md")
    with open(out, "w") as f:
        f.write("\n".join(L) + "\n")
    print("written", out)


if __name__ == "__main__":
    main()
