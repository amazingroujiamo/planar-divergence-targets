"""用独立的小型数据检查散度目标分析及其最终报告。"""

import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = Path("artifacts/planar_divergence_targets_v1")
CONDITIONS = [
    "channel_y075_r4", "channel_y075_r8", "channel_y256_r4",
    "channel_y256_r8", "iso_xy_r4", "iso_xy_r8",
]
LAMBDAS = [1e-6, 1e-5, 1e-4]
BASELINES = {11: 0.1, 22: 0.2, 33: 0.4}
CONFLICT_CONDITIONS = {"channel_y256_r4", "iso_xy_r4"}
LOW_TARGET_EFFECTS = {11: 0.02, 22: 0.005, 33: -0.01}


def write_csv(path, fieldnames, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_fixture(root, scale=1.0):
    art = root / ARTIFACTS
    for name in ("metrics", "manifests", "reports"):
        (art / name).mkdir(parents=True)
    rows = []
    configurations = [("A", 0.0)] + [
        (target, lam) for target in ("Z", "T") for lam in LAMBDAS
    ]
    for cond in CONDITIONS:
        for seed, base in BASELINES.items():
            for target, lam in configurations:
                relative = 0.0
                if target == "Z":
                    relative = {1e-6: 0.0, 1e-5: 0.1, 1e-4: 0.5}[lam]
                elif target == "T":
                    relative = {1e-6: 0.001, 1e-5: 0.002, 1e-4: 0.003}[lam]
                    if lam == 1e-6 and cond in CONFLICT_CONDITIONS:
                        relative = LOW_TARGET_EFFECTS[seed]
                    elif lam == 1e-6 and cond == "channel_y075_r4":
                        relative = {11: 0.004, 22: -0.002, 33: 0.001}[seed]
                for m in range(4):
                    row = {"condition": cond, "seed": seed, "target": target,
                           "lam": lam, "m": m, "E_v": scale * base * (1 + relative)}
                    row.update({key: 1.0 for key in (
                        "E_d", "A_d", "A_d_star", "gain_a", "chi2",
                        "obs_consistency", "E_v_c0", "E_v_c1")})
                    rows.append(row)
    write_csv(art / "metrics/per_field_prospective.csv", list(rows[0]), rows)
    write_csv(art / "manifests/experiment_matrix.csv", ["stage", "run_dir"], [])
    write_csv(art / "manifests/source_inventory.csv", ["roles"], [])
    json_files = {
        "manifests/pilot_weight_selection.json": {
            "chosen": LAMBDAS, "S_of_lambda": {},
            "chosen_S": [0.025, 0.17, 0.495], "coverage": {},
        },
        "manifests/test_cohort.json": {
            "n_verified": 0, "n_planned": 0,
            "time_index_overlap_with_legacy_stored_frames": [],
        },
        "manifests/train_stats.json": {"groups": {}},
        "reports/runtime_pilot.json": {"probes": []},
        "reports/operator_checks.json": {
            "checks": [], "all_required_passed": True,
            "torch_version": "fixture", "device_name": "fixture",
        },
    }
    for rel, content in json_files.items():
        (art / rel).write_text(json.dumps(content))
    return art


def run_script(name, root):
    subprocess.run(
        [sys.executable, str(ROOT / "experiments" / name), "--project-root", str(root)],
        check=True, capture_output=True, text=True,
    )


class PDTAnalysisReportingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.work = tempfile.TemporaryDirectory(prefix="pdt-regression-")
        cls.root = Path(cls.work.name)
        cls.art = write_fixture(cls.root)
        run_script("pdt_analyze.py", cls.root)
        run_script("pdt_report.py", cls.root)
        cls.classification = json.loads((cls.art / "reports/classification.json").read_text())
        cls.bootstrap = json.loads((cls.art / "reports/bootstrap.json").read_text())
        cls.report = (cls.art / "reports/results.md").read_text()

    @classmethod
    def tearDownClass(cls):
        cls.work.cleanup()

    def test_target_seed_effect_uses_its_own_baseline(self):
        for cond in CONFLICT_CONDITIONS:
            entry = self.classification[f"{cond}|target|1e-06"]
            for seed, expected in LOW_TARGET_EFFECTS.items():
                with self.subTest(condition=cond, seed=seed):
                    self.assertAlmostEqual(entry["seed_R"][str(seed)], expected)
            self.assertTrue(entry["seed_conflict_beyond_band"])
            self.assertEqual(entry["class"], "uncertain_or_unstable")

    def test_classification_is_invariant_to_error_units(self):
        with tempfile.TemporaryDirectory(prefix="pdt-rescaled-") as work:
            art = write_fixture(Path(work), scale=100.0)
            run_script("pdt_analyze.py", Path(work))
            scaled = json.loads((art / "reports/classification.json").read_text())
        for key, before in self.classification.items():
            with self.subTest(comparison=key):
                after = scaled[key]
                self.assertEqual(before["class"], after["class"])
                for seed, expected in before["seed_R"].items():
                    self.assertAlmostEqual(after["seed_R"][seed], expected)

    def test_small_mixed_seed_directions_inside_band_remain_equivalent(self):
        for target in ("T", "target"):
            entry = self.classification[f"channel_y075_r4|{target}|1e-06"]
            self.assertFalse(entry["seed_conflict_beyond_band"])
            self.assertEqual(entry["class"], "practical_equivalence")

    def test_only_primary_target_family_uses_bonferroni_intervals(self):
        for key, entry in self.classification.items():
            cond, target, lam = key.split("|")
            primary = target == "target" and lam == "1e-05"
            interval_type = "bonferroni_95" if primary else "descriptive_95"
            source = self.bootstrap["conditions"][cond][lam]["R_" + target]
            expected = ([source["bonf_lo"], source["bonf_hi"]] if primary
                        else [source["desc_lo95"], source["desc_hi95"]])
            with self.subTest(comparison=key):
                self.assertEqual(entry["interval_type"], interval_type)
                self.assertEqual(entry["ci95"], expected)

    def test_reference_summary_contains_only_reference_comparisons(self):
        line = next(line for line in self.report.splitlines()
                    if line.startswith("- 参考目标组："))
        self.assertIn("18 项", line)
        self.assertIn("16 项实际等效", line)
        self.assertIn("2 项不确定／不稳定", line)
        self.assertNotIn("|Z|", line)

    def test_main_table_has_contiguous_rows_and_averages_all_seeds(self):
        lines = self.report.splitlines()
        header = next(i for i, line in enumerate(lines) if line.startswith("| 条件 | 基线 E_v |"))
        rows = lines[header + 2:header + 2 + len(CONDITIONS)]
        self.assertEqual(len(rows), len(CONDITIONS))
        for cond, row in zip(CONDITIONS, rows):
            with self.subTest(condition=cond):
                self.assertTrue(row.startswith(f"| {cond} | 0.2333 |"))

    def test_report_discloses_stability_rule_refinement(self):
        self.assertIn("验收阶段", self.report)
        self.assertIn("均在 ±1% 带内", self.report)


if __name__ == "__main__":
    unittest.main()
