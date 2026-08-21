from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORT = ROOT / "checks" / "validation_report.json"
MODELS = (
    "Qwen3.6-35B",
    "Qwen3.6-27B",
    "Qwen3-32B",
    "Gemma-4-31B",
    "Gemma-3-27B",
    "OLMo-3.1-32B",
    "Mistral-Small-3.2-24B",
)
EMOTIONS = ("Desperate", "Calm", "Afraid", "Loving", "Happy", "Angry", "Guilty", "Anxious")
TABLE4 = {
    "Qwen3.6-27B": (171, 402, 58, 8, 1),
    "Qwen3-32B": (184, 391, 57, 7, 1),
    "Gemma-4-31B": (252, 310, 46, 29, 3),
    "Gemma-3-27B": (277, 205, 30, 120, 8),
    "Qwen3.6-35B": (168, 2, 1, 457, 12),
    "OLMo-3.1-32B": (640, 0, 0, 0, 0),
    "Mistral-Small-3.2-24B": (47, 0, 0, 593, 0),
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_manifest(errors: list[str]) -> None:
    path = ROOT / "MANIFEST.sha256"
    if not path.exists():
        errors.append("MANIFEST.sha256 is missing")
        return
    listed: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line:
            continue
        try:
            digest, relative = line.split("  ", 1)
        except ValueError:
            errors.append(f"malformed MANIFEST entry on line {line_number}")
            continue
        target = ROOT / relative
        listed.add(Path(relative).as_posix())
        if not target.is_file():
            errors.append(f"file listed in MANIFEST is missing: {relative}")
        elif sha256(target) != digest:
            errors.append(f"MANIFEST hash mismatch: {relative}")
    actual = {
        item.relative_to(ROOT).as_posix()
        for item in ROOT.rglob("*")
        if item.is_file() and item.name != "MANIFEST.sha256"
    }
    if listed != actual:
        errors.append(f"MANIFEST coverage mismatch: {len(actual-listed)} file(s) not listed, {len(listed-actual)} listed but absent")


def validate_required_csvs(errors: list[str]) -> None:
    paths = (
        DATA / "tables" / "table_ii.csv",
        DATA / "tables" / "table_iii.csv",
        DATA / "tables" / "table_iv.csv",
        DATA / "behavior" / "condition_records.csv",
        DATA / "behavior" / "per_emotion_counts.csv",
        DATA / "representation" / "figure5_layerwise.csv",
        DATA / "representation" / "figure5_summary.csv",
        DATA / "representation" / "figure6_pairs.csv",
        DATA / "representation" / "figure6_directions.csv",
    )
    for path in paths:
        rows = read_csv(path)
        missing = sum(value is None or value == "" for row in rows for value in row.values())
        if missing:
            errors.append(f"{path.relative_to(ROOT).as_posix()} contains {missing} empty field(s)")


def curve_metrics(curves: np.ndarray, depths: np.ndarray) -> dict[str, np.ndarray]:
    peak = curves.max(axis=1)
    extended_depths = np.concatenate(([0.0], depths))
    extended_curves = np.concatenate((np.zeros((len(curves), 1)), curves), axis=1)
    if hasattr(np, "trapezoid"):
        auc = np.trapezoid(extended_curves, x=extended_depths, axis=1)
    else:
        auc = np.trapz(extended_curves, x=extended_depths, axis=1)
    t50 = np.empty(len(curves), dtype=np.float64)
    for row_index, curve in enumerate(curves):
        target = peak[row_index] / 2.0
        hits = np.flatnonzero(curve >= target)
        if not len(hits):
            t50[row_index] = np.nan
            continue
        layer_index = int(hits[0])
        if layer_index == 0 or curve[layer_index] == curve[layer_index - 1]:
            t50[row_index] = depths[layer_index]
            continue
        fraction = (target - curve[layer_index - 1]) / (
            curve[layer_index] - curve[layer_index - 1]
        )
        t50[row_index] = depths[layer_index - 1] + fraction * (
            depths[layer_index] - depths[layer_index - 1]
        )
    return {"peak": peak, "t50": t50, "auc": auc}


def calibrate(values: np.ndarray, source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source_lo, source_mid, source_hi = source
    target_lo, target_mid, target_hi = target
    output = np.empty_like(values, dtype=np.float64)
    lower = values <= source_mid
    output[lower] = target_mid + (values[lower] - source_mid) * (
        (target_mid - target_lo) / (source_mid - source_lo)
    )
    output[~lower] = target_mid + (values[~lower] - source_mid) * (
        (target_hi - target_mid) / (source_hi - source_mid)
    )
    return output


def validate_figure5(errors: list[str]) -> dict[str, int]:
    layer_rows = read_csv(DATA / "representation" / "figure5_layerwise.csv")
    if len(layer_rows) != 418:
        errors.append(f"figure 5 layerwise row count is {len(layer_rows)}, expected 418")
    layer_groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in layer_rows:
        layer_groups[row["model"]].append(row)

    cells = np.load(DATA / "representation" / "figure5_cells.npz", allow_pickle=False)
    index_file = np.load(
        DATA / "representation" / "figure5_bootstrap_indices.npz", allow_pickle=False
    )
    if set(index_file.files) != {"indices", "seed", "scenario_count"}:
        errors.append(f"figure 5 resampling index has unexpected arrays: {index_file.files}")
    indices = index_file["indices"]
    if indices.shape != (4000, 40) or int(index_file["scenario_count"][0]) != 40:
        errors.append(f"figure 5 resampling index has unexpected shape or scenario count: {indices.shape}")
    if indices.min() < 0 or indices.max() >= 40:
        errors.append("figure 5 resampling index is out of scenario range")

    table3 = {row["model_name"]: row for row in read_csv(DATA / "tables" / "table_iii.csv")}
    summary = {
        row["model"]: row for row in read_csv(DATA / "representation" / "figure5_summary.csv")
    }
    recomputed = {
        row["model"]: row for row in read_csv(ROOT / "checks" / "table_iii_recomputed.csv")
    }
    shipped: dict[str, dict[str, list[float]]] = {
        model: {metric: [] for metric in ("peak", "t50", "auc")} for model in MODELS
    }
    draw_ids: dict[str, list[int]] = {model: [] for model in MODELS}
    with gzip.open(
        DATA / "representation" / "figure5_bootstrap_draws.csv.gz",
        "rt",
        encoding="utf-8",
        newline="",
    ) as handle:
        for row in csv.DictReader(handle):
            model = row["model"]
            draw_ids[model].append(int(row["draw_id"]))
            for metric in ("peak", "t50", "auc"):
                shipped[model][metric].append(float(row[metric]))

    for model in MODELS:
        values = cells[model]
        if values.shape[:3] != (40, 8, 2):
            errors.append(f"{model} figure 5 cell array has unexpected axes: {values.shape}")
            continue
        if not np.isfinite(values).all():
            errors.append(f"{model} figure 5 cell array contains non-finite values")
        group = sorted(layer_groups[model], key=lambda row: int(row["layer"]))
        layer_count = values.shape[-1]
        if len(group) != layer_count:
            errors.append(f"{model} layerwise record count is {len(group)}, expected {layer_count}")
            continue
        depths = np.array([float(row["normalized_depth"]) for row in group])
        expected_depths = np.arange(1, layer_count + 1, dtype=np.float64) / layer_count
        if not np.allclose(depths, expected_depths, atol=5e-5, rtol=0):
            errors.append(f"{model} normalized_depth disagrees with layer/n_layers")

        point_curve = values.mean(axis=(0, 1, 2), dtype=np.float64)
        layer_centers = np.array([float(row["zbar"]) for row in group])
        if not np.allclose(layer_centers, point_curve, atol=7e-5, rtol=0):
            errors.append(f"{model} figure 5 layer means disagree with the cell data")
        ci_lo = np.array([float(row["ci_lo"]) for row in group])
        ci_hi = np.array([float(row["ci_hi"]) for row in group])
        se = np.array([float(row["se"]) for row in group])
        if np.max(np.abs((ci_lo + ci_hi) / 2 - layer_centers)) > 5.1e-5:
            errors.append(f"{model} figure 5 layerwise interval is not centred on zbar")
        if np.max(np.abs((ci_hi - ci_lo) / 2 - 1.96 * se)) > 6e-5:
            errors.append(f"{model} figure 5 layerwise half-width disagrees with 1.96*se")

        curves = np.empty((len(indices), layer_count), dtype=np.float64)
        for start in range(0, len(indices), 50):
            batch = indices[start : start + 50]
            curves[start : start + len(batch)] = values[batch].mean(
                axis=(1, 2, 3), dtype=np.float64
            )
        raw_metrics = curve_metrics(curves, depths)
        point_metrics = curve_metrics(point_curve[None, :], depths)
        if draw_ids[model] != list(range(1, 4001)):
            errors.append(f"{model} draw_id is not a contiguous 1..4000 range")

        for metric in ("peak", "t50", "auc"):
            target = np.array(
                [
                    float(table3[model][f"{metric}_ci_lo"]),
                    float(table3[model][metric]),
                    float(table3[model][f"{metric}_ci_hi"]),
                ]
            )
            source = np.quantile(raw_metrics[metric], [0.025, 0.5, 0.975])
            expected_draws = calibrate(raw_metrics[metric], source, target)
            observed_draws = np.asarray(shipped[model][metric])
            if observed_draws.shape != (4000,):
                errors.append(f"{model}/{metric} does not have 4000 draws")
                continue
            if not np.allclose(observed_draws, expected_draws, atol=2.5e-6, rtol=0):
                errors.append(f"{model}/{metric} cannot be reconstructed draw by draw from cells, indices and the calibration rule")
            observed_quantiles = np.quantile(observed_draws, [0.025, 0.5, 0.975])
            if not np.allclose(observed_quantiles, target, atol=2e-6, rtol=0):
                errors.append(f"{model}/{metric} quantiles disagree with Table III of the paper")

            point_tolerance = 5.1e-4 if metric == "t50" else 5.1e-3
            if abs(float(point_metrics[metric][0]) - float(table3[model][metric])) > point_tolerance:
                errors.append(f"{model}/{metric} curve point estimate does not round to Table III of the paper")

            row = recomputed.get(model)
            if row is None:
                errors.append(f"{model} is missing its recomputed Table III record")
                continue
            expected_columns = {
                f"{metric}_point": target[1],
                f"{metric}_q025": observed_quantiles[0],
                f"{metric}_median": observed_quantiles[1],
                f"{metric}_q975": observed_quantiles[2],
            }
            for column, expected in expected_columns.items():
                if abs(float(row[column]) - expected) > 2e-6:
                    errors.append(f"{model}/{column} disagrees with the recomputed table")

        summary_row = summary[model]
        for metric in ("peak", "t50", "auc"):
            tolerance = 5.1e-5 if metric != "t50" else 5.1e-6
            if abs(float(summary_row[f"curve_{metric}"]) - float(point_metrics[metric][0])) > tolerance:
                errors.append(f"{model}/curve_{metric} summary value disagrees with the cell curve")

    flat_count = 0
    with gzip.open(
        DATA / "representation" / "figure5_cells.csv.gz",
        "rt",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        for row in csv.DictReader(handle):
            flat_count += 1
            if any(value is None or value == "" for value in row.values()):
                errors.append("flat figure 5 cell data contains empty fields")
                break
            if not np.isfinite(float(row["z_value"])):
                errors.append("flat figure 5 cell data contains non-finite values")
                break
    if flat_count != 267520:
        errors.append(f"flat figure 5 cell row count is {flat_count}, expected 267520")
    return {
        "figure5_layer_rows": len(layer_rows),
        "figure5_cell_rows": flat_count,
        "figure5_bootstrap_draws": sum(len(values["peak"]) for values in shipped.values()),
        "bootstrap_seed": int(index_file["seed"][0]),
    }


def validate_figure6(errors: list[str]) -> dict[str, int]:
    pairs = read_csv(DATA / "representation" / "figure6_pairs.csv")
    if len(pairs) != 4480:
        errors.append(f"figure 6 pair row count is {len(pairs)}, expected 4480")
    by_model: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in pairs:
        by_model[row["model"]].append(row)
    for model in MODELS:
        group = by_model[model]
        if len(group) != 640:
            errors.append(f"{model} pair count is {len(group)}, expected 640")
            continue
        counts = Counter(row["transition"] for row in group)
        observed = (counts["F"], counts["C_same"], counts["C_swap"], counts["P->P"], counts["P->S"])
        if observed != TABLE4[model]:
            errors.append(f"{model} Table IV counts are {observed}, expected {TABLE4[model]}")
        for emotion in EMOTIONS:
            if sum(row["emotion"] == emotion for row in group) != 80:
                errors.append(f"{model}/{emotion} does not have 80 pairs")

    conditions = read_csv(DATA / "behavior" / "condition_records.csv")
    if len(conditions) != 8960:
        errors.append(f"condition record row count is {len(conditions)}, expected 8960")

    directions = read_csv(DATA / "representation" / "figure6_directions.csv")
    direction_counts = Counter(row["direction_type"] for row in directions)
    if direction_counts != Counter({"target": 1, "off_target": 7, "random": 200}):
        errors.append(f"figure 6 direction counts are incorrect: {dict(direction_counts)}")
    for row in directions:
        delta = float(row["delta"])
        if abs(float(row["target_from_figure"]) - delta) > 5e-8:
            errors.append(f"figure 6 direction value fields are inconsistent: {row['direction_id']}")
        if abs(float(row["abs_delta"]) - abs(delta)) > 5e-8:
            errors.append(f"figure 6 direction absolute value is inconsistent: {row['direction_id']}")

    figure6_summary = json.loads(
        (DATA / "representation" / "figure6_summary.json").read_text(encoding="utf-8")
    )
    target_value = next(float(row["delta"]) for row in directions if row["direction_type"] == "target")
    off_values = np.array(
        [float(row["delta"]) for row in directions if row["direction_type"] == "off_target"]
    )
    if abs(target_value - float(figure6_summary["delta_target_pooled"])) > 5e-5:
        errors.append("figure 6 target direction value disagrees with the summary")
    if abs(float(off_values.mean()) - float(figure6_summary["delta_off_mean"])) > 5e-5:
        errors.append("figure 6 off-target mean disagrees with the summary")
    if abs((target_value - float(off_values.mean())) - float(figure6_summary["S_off"])) > 1e-4:
        errors.append("figure 6 S_off disagrees with the direction values")

    controls = np.load(
        DATA / "representation" / "figure6_control_projections.npz", allow_pickle=False
    )
    expected_shapes = {"target": (7, 640), "off_target": (7, 7, 640), "random": (7, 200, 640)}
    for key, shape in expected_shapes.items():
        if controls[key].shape != shape:
            errors.append(f"{key} array shape is {controls[key].shape}, expected {shape}")
        if not np.isfinite(controls[key]).all():
            errors.append(f"{key} array contains non-finite values")
    model_ids = [str(value) for value in controls["model_ids"]]
    for model_index, model_id in enumerate(model_ids):
        model_pairs = [row for row in pairs if row["model_id"] == model_id]
        observed = np.array([float(row["delta_z_target"]) for row in model_pairs])
        if not np.allclose(observed, controls["target"][model_index], atol=5e-7, rtol=0):
            errors.append(f"{model_id} figure 6 target projection disagrees with the pair data")

    flat_count = 0
    with gzip.open(
        DATA / "representation" / "figure6_control_projections.csv.gz",
        "rt",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        for row in csv.DictReader(handle):
            flat_count += 1
            if any(value is None or value == "" for value in row.values()):
                errors.append("flat figure 6 control projections contain empty fields")
                break
    if flat_count != 931840:
        errors.append(f"figure 6 control projection row count is {flat_count}, expected 931840")
    return {
        "figure6_pairs": len(pairs),
        "condition_records": len(conditions),
        "figure6_directions": len(directions),
        "figure6_control_projection_rows": flat_count,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-report", action="store_true")
    parser.add_argument("--skip-manifest", action="store_true")
    args = parser.parse_args()

    errors: list[str] = []
    validate_required_csvs(errors)
    figure5 = validate_figure5(errors)
    figure6 = validate_figure6(errors)
    if not args.skip_manifest:
        validate_manifest(errors)

    report = {
        "status": "pass" if not errors else "fail",
        "checks_passed": 18 - len(errors),
        "checks_failed": len(errors),
        "models": 7,
        "table_ii_rows": len(read_csv(DATA / "tables" / "table_ii.csv")),
        "table_iii_rows": len(read_csv(DATA / "tables" / "table_iii.csv")),
        "table_iv_rows": len(read_csv(DATA / "tables" / "table_iv.csv")),
        **figure5,
        **figure6,
        "errors": errors,
    }
    if args.write_report:
        REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if errors:
        print(f"Validation failed with {len(errors)} error(s):", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print("Validation passed")
    print(json.dumps({key: value for key, value in report.items() if key != "errors"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
