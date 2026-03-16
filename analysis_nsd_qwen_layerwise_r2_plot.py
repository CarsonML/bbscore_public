import argparse
import os
import re
import pickle
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FILENAME_REGEX = re.compile(
    r"^qwen3_vl_8b_(?P<modality>img|txt)_model\.language_model\.layers\."
    r"(?P<layer>\d+)_(?P<brain_area>[A-Za-z0-9]+(?:Caption)?Shared)\.pkl$"
)


def parse_filename(fname: str) -> Tuple[str, int, str]:
    """
    Parse modality, layer index, and brain area name from a result filename.

    Expected patterns include, for example:
    - qwen3_vl_8b_img_model.language_model.layers.12_NSDHighLateralShared.pkl
    - qwen3_vl_8b_txt_model.language_model.layers.12_NSDHighLateralCaptionShared.pkl
    """
    m = FILENAME_REGEX.match(fname)
    if m is None:
        raise ValueError(f"Unrecognized result filename pattern: {fname}")

    modality_raw = m.group("modality")
    modality = "image" if modality_raw == "img" else "caption"
    layer = int(m.group("layer"))
    brain_token = m.group("brain_area")

    # Normalize brain area label:
    # strip 'NSD' prefix and 'Shared' / 'CaptionShared' suffixes.
    area = brain_token
    if area.startswith("NSD"):
        area = area[3:]
    if area.endswith("CaptionShared"):
        area = area[: -len("CaptionShared")]
    elif area.endswith("Shared"):
        area = area[: -len("Shared")]

    return modality, layer, area


def extract_final_r2(obj) -> float:
    """
    Extract a scalar final_r2 from a loaded result object.

    For these NSD/Qwen results the interesting value lives under the
    'metrics' dict as a scalar 'final_r2'. In some cases this is nested
    under a metric name such as 'torch_ridge'.
    """
    if not isinstance(obj, dict):
        raise KeyError("Result object is not a dict; cannot extract 'final_r2'.")

    metrics = obj.get("metrics")
    if not isinstance(metrics, dict):
        raise KeyError("Result object has no 'metrics' dict; cannot extract 'final_r2'.")

    # 1) Direct key at metrics['final_r2'] (if present).
    if "final_r2" in metrics:
        return float(metrics["final_r2"])

    # 2) Nested under a specific metric (e.g., metrics['torch_ridge']['final_r2']).
    for value in metrics.values():
        if isinstance(value, dict) and "final_r2" in value:
            return float(value["final_r2"])

    raise KeyError("Could not find a scalar 'final_r2' in result object's 'metrics'.")


def collect_results(results_dir: str) -> pd.DataFrame:
    """
    Scan results_dir for Qwen NSD joint .pkl files and build a tidy DataFrame
    with columns: brain_area, layer, modality, r2.
    """
    records: List[Dict] = []

    if not os.path.isdir(results_dir):
        raise FileNotFoundError(f"Results directory does not exist: {results_dir}")

    for fname in os.listdir(results_dir):
        if not fname.endswith(".pkl"):
            continue
        try:
            modality, layer, brain_area = parse_filename(fname)
        except ValueError:
            # Skip files that do not match the expected Qwen NSD pattern.
            continue

        fpath = os.path.join(results_dir, fname)
        with open(fpath, "rb") as f:
            obj = pickle.load(f)

        try:
            r2 = extract_final_r2(obj)
        except KeyError:
            # If structure is unexpected, skip but log a message.
            print(f"Warning: skipping {fname} (no scalar final_r2 found).")
            continue

        records.append(
            {
                "brain_area": brain_area,
                "layer": layer,
                "modality": modality,
                "r2": r2,
            }
        )

    if not records:
        raise RuntimeError(
            f"No valid Qwen NSD result .pkl files found in directory: {results_dir}"
        )

    df = pd.DataFrame.from_records(records)
    df = df.sort_values(["brain_area", "modality", "layer"]).reset_index(drop=True)
    return df


def compute_global_ylim(df: pd.DataFrame, margin_ratio: float = 0.05) -> Tuple[float, float]:
    """
    Compute a global y-limit (shared y-axis) across all brain areas and modalities.
    """
    r2_values = df["r2"].values
    r2_min = float(np.min(r2_values))
    r2_max = float(np.max(r2_values))
    if r2_min == r2_max:
        # Degenerate case: expand a bit around the single value
        delta = max(abs(r2_min), 1.0) * margin_ratio
        return r2_min - delta, r2_max + delta

    r2_range = r2_max - r2_min
    pad = r2_range * margin_ratio
    return r2_min - pad, r2_max + pad


def plot_per_brain_area(
    df: pd.DataFrame,
    output_path: str,
) -> None:
    """
    Create one subplot per brain area, with shared axes so values are comparable.

    - x-axis: layer index
    - y-axis: final r^2 (shared across subplots)
    - Two lines per subplot: image vs caption.
    """
    brain_areas = sorted(df["brain_area"].unique())
    n_areas = len(brain_areas)

    # Determine subplot grid (roughly square).
    n_cols = min(4, n_areas)
    n_rows = int(np.ceil(n_areas / n_cols))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4 * n_cols, 3 * n_rows),
        sharex=True,
        sharey=True,
    )

    # axes could be a single Axes if n_areas == 1
    if isinstance(axes, plt.Axes):
        axes = np.array([[axes]])
    elif axes.ndim == 1:
        axes = axes.reshape(1, -1)

    ylim = compute_global_ylim(df)

    for idx, brain_area in enumerate(brain_areas):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        area_df = df[df["brain_area"] == brain_area]

        for modality, style, color, label in [
            ("image", "-", "purple", "Image"),
            ("caption", "--", "red", "Text"),
        ]:
            sub = area_df[area_df["modality"] == modality].sort_values("layer")
            if sub.empty:
                continue
            ax.plot(
                sub["layer"].values,
                sub["r2"].values,
                linestyle=style,
                color=color,
                marker="o",
                label=label,
            )

        ax.set_title(brain_area)
        ax.set_ylim(*ylim)

    # Hide any unused axes.
    for j in range(n_areas, n_rows * n_cols):
        row = j // n_cols
        col = j % n_cols
        fig.delaxes(axes[row, col])

    # Shared labels.
    fig.text(0.5, 0.04, "Layer index", ha="center")
    fig.text(0.04, 0.5, "Final $R^2$", va="center", rotation="vertical")
    fig.suptitle("NSD Qwen: Final $R^2$ Across Layers (Image vs Text)", y=0.98)

    # Create a single legend using the first valid axis.
    handles = []
    labels = []
    for ax in fig.axes:
        h, l = ax.get_legend_handles_labels()
        if h:
            handles, labels = h, l
            break
    if handles:
        fig.legend(handles, labels, loc="upper right")

    fig.tight_layout(rect=[0.06, 0.06, 0.9, 0.93])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=200)
    print(f"Saved figure to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot NSD Qwen joint final R^2 across layers and brain areas, "
            "comparing image vs text on shared axes."
        )
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="joint_test_results/results",
        help="Directory containing Qwen NSD joint .pkl result files.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="joint_test_results/nsd_qwen_layerwise_r2.png",
        help="Path to save the output figure.",
    )
    parser.add_argument(
        "--brain-areas",
        type=str,
        default=None,
        help=(
            "Optional comma-separated list of brain areas to include "
            "(after normalization, e.g., HighLateral,Ventral,Parietal). "
            "Defaults to all areas found."
        ),
    )

    args = parser.parse_args()

    df = collect_results(args.results_dir)

    if args.brain_areas:
        requested = {area.strip() for area in args.brain_areas.split(",") if area.strip()}
        df = df[df["brain_area"].isin(requested)]
        if df.empty:
            raise RuntimeError(
                f"No records match requested brain areas: {sorted(requested)}"
            )

    plot_per_brain_area(df, args.output_path)


if __name__ == "__main__":
    main()

