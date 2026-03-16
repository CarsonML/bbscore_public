import argparse
import os
import re
import pickle
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Language-pathway (image input through language model) filenames, same as the
# existing layerwise script.
LANG_FILENAME_REGEX = re.compile(
    r"^qwen3_vl_8b_img_model\.language_model\.layers\."
    r"(?P<layer>\d+)_(?P<brain_area>[A-Za-z0-9]+Shared)\.pkl$"
)

# Visual-pathway filenames: image input through visual blocks.
VIS_FILENAME_REGEX = re.compile(
    r"^qwen3_vl_8b_img_model\.visual\.blocks\."
    r"(?P<layer>\d+)_(?P<brain_area>[A-Za-z0-9]+Shared)\.pkl$"
)


def _normalize_brain_area(token: str) -> str:
    """
    Normalize NSD brain area label:
    - strip 'NSD' prefix
    - strip 'Shared' suffix.
    """
    area = token
    if area.startswith("NSD"):
        area = area[3:]
    if area.endswith("Shared"):
        area = area[: -len("Shared")]
    return area


def _extract_final_r2(obj) -> float:
    """
    Extract a scalar final_r2 from a loaded result object.

    Mirrors logic from analysis_nsd_qwen_layerwise_r2_plot.py:
    look for metrics['final_r2'] or nested metrics[some_metric]['final_r2'].
    """
    if not isinstance(obj, dict):
        raise KeyError("Result object is not a dict; cannot extract 'final_r2'.")

    metrics = obj.get("metrics")
    if not isinstance(metrics, dict):
        raise KeyError("Result object has no 'metrics' dict; cannot extract 'final_r2'.")

    if "final_r2" in metrics:
        return float(metrics["final_r2"])

    for value in metrics.values():
        if isinstance(value, dict) and "final_r2" in value:
            return float(value["final_r2"])

    raise KeyError("Could not find a scalar 'final_r2' in result object's 'metrics'.")


def _collect_best_by_area(results_dir: str, regex: re.Pattern) -> pd.DataFrame:
    """
    Scan results_dir with the given filename regex and return a DataFrame:
    columns: brain_area, layer, r2
    filtered to rows with the best r2 per brain_area.
    """
    records: List[Dict] = []

    if not os.path.isdir(results_dir):
        raise FileNotFoundError(f"Results directory does not exist: {results_dir}")

    for fname in os.listdir(results_dir):
        if not fname.endswith(".pkl"):
            continue
        m = regex.match(fname)
        if m is None:
            continue

        layer = int(m.group("layer"))
        brain_token = m.group("brain_area")
        brain_area = _normalize_brain_area(brain_token)

        fpath = os.path.join(results_dir, fname)
        with open(fpath, "rb") as f:
            obj = pickle.load(f)

        try:
            r2 = _extract_final_r2(obj)
        except KeyError:
            print(f"Warning: skipping {fname} (no scalar final_r2 found).")
            continue

        records.append(
            {
                "brain_area": brain_area,
                "layer": layer,
                "r2": r2,
            }
        )

    if not records:
        raise RuntimeError(
            f"No valid result .pkl files found in directory: {results_dir}"
        )

    df = pd.DataFrame.from_records(records)
    # For each brain_area, keep only the row with max r2 (break ties by smallest layer index).
    df = df.sort_values(["brain_area", "r2", "layer"], ascending=[True, False, True])
    best = df.groupby("brain_area", as_index=False).first()
    return best


def plot_best_image_vs_language(
    visual_df: pd.DataFrame,
    language_df: pd.DataFrame,
    output_path: str,
) -> None:
    """
    Given per-area best-layer DataFrames for:
      - visual_df: image via visual pathway
      - language_df: image via language pathway

    create a bar chart with one group per brain area and two bars:
      - best visual-pathway R^2
      - best language-pathway R^2
    """
    # Align brain areas present in both.
    common_areas = sorted(
        set(visual_df["brain_area"].unique()).intersection(
            language_df["brain_area"].unique()
        )
    )
    if not common_areas:
        raise RuntimeError("No overlapping brain areas between visual and language sets.")

    visual_map = {
        row["brain_area"]: row["r2"] for _, row in visual_df.iterrows()
    }
    lang_map = {
        row["brain_area"]: row["r2"] for _, row in language_df.iterrows()
    }

    x = np.arange(len(common_areas))
    width = 0.35

    visual_vals = [visual_map[area] for area in common_areas]
    lang_vals = [lang_map[area] for area in common_areas]

    fig, ax = plt.subplots(figsize=(1.6 * len(common_areas), 4))

    # Image via visual pathway: green; image via language pathway: orange.
    ax.bar(
        x - width / 2,
        visual_vals,
        width,
        label="Image (visual path)",
        color="green",
    )
    ax.bar(
        x + width / 2,
        lang_vals,
        width,
        label="Image (language path)",
        color="orange",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(common_areas, rotation=45, ha="right")
    ax.set_ylabel("Best final $R^2$")
    ax.set_title("NSD Qwen: Best-layer $R^2$ (Image via Visual vs Language Pathways)")
    ax.legend()

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=200)
    print(f"Saved bar chart to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare best-layer NSD Qwen $R^2$ for image input via the visual "
            "pathway vs image input via the language pathway."
        )
    )
    parser.add_argument(
        "--visual-results-dir",
        type=str,
        required=True,
        help=(
            "Directory with Qwen visual-pathway results, e.g. "
            "qwen_visual_vs_language_subset_rp389376/results"
        ),
    )
    parser.add_argument(
        "--language-results-dir",
        type=str,
        required=True,
        help=(
            "Directory with Qwen language-pathway (image input) results, e.g. "
            "joint_test_results/results or another NSD Qwen sweep."
        ),
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="nsd_qwen_best_layer_image_vs_language.png",
        help="Path to save the output bar chart.",
    )

    args = parser.parse_args()

    visual_best = _collect_best_by_area(args.visual_results_dir, VIS_FILENAME_REGEX)
    language_best = _collect_best_by_area(args.language_results_dir, LANG_FILENAME_REGEX)

    plot_best_image_vs_language(visual_best, language_best, args.output_path)


if __name__ == "__main__":
    main()

