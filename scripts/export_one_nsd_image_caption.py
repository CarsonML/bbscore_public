#!/usr/bin/env python3
"""
Save one NSD test image + its caption from a JSONL (for figures / reports).

Caption JSONL format: {"index": int, "caption": str} per line (batch_captions_nsd.py).

Examples:
  cd ~/cs375/bbscore_public
  python scripts/export_one_nsd_image_caption.py \\
    --captions captions/nsd_qwen3vl/captions_Describe_this_image.jsonl \\
    --out-dir ./report_assets

  # NSD image index 42 (must exist as a line in the JSONL)
  python scripts/export_one_nsd_image_caption.py -c "$NSD_CAPTIONS_PATH" --nsd-index 42

  # First line of the JSONL (default)
  python scripts/export_one_nsd_image_caption.py -c captions/nsd_qwen3vl/captions_Describe_every_part.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Repo root = parent of scripts/
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--captions",
        "-c",
        default=None,
        help="Path to captions JSONL (default: $NSD_CAPTIONS_PATH)",
    )
    parser.add_argument(
        "--nsd-index",
        type=int,
        default=None,
        help="NSD test image index (must appear in JSONL). Default: first record in file.",
    )
    parser.add_argument(
        "--out-dir",
        "-o",
        type=Path,
        default=Path("report_assets"),
        help="Output directory (default: ./report_assets)",
    )
    parser.add_argument(
        "--basename",
        default="nsd_example",
        help="Base filename without extension (default: nsd_example)",
    )
    args = parser.parse_args()

    cap_raw = args.captions or os.environ.get("NSD_CAPTIONS_PATH")
    if not cap_raw:
        print(
            "ERROR: Pass --captions /path/to.jsonl or set NSD_CAPTIONS_PATH.",
            file=sys.stderr,
        )
        return 1

    cap_path = Path(cap_raw).expanduser().resolve()
    if not cap_path.is_file():
        print(f"ERROR: captions file not found: {cap_path}", file=sys.stderr)
        return 1

    idx_want = args.nsd_index
    caption = None
    idx_found = None
    with cap_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(rec, dict) or "index" not in rec or "caption" not in rec:
                continue
            i = rec["index"]
            if idx_want is None:
                idx_found, caption = i, rec["caption"]
                break
            if i == idx_want:
                idx_found, caption = i, rec["caption"]
                break

    if caption is None or idx_found is None:
        print(
            "ERROR: No matching caption "
            + (f"for nsd-index={idx_want}" if idx_want is not None else "in file"),
            file=sys.stderr,
        )
        return 1

    from data.NSDShared import NSDStimulusSet  # noqa: WPS433 — after sys.path

    ds = NSDStimulusSet()
    ds._prepare_images()
    if idx_found < 0 or idx_found >= len(ds.test_image_data):
        print(f"ERROR: NSD index {idx_found} out of range [0, {len(ds.test_image_data)})", file=sys.stderr)
        return 1

    raw = ds.test_image_data[idx_found]
    from PIL import Image
    import numpy as np

    if isinstance(raw, np.ndarray):
        pil = Image.fromarray(raw)
    else:
        pil = raw

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.basename}_idx{idx_found}"
    img_path = args.out_dir / f"{stem}.png"
    txt_path = args.out_dir / f"{stem}_caption.txt"

    pil.save(img_path)
    txt_path.write_text(caption.strip() + "\n", encoding="utf-8")

    print(f"NSD index: {idx_found}")
    print(f"Image:     {img_path}")
    print(f"Caption:   {txt_path}")
    print("--- caption preview ---")
    preview = caption.strip().replace("\n", " ")
    if len(preview) > 400:
        preview = preview[:400] + "…"
    print(preview)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
