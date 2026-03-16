"""
Batch caption the full NSD stimulus set with Qwen3-VL.
Saves captions as JSONL (one {"index": int, "caption": str} per line).
Each line is written and synced immediately so if the job fails or is killed,
re-run with --resume to skip already-captioned indices and continue.
"""
import argparse
import json
import os
import re

import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from data.NSDShared import NSDStimulusSet

# ImageNet un-normalization (same as test_captions.py)
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"
DEFAULT_PROMPT = "Describe this image in one sentence."
MAX_NEW_TOKENS = 750


def unnormalize_to_pil(tensor):
    """Reverts PyTorch ImageNet normalization to viewable PIL."""
    tensor = tensor.cpu()
    tensor = tensor * STD + MEAN
    tensor = torch.clamp(tensor, 0, 1)
    array = (tensor.numpy() * 255).astype(np.uint8)
    array = np.transpose(array, (1, 2, 0))
    return Image.fromarray(array)


def prompt_to_slug(prompt: str, max_len: int = 20) -> str:
    """Short filesystem-safe slug from prompt (for default output filename)."""
    slug = re.sub(r"[^\w\s-]", "", prompt)[:max_len].strip().replace(" ", "_") or "caption"
    return slug or "caption"


def get_existing_indices(jsonl_path: str) -> set:
    """Read already-captioned indices from a JSONL file (for resume)."""
    if not os.path.isfile(jsonl_path):
        return set()
    done = set()
    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                done.add(rec["index"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def build_messages(prompt: str, pil_img: Image.Image) -> list[dict]:
    """Build one Qwen chat conversation for a single image."""
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_img},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def is_oom_error(exc: BaseException) -> bool:
    """Best-effort detection for CUDA memory failures."""
    if isinstance(exc, torch.OutOfMemoryError):
        return True
    return "out of memory" in str(exc).lower()


def main():
    parser = argparse.ArgumentParser(
        description="Batch caption the full NSD stimulus set with Qwen3-VL; save as JSONL (resumable)."
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=DEFAULT_PROMPT,
        help=f"Prompt for the model (default: '{DEFAULT_PROMPT}')",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="captions/nsd_qwen3vl",
        help="Directory for output JSONL file (default: captions/nsd_qwen3vl)",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Output JSONL filename; default is captions_<prompt_slug>.jsonl in output_dir",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip indices that already have a caption in the output file",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="First index (inclusive) to process (default: 0)",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="Last index (exclusive) to process (default: len(dataset))",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=MAX_NEW_TOKENS,
        help=f"Max new tokens per caption (default: {MAX_NEW_TOKENS})",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Number of images to caption per generation batch (default: 1)",
    )
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch_size must be at least 1")

    os.makedirs(args.output_dir, exist_ok=True)
    if args.output_file:
        out_path = os.path.join(args.output_dir, args.output_file)
    else:
        slug = prompt_to_slug(args.prompt)
        out_path = os.path.join(args.output_dir, f"captions_{slug}.jsonl")

    print("Loading NSD Dataset...")
    dataset = NSDStimulusSet()
    total = len(dataset)
    end = args.end if args.end is not None else total
    end = min(end, total)
    indices = list(range(args.start, end))

    if args.resume:
        done = get_existing_indices(out_path)
        indices = [i for i in indices if i not in done]
        print(f"Resume: {len(done)} already done, {len(indices)} remaining in range [{args.start}, {end})")
    else:
        print(f"Processing indices [{args.start}, {end}) ({len(indices)} images)")

    if not indices:
        print("Nothing to do. Exiting.")
        return

    print("Loading Qwen3-VL-8B-Instruct...")
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_NAME, torch_dtype="auto", device_map="auto"
    )
    print(
        f"Prompt: \"{args.prompt}\"\n"
        f"Output: {os.path.abspath(out_path)}\n"
        f"Requested batch size: {args.batch_size}\n"
    )

    mode = "a" if args.resume and os.path.isfile(out_path) else "w"
    current_batch_size = args.batch_size
    processed = 0
    with open(out_path, mode) as f:
        while processed < len(indices):
            batch_indices = indices[processed:processed + current_batch_size]
            print(
                f"[{processed + 1}-{processed + len(batch_indices)}/{len(indices)}] "
                f"NSD indices {batch_indices} (batch_size={len(batch_indices)})..."
            )

            try:
                pil_images = [unnormalize_to_pil(dataset[idx]) for idx in batch_indices]
                messages_batch = [build_messages(args.prompt, pil_img) for pil_img in pil_images]
                inputs = processor.apply_chat_template(
                    conversation=messages_batch,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                )
                inputs = inputs.to(model.device)
                generated_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
            except Exception as exc:
                if is_oom_error(exc) and len(batch_indices) > 1:
                    new_batch_size = max(1, len(batch_indices) // 2)
                    print(
                        f"CUDA OOM for batch of {len(batch_indices)} at indices {batch_indices}. "
                        f"Retrying with batch_size={new_batch_size}."
                    )
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    current_batch_size = new_batch_size
                    continue
                raise RuntimeError(
                    f"Caption generation failed for indices {batch_indices}"
                ) from exc

            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            captions = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            if len(captions) != len(batch_indices):
                raise RuntimeError(
                    f"Decoded {len(captions)} captions for {len(batch_indices)} inputs."
                )

            for idx, caption in zip(batch_indices, captions):
                rec = {"index": idx, "caption": caption}
                f.write(json.dumps(rec) + "\n")
                print(f"  NSD index {idx} -> {caption[:80]}{'...' if len(caption) > 80 else ''}")

            f.flush()
            os.fsync(f.fileno())  # force kernel to persist so resume doesn't lose last completed batch
            processed += len(batch_indices)

    print(f"\nDone. Captions written to {os.path.abspath(out_path)}")


if __name__ == "__main__":
    main()
