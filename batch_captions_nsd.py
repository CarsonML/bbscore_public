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
    args = parser.parse_args()

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
    print(f"Prompt: \"{args.prompt}\"\nOutput: {os.path.abspath(out_path)}\n")

    mode = "a" if args.resume and os.path.isfile(out_path) else "w"
    with open(out_path, mode) as f:
        for k, idx in enumerate(indices):
            print(f"[{k+1}/{len(indices)}] NSD index {idx}...")
            tensor_img = dataset[idx]
            pil_img = unnormalize_to_pil(tensor_img)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": pil_img},
                        {"type": "text", "text": args.prompt},
                    ],
                }
            ]
            inputs = processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
            )
            inputs = inputs.to(model.device)
            generated_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            caption = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            rec = {"index": idx, "caption": caption}
            f.write(json.dumps(rec) + "\n")
            f.flush()
            os.fsync(f.fileno())  # force kernel to persist so resume doesn't lose last caption on kill
            print(f"  -> {caption[:80]}{'...' if len(caption) > 80 else ''}")

    print(f"\nDone. Captions written to {os.path.abspath(out_path)}")


if __name__ == "__main__":
    main()
