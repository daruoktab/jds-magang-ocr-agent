"""
Embedding-based image retrieval over the FiftyOne datatest dataset.

Reads a run_eval_report.md (or raw JSON output from main.py), extracts the
embedded text (VLM extraction + OCR) per sample, then uses a multimodal
embedding model to find the most visually+semantically similar images from
the full dataset.

Usage:
    # From a report.md produced by run_eval_report.py:
    python scripts/run_embed_search.py --input output/report.md

    # From a single JSON result file (output of main.py --out-dir):
    python scripts/run_embed_search.py --input input/datatest/data/train_000000.json

    # Specify dataset dir, top-k, and model
    python scripts/run_embed_search.py --input output/report.md --dataset input/datatest/data --top-k 5
    python scripts/run_embed_search.py --input output/report.md --model clip-ViT-B-32 --device cpu
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

os_env = {}

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


MODEL_ID = "clip-ViT-B-32"
DEFAULT_DATASET_DIR = r"input/datatest/data"


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="run_embed_search",
        description="Embed search: match report/JSON samples against full image dataset.",
    )
    p.add_argument("--input", required=True,
                   help="Path to report.md or JSON result from main.py")
    p.add_argument("--dataset", default=DEFAULT_DATASET_DIR,
                   help=f"Directory of PNG images to search (default: {DEFAULT_DATASET_DIR})")
    p.add_argument("--top-k", type=int, default=5,
                   help="Number of closest images to return per sample (default: 5)")
    p.add_argument("--output", default=None,
                   help="Path to write results JSON (default: stdout)")
    p.add_argument("--batch-size", type=int, default=16,
                   help="Batch size for embedding (default: 16)")
    p.add_argument("--model", default=MODEL_ID,
                   help=f"Model id for SentenceTransformer (default: {MODEL_ID})")
    p.add_argument("--device", default="auto",
                   help="Device: auto (prefer GPU), cpu, cuda (default: auto)")
    return p.parse_args(argv)


def extract_samples_from_report(report_path: Path) -> List[Dict[str, Any]]:
    """Parse run_eval_report.md and extract per-sample image path + text."""
    text = report_path.read_text(encoding="utf-8")
    samples: List[Dict[str, Any]] = []

    # Split by sample headers: "## Sample NNN — filename.png"
    parts = re.split(r"^## Sample \d+ — ", text, flags=re.MULTILINE)
    if len(parts) <= 1:
        # Try alternative format
        parts = re.split(r"^## ", text, flags=re.MULTILINE)

    for i, part in enumerate(parts):
        lines = part.strip().splitlines()
        if not lines:
            continue

        name = lines[0].strip() if lines else f"sample_{i:03d}.png"
        rest = "\n".join(lines[1:])

        # Extract image path from markdown image link: ![name](path)
        img_match = re.search(r"!\[.*?\]\((.+?)\)", rest)
        image_path = img_match.group(1).replace("\\", "/") if img_match else name

        # Extract OCR text from ``` block
        ocr_match = re.search(r"- \*\*OCR text:\*\*\s*\n```(?:text)?\n(.*?)\n```", rest, re.DOTALL)
        ocr_text = ocr_match.group(1).strip() if ocr_match else ""

        # Extract VLM extraction JSON
        vlm_match = re.search(r"- \*\*VLM extraction:\*\*\s*\n```(?:json)?\n(.*?)\n```", rest, re.DOTALL)
        vlm_text = ""
        if vlm_match:
            try:
                vlm_obj = json.loads(vlm_match.group(1))
                vlm_text = json.dumps(vlm_obj, ensure_ascii=False)
            except json.JSONDecodeError:
                vlm_text = vlm_match.group(1).strip()

        combined = f"{vlm_text}\n{ocr_text}".strip()
        if combined:
            samples.append({
                "name": name,
                "image_path": image_path,
                "vlm_text": vlm_text,
                "ocr_text": ocr_text,
                "combined": combined,
            })

    return samples


def extract_samples_from_json(json_path: Path) -> List[Dict[str, Any]]:
    """Parse a single JSON result from main.py output."""
    data = json.loads(json_path.read_text(encoding="utf-8"))
    final = data.get("final_result", data)
    extraction = final.get("extraction", {})
    ocr_text = ""
    vlm_text = json.dumps(extraction, ensure_ascii=False)

    # Try to infer image name from context
    image_name = json_path.stem + ".png"
    return [{
        "name": image_name,
        "image_path": str(json_path),
        "vlm_text": vlm_text,
        "ocr_text": ocr_text,
        "combined": vlm_text,
    }]


def discover_dataset_images(dataset_dir: Path) -> List[Path]:
    """Find all PNG files in the dataset directory."""
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    png_files = sorted(dataset_dir.glob("*.png"))
    subdirs = [d for d in dataset_dir.iterdir() if d.is_dir()]
    for d in subdirs:
        png_files.extend(sorted(d.glob("*.png")))
    return png_files


def _resolve_device(requested: str) -> str:
    """Resolve 'auto' to best available device (xpu > cuda > cpu)."""
    if requested != "auto":
        return requested
    import torch
    if torch.xpu.is_available():
        return "xpu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_model(model_id: str, device: str = "cpu"):
    """Lazy-load the embedding model, falling back to CPU on error."""
    if SentenceTransformer is None:
        raise ImportError("sentence_transformers not installed. Run: pip install sentence-transformers")
    print(f"Loading embedding model: {model_id} (device={device}) ...", file=sys.stderr, flush=True)
    try:
        return SentenceTransformer(model_id, device=device)
    except Exception as e:
        if device != "cpu":
            print(f"  {type(e).__name__}: {e} -> falling back to CPU", file=sys.stderr, flush=True)
            return SentenceTransformer(model_id, device="cpu")
        raise


def embed_texts(model, texts: List[str], batch_size: int) -> Any:
    return model.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)


def embed_images(model, image_paths: List[str], batch_size: int) -> Any:
    return model.encode(image_paths, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True,
                        prompt="Describe this image in detail: ")


def cosine_similarity(a: Any, b: Any) -> Any:
    """Compute cosine similarity between two arrays/matrices."""
    import numpy as np
    a_norm = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-8)
    b_norm = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-8)
    return a_norm @ b_norm.T


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    input_path = Path(args.input)
    dataset_dir = Path(args.dataset)
    top_k = args.top_k
    batch_size = args.batch_size

    # --- Discover dataset images ----------------------------------------
    print(f"Scanning dataset: {dataset_dir}", file=sys.stderr, flush=True)
    png_files = discover_dataset_images(dataset_dir)
    if not png_files:
        print(f"No PNG files found in {dataset_dir}", file=sys.stderr)
        return 1
    print(f"Found {len(png_files)} images in dataset.", file=sys.stderr, flush=True)

    # --- Load embedding model -------------------------------------------
    device = _resolve_device(args.device)
    print(f"Using device: {device}", file=sys.stderr, flush=True)
    model = load_model(args.model, device=device)

    # --- Embed dataset images -------------------------------------------
    print(f"Embedding {len(png_files)} dataset images ...", file=sys.stderr, flush=True)
    img_paths_str = [str(p) for p in png_files]
    img_embeddings = embed_images(model, img_paths_str, batch_size)
    print(f"Dataset embeddings shape: {img_embeddings.shape}", file=sys.stderr, flush=True)

    # --- Parse input samples --------------------------------------------
    if input_path.suffix == ".md":
        samples = extract_samples_from_report(input_path)
    elif input_path.suffix == ".json":
        samples = extract_samples_from_json(input_path)
    else:
        print(f"Unsupported input format: {input_path.suffix}. Use .md or .json", file=sys.stderr)
        return 1

    print(f"Extracted {len(samples)} samples from input.", file=sys.stderr, flush=True)

    if not samples:
        print("No samples found in input.", file=sys.stderr)
        return 1

    # --- Embed query texts and search -----------------------------------
    results = []
    for sample in samples:
        combined = sample["combined"]
        if not combined:
            print(f"  SKIP {sample['name']}: no text content", file=sys.stderr)
            continue

        query_emb = embed_texts(model, [combined], batch_size)
        sims = cosine_similarity(query_emb, img_embeddings)[0]

        # Top-k indices (descending similarity)
        top_indices = sims.argsort()[::-1][:top_k]
        matches = []
        for idx in top_indices:
            score = float(sims[idx])
            matches.append({
                "image": str(png_files[idx]),
                "similarity": round(score, 4),
            })

        results.append({
            "sample_name": sample["name"],
            "query_text_preview": combined[:200],
            "matches": matches,
        })
        print(f"  {sample['name']}: top match = {matches[0]['image']} ({matches[0]['similarity']:.4f})",
              file=sys.stderr, flush=True)

    # --- Write output ---------------------------------------------------
    output_data = {
        "model": args.model,
        "dataset_dir": str(dataset_dir),
        "num_dataset_images": len(png_files),
        "num_samples": len(results),
        "top_k": top_k,
        "results": results,
    }

    output_str = json.dumps(output_data, indent=2, ensure_ascii=False)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output_str, encoding="utf-8")
        print(f"Results written to {out_path}", file=sys.stderr)
    else:
        print(output_str)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
