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
import os
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
    p.add_argument("--highlight", action="store_true",
                   help="Generate highlighted HTML report instead of JSON")
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


def _extract_doc_type(vlm_json: str) -> str:
    try:
        return json.loads(vlm_json).get("doc_type", "")
    except Exception:
        return ""


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def _make_rel(abs_path: str, output_path: Path) -> str:
    """Return a forward-slash relative path from output_dir to abs_path."""
    out_dir = str(output_path.parent).replace("\\", "/")
    norm_path = abs_path.replace("\\", "/")
    rel = os.path.relpath(norm_path, out_dir)
    return rel.replace("\\", "/")


_CSS = """\
<style>
body{font-family:system-ui,sans-serif;max-width:960px;margin:2em auto;padding:0 1em;background:#f8f9fa;color:#1a1a2e}
h1{border-bottom:2px solid #16213e;padding-bottom:.3em}
.sample{background:#fff;border-radius:8px;padding:1.2em;margin-bottom:1.5em;box-shadow:0 1px 4px rgba(0,0,0,.08)}
.sample h3{margin-top:0}
code,pre{background:#f0f0f5;padding:2px 6px;border-radius:4px;font-size:.85em}
pre{padding:.8em;overflow-x:auto;max-height:300px;overflow-y:auto}
.match{display:flex;align-items:center;gap:.8em;padding:.5em .8em;border-radius:6px;margin:.3em 0}
.match.gt{background:#d4edda;border-left:4px solid #28a745}
.match.near{background:#fff3cd;border-left:4px solid #ffc107}
.match.other{background:#f8f9fa;border-left:4px solid #dee2e6}
.match img{width:80px;height:60px;object-fit:cover;border-radius:4px;flex-shrink:0}
.match-info{flex:1}
.match-name{font-weight:600}
.match-score{font-size:.85em;color:#555}
.tag{display:inline-block;padding:2px 8px;border-radius:12px;font-size:.75em;font-weight:600;margin-left:.5em}
.tag-form{background:#e3f2fd;color:#1565c0}.tag-invoice{background:#fce4ec;color:#c62828}
.tag-receipt{background:#e8f5e9;color:#2e7d32}.tag-table{background:#f3e5f5;color:#6a1b9a}
.tag-default{background:#eee;color:#333}
.sec{font-size:.75em;text-transform:uppercase;letter-spacing:.05em;color:#888;margin:.5em 0 .2em}
.gt-badge{color:#28a745;font-weight:700;font-size:.8em}
</style>
"""


def build_html(samples_results: List[Dict[str, Any]], output_path: Path) -> str:
    parts = [
        "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>",
        "<title>Embed Search Report</title>", _CSS, "</head><body>",
        "<h1>Embedding Search Report</h1>",
        f"<p style='color:#555'>Generated: {__import__('datetime').datetime.now().isoformat()[:19]}</p><hr>",
    ]
    for sr in samples_results:
        name = sr["sample_name"]
        doc_type = sr.get("doc_type", "")
        vlm_text = sr.get("vlm_text", "")
        ocr_text = sr.get("ocr_text", "")
        matches = sr.get("matches", [])

        tag_cls = f"tag tag-{doc_type}" if doc_type else "tag tag-default"
        tag_html = f"<span class='{tag_cls}'>{doc_type or 'unknown'}</span>"

        parts.append(f"<div class='sample'>")
        parts.append(f"<h3>{name} &nbsp;{tag_html}</h3>")

        if vlm_text:
            parts.append("<div class='sec'>VLM Extraction</div>")
            parts.append(f"<pre>{_esc(vlm_text[:800])}</pre>")
        if ocr_text:
            parts.append("<div class='sec'>OCR Text</div>")
            parts.append(f"<pre>{_esc(ocr_text[:600])}</pre>")

        parts.append("<div class='sec'>Top Matches</div>")
        for m in matches:
            is_gt = m["is_ground_truth"]
            cls = "gt" if is_gt else ("near" if m["similarity"] >= 0.25 else "other")
            gt_lbl = f'<span class="gt-badge"> ★ GROUND TRUTH</span>' if is_gt else ""
            img_src = _make_rel(m["image"], output_path)
            title = ' title="Ground truth match"' if is_gt else ""
            parts.append(
                f"<div class='match {cls}'>"
                f"<img src='{img_src}' alt='{m['name']}'{title} />"
                f"<div class='match-info'>"
                f"<span class='match-name'>{m['name']}</span>{gt_lbl}<br>"
                f"<span class='match-score'>similarity: {m['similarity']:.4f}</span>"
                f"</div></div>"
            )
        parts.append("</div>")
    parts += ["</body></html>"]
    return "\n".join(parts)


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
                "name": png_files[idx].name,
                "similarity": round(score, 4),
                "is_ground_truth": png_files[idx].name == sample["name"],
            })

        results.append({
            "sample_name": sample["name"],
            "doc_type": _extract_doc_type(sample["vlm_text"]),
            "vlm_text": sample["vlm_text"],
            "ocr_text": sample["ocr_text"],
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

    if args.highlight:
        out_path = Path(args.output) if args.output else input_path.with_suffix(".html")
        html = build_html(results, out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(html, encoding="utf-8")
        print(f"HTML report → {out_path}", file=sys.stderr)
        return 0

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
