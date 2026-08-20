"""
Zenodo10K Candidate Downloader

This script downloads the top candidate PPTX presentations identified during
Stage 1 metadata screening (Indonesian and English).
It checks local storage first and skips files that are already present.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

_PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DEFAULT_REPO_ID = "Forceless/Zenodo10K"
DEFAULT_INPUT_DIR = _PROJECT_ROOT / "dataset" / "metadata"
DEFAULT_OUTPUT_DIR = _PROJECT_ROOT / "dataset" / "download"


def get_target_filename(candidate: dict[str, str], use_rank_prefix: bool = True) -> str:
    """Generate the standardized local filename for a candidate."""
    filename = candidate.get("filename", "").strip()
    rank_str = candidate.get("candidate_rank", "").strip()
    if use_rank_prefix and rank_str:
        try:
            rank_num = int(rank_str)
            return f"{rank_num:02d}_{filename}"
        except ValueError:
            pass
    return filename


def check_local_file_exists(
    target_dir: Path, candidate: dict[str, str], use_rank_prefix: bool = True
) -> tuple[bool, Path | None, int]:
    """
    Check if the candidate file already exists locally.
    Checks both prefixed name (e.g. 01_file.pptx) and plain name (file.pptx).
    Returns (exists: bool, file_path: Optional[Path], file_size_bytes: int).
    """
    raw_filename = candidate.get("filename", "").strip()
    prefixed_filename = get_target_filename(candidate, use_rank_prefix=True)
    plain_filename = get_target_filename(candidate, use_rank_prefix=False)

    # 1. Check preferred destination path
    primary_name = prefixed_filename if use_rank_prefix else plain_filename
    primary_path = target_dir / primary_name
    if (
        primary_path.exists()
        and primary_path.is_file()
        and primary_path.stat().st_size > 0
    ):
        return True, primary_path, primary_path.stat().st_size

    # 2. Check alternative name if previously downloaded without prefix
    if use_rank_prefix:
        alt_path = target_dir / plain_filename
        if alt_path.exists() and alt_path.is_file() and alt_path.stat().st_size > 0:
            # Rename to standard prefixed name
            try:
                alt_path.rename(primary_path)
                return True, primary_path, primary_path.stat().st_size
            except Exception:
                return True, alt_path, alt_path.stat().st_size

    return False, None, 0


def load_candidates_csv(csv_path: Path, top_n: int) -> list[dict[str, str]]:
    """Load top N candidate records from a candidate CSV file."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        print(f"Warning: Candidate CSV not found at {csv_path}")
        return []

    records: list[dict[str, str]] = []
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        sample = f.read(4096)
        f.seek(0)
        delimiter = ","
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=";,")
            delimiter = dialect.delimiter
        except Exception:
            first_line = sample.splitlines()[0] if sample else ""
            if ";" in first_line and first_line.count(";") > first_line.count(","):
                delimiter = ";"

        reader = csv.DictReader(f, delimiter=delimiter)
        for row in reader:
            clean_row = {
                (k.strip() if k else ""): (v.strip() if v else "")
                for k, v in row.items()
                if k
            }
            if clean_row.get("filename"):
                records.append(clean_row)
            if len(records) >= top_n:
                break
    return records


def build_repo_file_index(repo_id: str) -> dict[str, str]:
    """
    Fetch repo file tree and create an index from filename to full repo path.
    Example repo path: pptx/cc-by-2.0/2020/0a0c7478080d6346c27a1c6c556feb29-ppt_penulisan artikel RJI.pptx
    """
    print(f"Connecting to Hugging Face Hub to index repo '{repo_id}'...")
    api = HfApi()
    all_files = api.list_repo_files(repo_id, repo_type="dataset")

    index: dict[str, str] = {}
    for path in all_files:
        if not path.lower().endswith(".pptx"):
            continue
        basename = os.path.basename(path)
        # Match filename after the hash prefix (e.g. '<hash>-<filename>')
        if "-" in basename:
            orig_name = basename.split("-", 1)[1]
            index[orig_name] = path
            index[orig_name.lower()] = path
        index[basename] = path
        index[basename.lower()] = path
        index[path] = path

    print(f"Indexed {len(index)} PPTX lookup references from repository.")
    return index


def resolve_repo_path(filename: str, repo_index: dict[str, str]) -> str | None:
    """Find the full repository path for a given candidate filename."""
    if filename in repo_index:
        return repo_index[filename]
    if filename.lower() in repo_index:
        return repo_index[filename.lower()]
    # Suffix search fallback
    for orig_name, repo_path in repo_index.items():
        if repo_path.endswith(filename) or repo_path.lower().endswith(filename.lower()):
            return repo_path
    return None


def download_single_candidate(
    candidate: dict[str, str],
    language: str,
    repo_id: str,
    repo_index: dict[str, str],
    target_dir: Path,
    use_rank_prefix: bool = True,
    force: bool = False,
) -> tuple[bool, str, str]:
    """
    Download a single candidate PPTX.
    Returns (success: bool, filename: str, message: str).
    """
    filename = candidate.get("filename", "").strip()
    save_name = get_target_filename(candidate, use_rank_prefix=use_rank_prefix)

    if not filename:
        return False, "", "Empty filename"

    target_path = target_dir / save_name

    # Check local file existence before downloading
    if not force:
        exists, existing_path, size_bytes = check_local_file_exists(
            target_dir, candidate, use_rank_prefix=use_rank_prefix
        )
        if exists:
            return (
                True,
                save_name,
                f"Already exists locally ({size_bytes / (1024 * 1024):.2f} MB) - skipped",
            )

    repo_path = resolve_repo_path(filename, repo_index)
    if not repo_path:
        return (
            False,
            save_name,
            f"File not found in Hugging Face repository '{repo_id}'",
        )

    try:
        downloaded_file = hf_hub_download(
            repo_id=repo_id,
            filename=repo_path,
            repo_type="dataset",
        )
        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(downloaded_file, target_path)
        actual_size = target_path.stat().st_size
        return (
            True,
            save_name,
            f"Downloaded successfully ({actual_size / (1024 * 1024):.2f} MB)",
        )
    except Exception as e:
        return False, save_name, f"Download failed: {e}"


def run_download(
    top_n: int = 50,
    limit: int | None = None,
    input_dir: Path = DEFAULT_INPUT_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    repo_id: str = DEFAULT_REPO_ID,
    max_workers: int = 4,
    use_rank_prefix: bool = True,
    force: bool = False,
) -> None:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    indo_csv = input_dir / "zenodo10k_indonesian_candidates.csv"
    eng_csv = input_dir / "zenodo10k_english_candidates.csv"

    indo_candidates = load_candidates_csv(indo_csv, top_n=top_n)
    eng_candidates = load_candidates_csv(eng_csv, top_n=top_n)

    tasks: list[tuple[dict[str, str], str, Path]] = []
    indo_dir = output_dir / "indonesian"
    eng_dir = output_dir / "english"

    indo_dir.mkdir(parents=True, exist_ok=True)
    eng_dir.mkdir(parents=True, exist_ok=True)

    for c in indo_candidates:
        tasks.append((c, "Indonesian", indo_dir))
    for c in eng_candidates:
        tasks.append((c, "English", eng_dir))

    if limit is not None and limit > 0:
        tasks = tasks[:limit]

    total_tasks = len(tasks)
    if total_tasks == 0:
        print("No candidate tasks to process.")
        return

    # Check local availability status for all tasks
    already_existing: list[tuple[dict[str, str], str, Path, int]] = []
    need_download: list[tuple[dict[str, str], str, Path]] = []

    for task in tasks:
        cand, lang, dest_dir = task
        if force:
            need_download.append(task)
        else:
            exists, path, size = check_local_file_exists(
                dest_dir, cand, use_rank_prefix=use_rank_prefix
            )
            if exists:
                already_existing.append((cand, lang, dest_dir, size))
            else:
                need_download.append(task)

    print("Zenodo10K Candidate Downloader")
    print("==============================")
    print(f"Total candidate files to verify : {total_tasks}")
    print(f"Already exists locally (cached) : {len(already_existing)}")
    print(f"Files needing download          : {len(need_download)}")
    print("=" * 60)

    if len(need_download) == 0:
        print("\nAll files are already downloaded and present locally! Nothing to do.")
        print(f"Output directories:\n  - {indo_dir}\n  - {eng_dir}")
        return

    repo_index = build_repo_file_index(repo_id)

    print(f"\nStarting download for {len(need_download)} file(s)...")
    print("=" * 60)

    success_count = len(already_existing)
    fail_count = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {
            executor.submit(
                download_single_candidate,
                task[0],
                task[1],
                repo_id,
                repo_index,
                task[2],
                use_rank_prefix,
                force,
            ): task
            for task in need_download
        }

        for idx, future in enumerate(as_completed(future_to_task), start=1):
            task = future_to_task[future]
            cand, lang, dest_dir = task
            filename = cand.get("filename", "")
            rank = cand.get("candidate_rank", "?")

            try:
                success, fname, msg = future.result()
                if success:
                    success_count += 1
                    status = "[OK]"
                else:
                    fail_count += 1
                    status = "[FAIL]"
                print(
                    f"[{idx}/{len(need_download)}] {status} [{lang} #{rank}] {fname} -> {msg}"
                )
            except Exception as exc:
                fail_count += 1
                print(
                    f"[{idx}/{len(need_download)}] [ERROR] [{lang} #{rank}] {filename} -> Exception: {exc}"
                )

    print("=" * 60)
    print(
        f"Summary: {success_count} ready locally ({len(already_existing)} cached, {success_count - len(already_existing)} newly downloaded), {fail_count} failed out of {total_tasks} total."
    )
    print(f"Output directories:\n  - {indo_dir}\n  - {eng_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download top Zenodo10K Indonesian and English PPTX candidates with local existence check."
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=50,
        help="Number of candidates to load from each candidate CSV (default: 50)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional safety cap on total files to download (e.g. --limit 2 for testing)",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing candidate CSV files (default: dataset/metadata)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for downloaded PPTX files (default: dataset/download)",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=DEFAULT_REPO_ID,
        help=f"Hugging Face dataset repository ID (default: {DEFAULT_REPO_ID})",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Number of concurrent download workers (default: 4)",
    )
    parser.add_argument(
        "--no-rank-prefix",
        action="store_true",
        help="Do not prefix saved filenames with candidate rank.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force redownload even if local file already exists.",
    )

    args = parser.parse_args()

    run_download(
        top_n=args.top_n,
        limit=args.limit,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        repo_id=args.repo_id,
        max_workers=args.max_workers,
        use_rank_prefix=not args.no_rank_prefix,
        force=args.force,
    )


if __name__ == "__main__":
    main()
