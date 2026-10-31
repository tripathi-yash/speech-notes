"""Compare each transcript with its manually checked reference.

Reads:   data/references/<clip>.txt        (typed by hand while listening)
         results/transcripts/<clip>.txt    (written by transcribe.py)
Writes:  results/metrics.csv               (one row per clip)
         results/error_review.md           (every mismatch, grouped by clip)
"""
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from wer import compute_wer, error_lines  # noqa: E402

REF_DIR = ROOT / "data" / "references"
HYP_DIR = ROOT / "results" / "transcripts"
RESULTS = ROOT / "results"


def evaluate() -> list[dict]:
    rows, review = [], []
    refs = sorted(REF_DIR.glob("*.txt"))
    if not refs:
        sys.exit(f"No reference transcripts in {REF_DIR}. Write one per clip.")

    for ref_path in refs:
        hyp_path = HYP_DIR / ref_path.name
        if not hyp_path.exists():
            print(f"Skipping {ref_path.stem}: no transcript yet. Run transcribe.py first.")
            continue
        result = compute_wer(ref_path.read_text(encoding="utf-8"),
                             hyp_path.read_text(encoding="utf-8"))
        rows.append({
            "clip": ref_path.stem,
            "reference_words": result.ref_words,
            "substitutions": result.substitutions,
            "deletions": result.deletions,
            "insertions": result.insertions,
            "wer_percent": round(100 * result.wer, 2),
        })
        review.append(f"## {ref_path.stem}\n")
        mismatches = error_lines(result)
        if mismatches:
            review.extend(f"- {m}" for m in mismatches)
        else:
            review.append("- No word-level errors.")
        review.append("")

    if rows:
        RESULTS.mkdir(parents=True, exist_ok=True)
        with open(RESULTS / "metrics.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        (RESULTS / "error_review.md").write_text(
            "# Error review\n\nWord-level mismatches after normalising case and punctuation.\n\n"
            + "\n".join(review), encoding="utf-8")
    return rows


if __name__ == "__main__":
    for row in evaluate():
        print(f"{row['clip']}: WER {row['wer_percent']}% "
              f"(S={row['substitutions']} D={row['deletions']} I={row['insertions']})")
