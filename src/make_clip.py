"""Build a clip and its reference from LibriSpeech utterances.

Joins the given utterances (in order) into one mono 16 kHz WAV file in
data/clips/, and writes the matching transcript to data/references/.
The reference is the official LibriSpeech human transcript; listen to the
clip and correct the reference if you want a fully hand-checked benchmark.

Usage (run from the project folder):
  python src\\make_clip.py <chapter_folder> <name> <utt_id> [<utt_id> ...]

Example:
  python src\\make_clip.py librispeech\\LibriSpeech\\dev-clean\\1272\\128104 clip1 1272-128104-0000 1272-128104-0001 1272-128104-0002
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from check_env import check  # noqa: E402


def load_transcripts(chapter: Path) -> dict[str, str]:
    """Map utterance ID -> text from the chapter's .trans.txt file."""
    texts = {}
    for trans_file in chapter.glob("*.trans.txt"):
        for line in trans_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                utt_id, _, text = line.partition(" ")
                texts[utt_id] = text.strip()
    return texts


def main() -> None:
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    chapter = Path(sys.argv[1])
    name = sys.argv[2]
    utt_ids = sys.argv[3:]

    if not check():
        sys.exit(1)

    texts = load_transcripts(chapter)
    missing = [u for u in utt_ids if u not in texts or not (chapter / f"{u}.flac").exists()]
    if missing:
        sys.exit(f"Not found in {chapter}: {', '.join(missing)}. Check the IDs with: dir {chapter}\\*.flac")

    inputs = []
    for u in utt_ids:
        inputs += ["-i", str(chapter / f"{u}.flac")]
    n = len(utt_ids)
    labels = "".join(f"[{i}:a]" for i in range(n))
    out_wav = ROOT / "data" / "clips" / f"{name}.wav"
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", *inputs,
           "-filter_complex", f"{labels}concat=n={n}:v=0:a=1",
           "-ac", "1", "-ar", "16000", str(out_wav)]
    subprocess.run(cmd, check=True)

    ref_dir = ROOT / "data" / "references"
    ref_dir.mkdir(parents=True, exist_ok=True)
    reference = " ".join(texts[u] for u in utt_ids).lower()
    (ref_dir / f"{name}.txt").write_text(reference + "\n", encoding="utf-8")
    print(f"Created {out_wav}")
    print(f"Created {ref_dir / (name + '.txt')}")


if __name__ == "__main__":
    main()