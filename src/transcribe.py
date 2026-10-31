"""Run a pretrained Whisper model on every clip in data/clips/.

Outputs, per clip:
  results/transcripts/<clip>.txt   cleaned transcript
  results/transcripts/<clip>.json  raw Whisper output plus the settings used
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cleanup import clean_transcript  # noqa: E402

CLIPS_DIR = ROOT / "data" / "clips"
OUT_DIR = ROOT / "results" / "transcripts"
AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".ogg", ".flac"}

# Settings are recorded in every output file so results can be reproduced.
MODEL_SIZE = "base"      # tiny | base | small | medium  (bigger = slower, usually more accurate)
LANGUAGE = "en"          # set to None to auto-detect, but test one language at a time


def main() -> None:
    import whisper  # imported here so the tests can run without whisper installed

    clips = sorted(p for p in CLIPS_DIR.iterdir() if p.suffix.lower() in AUDIO_EXTS)
    if not clips:
        sys.exit(f"No audio files found in {CLIPS_DIR}. Add 20-40 second clips there.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loading Whisper '{MODEL_SIZE}' model...")
    model = whisper.load_model(MODEL_SIZE)
    settings = {
        "engine": "openai-whisper",
        "whisper_version": getattr(whisper, "__version__", "unknown"),
        "model_size": MODEL_SIZE,
        "language": LANGUAGE,
        "fp16": False,
    }

    for clip in clips:
        print(f"Transcribing {clip.name} ...")
        raw = model.transcribe(str(clip), language=LANGUAGE, fp16=False)
        cleaned = clean_transcript(raw["text"])
        (OUT_DIR / f"{clip.stem}.txt").write_text(cleaned + "\n", encoding="utf-8")
        record = {"clip": clip.name, "settings": settings, "raw_text": raw["text"]}
        (OUT_DIR / f"{clip.stem}.json").write_text(
            json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Done. Transcripts saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
