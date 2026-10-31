---
title: Speech Notes
emoji: 🎙️
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 5.50.0
app_file: app.py
pinned: false
---

# Speech Notes

Turn a short recording into text, and measure accuracy against a reference transcript using Word Error Rate (WER). Transcription uses the open-source Whisper model.

## Structure

- `app.py`: web app (Gradio)
- `src/`: transcription, clean-up, WER and helper code
- `tests/`: unit tests (no model needed)
- `run_pipeline.py`: command-line pipeline for a folder of clips and references
- `requirements.txt`, `packages.txt`: dependencies (`packages.txt` lists ffmpeg)

## Run locally

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Open `http://localhost:7860`. Upload or record a clip, optionally paste its reference,
and click **Transcribe**.

## Command-line pipeline

1. Put 2–3 clips of 20–40 seconds in `data/clips/`, and their reference transcripts in
   `data/references/` with the same file names (`clip1.wav` → `clip1.txt`).
2. Run `python run_pipeline.py`.
3. See `results/metrics.csv`, `results/error_review.md` and `report/speech_notes_report.pdf`.

## Tests

```
python -m unittest discover -s tests -v
```
## Clone 

```
git clone https://github.com/tripathi-yash/speech-notes.git

```

## Deployed 

```
https://huggingface.co/spaces/Ykode/speech-notes
```
## Responsible use

- Use only consented or public audio.
- Locally, Whisper runs on your computer. On Hugging Face Spaces, it runs on the Space's server.
- Check transcripts before relying on them.