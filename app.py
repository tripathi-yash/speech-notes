"""Speech Notes: Gradio web app for Hugging Face Spaces (ZeroGPU)."""
import sys
import tempfile
import time
from pathlib import Path

import gradio as gr
import whisper

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from check_env import check as check_ffmpeg  # noqa: E402
from cleanup import clean_transcript  # noqa: E402
from wer import compute_wer, error_lines  # noqa: E402

try:
    import spaces
    gpu = spaces.GPU
except ImportError:  # running locally, where the spaces package is not installed
    def gpu(fn):
        return fn

MODEL_SIZE = "base"
FFMPEG_OK = check_ffmpeg()
_model = None
HISTORY = []  # kept in memory; cleared when the Space restarts

THEME = gr.themes.Base(
    primary_hue="violet",
    secondary_hue="fuchsia",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
).set(
    body_background_fill="#0b1020",
    body_text_color="#e2e8f0",
    block_background_fill="rgba(255,255,255,0.04)",
    block_border_color="rgba(255,255,255,0.08)",
    block_radius="18px",
    block_label_text_color="#94a3b8",
    input_background_fill="rgba(255,255,255,0.06)",
    input_border_color="rgba(255,255,255,0.12)",
    input_radius="12px",
    button_primary_background_fill="#6366f1",
    button_primary_text_color="#ffffff",
)

CSS = """
body { background: #0b1020 !important; }
body::before {
  content: ""; position: fixed; inset: 0; z-index: -1; pointer-events: none;
  background:
    radial-gradient(900px 520px at 8% -8%, rgba(99,102,241,0.38), transparent 60%),
    radial-gradient(820px 520px at 100% 0%, rgba(236,72,153,0.26), transparent 60%),
    radial-gradient(760px 520px at 50% 115%, rgba(34,211,238,0.18), transparent 60%);
}
.gradio-container { background: transparent !important; max-width: 1200px !important; margin: auto !important; }
footer { display: none !important; }

/* Hero */
.hero {
  position: relative; overflow: hidden; border-radius: 28px; padding: 46px 46px 38px;
  background: linear-gradient(135deg, rgba(99,102,241,0.36), rgba(168,85,247,0.24) 45%, rgba(236,72,153,0.22));
  border: 1px solid rgba(255,255,255,0.12); backdrop-filter: blur(18px);
  box-shadow: 0 30px 80px -30px rgba(99,102,241,0.6); margin-bottom: 22px;
}
.hero h1 {
  font-size: 48px; font-weight: 800; letter-spacing: -0.03em; margin: 0 0 10px;
  background: linear-gradient(90deg, #c7d2fe, #f0abfc, #fbcfe8, #c7d2fe);
  background-size: 300% 100%; -webkit-background-clip: text; background-clip: text;
  color: transparent; animation: shimmer 7s linear infinite;
}
@keyframes shimmer { to { background-position: 300% 0; } }
.hero p { color: #cbd5e1; font-size: 17px; max-width: 620px; margin: 0 0 20px; line-height: 1.55; }
.steps { display: flex; gap: 10px; flex-wrap: wrap; }
.step {
  display: flex; align-items: center; gap: 9px; padding: 8px 15px; border-radius: 999px;
  background: rgba(255,255,255,0.08); border: 1px solid rgba(255,255,255,0.14);
  color: #e2e8f0; font-size: 13px; font-weight: 600;
}
.step b {
  width: 22px; height: 22px; border-radius: 50%; display: grid; place-items: center;
  background: linear-gradient(135deg, #818cf8, #e879f9); color: #fff; font-size: 12px;
}

/* Panels */
.panel {
  background: rgba(255,255,255,0.04) !important; border: 1px solid rgba(255,255,255,0.09) !important;
  border-radius: 24px !important; padding: 26px !important; backdrop-filter: blur(16px);
  box-shadow: 0 24px 56px -30px rgba(0,0,0,0.7) !important;
}
.sec { display: flex; align-items: center; gap: 10px; font-size: 14px; font-weight: 700; color: #e2e8f0; margin: 6px 0 12px; }
.sec i {
  width: 28px; height: 28px; border-radius: 10px; display: grid; place-items: center; font-style: normal;
  font-size: 13px; background: linear-gradient(135deg, rgba(129,140,248,0.35), rgba(232,121,249,0.3));
  border: 1px solid rgba(255,255,255,0.15);
}

/* Button */
.primary-btn {
  height: 56px !important; border-radius: 16px !important; border: none !important;
  font-size: 16px !important; font-weight: 700 !important; color: #ffffff !important;
  background: linear-gradient(120deg, #6366f1, #a855f7, #ec4899, #6366f1) !important;
  background-size: 250% 100% !important; animation: btnflow 6s ease infinite;
  box-shadow: 0 16px 36px -14px rgba(168,85,247,0.9) !important; transition: transform 0.15s ease;
}
.primary-btn:hover { transform: translateY(-2px); }
@keyframes btnflow { 0%, 100% { background-position: 0% 50%; } 50% { background-position: 100% 50%; } }

/* Stat cards */
.stats { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }
.stat {
  border-radius: 18px; padding: 16px 10px; text-align: center;
  background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1);
}
.stat .v { font-size: 26px; font-weight: 800; color: #ffffff; }
.stat .k { font-size: 11px; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.1em; margin-top: 4px; }
.stat.good { background: linear-gradient(180deg, rgba(34,197,94,0.2), rgba(34,197,94,0.05)); border-color: rgba(34,197,94,0.4); }
.stat.good .v { color: #86efac; }
.stat.warn { background: linear-gradient(180deg, rgba(245,158,11,0.2), rgba(245,158,11,0.05)); border-color: rgba(245,158,11,0.4); }
.stat.warn .v { color: #fcd34d; }
.stat.bad { background: linear-gradient(180deg, rgba(244,63,94,0.2), rgba(244,63,94,0.05)); border-color: rgba(244,63,94,0.4); }
.stat.bad .v { color: #fda4af; }
.ref-note { color: #94a3b8; font-size: 13px; margin: 10px 2px 0; }

.hint, .empty {
  text-align: center; color: #94a3b8; font-size: 14px; line-height: 1.5;
  border: 1px dashed rgba(255,255,255,0.16); border-radius: 18px; padding: 26px 16px;
}
.empty .big { font-size: 40px; margin-bottom: 6px; }
"""

HERO = """
<div class="hero">
  <h1>Speech Notes</h1>
  <p>Turn recordings into clean, searchable text, then measure exactly how accurate
     the transcript is.</p>
  <div class="steps">
    <div class="step"><b>1</b> Upload or record</div>
    <div class="step"><b>2</b> Transcribe with Whisper</div>
    <div class="step"><b>3</b> Measure with WER</div>
  </div>
</div>
"""

EMPTY = ('<div class="empty"><div class="big">🎧</div>'
         'Your transcript and accuracy report will appear here.</div>')
HINT = ('<div class="hint">Paste a reference transcript to see the accuracy (WER) and a '
        'breakdown of every error.</div>')


def get_model():
    """Load Whisper on the CPU once. It moves to the GPU only during transcription."""
    global _model
    if _model is None:
        _model = whisper.load_model(MODEL_SIZE, device="cpu")
    return _model


@gpu
def transcribe_audio(audio_path, language):
    """Runs on the GPU when ZeroGPU provides one, otherwise on the CPU."""
    import torch
    model = get_model()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return model.transcribe(audio_path, language=language, fp16=(device == "cuda"))


def stat_cards(wer_percent, subs, dels, ins, words):
    tone = "good" if wer_percent <= 10 else "warn" if wer_percent <= 25 else "bad"
    return f"""
<div class="stats">
  <div class="stat {tone}"><div class="v">{wer_percent:.2f}%</div><div class="k">WER</div></div>
  <div class="stat"><div class="v">{subs}</div><div class="k">Substitutions</div></div>
  <div class="stat"><div class="v">{dels}</div><div class="k">Deletions</div></div>
  <div class="stat"><div class="v">{ins}</div><div class="k">Insertions</div></div>
</div>
<div class="ref-note">Measured against {words} reference words. Lower is better.</div>
"""


def run(audio_path, reference, language):
    if not audio_path:
        raise gr.Error("Upload or record a clip first.")
    if not FFMPEG_OK:
        raise gr.Error("ffmpeg is not available on this server.")

    lang = None if language == "auto" else language
    raw = transcribe_audio(audio_path, lang)
    text = clean_transcript(raw["text"])

    stats, errors, wer_value = HINT, "", "-"
    if reference and reference.strip():
        result = compute_wer(reference, text)
        wer_value = f"{100 * result.wer:.2f}"
        stats = stat_cards(100 * result.wer, result.substitutions,
                           result.deletions, result.insertions, result.ref_words)
        errors = "\n".join(error_lines(result)) or "No word-level errors. ✅"

    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_path = Path(tempfile.gettempdir()) / f"transcript-{stamp}.txt"
    out_path.write_text(text + "\n", encoding="utf-8")

    HISTORY.insert(0, [time.strftime("%d %b, %H:%M"), Path(audio_path).name, wer_value, text[:90]])
    del HISTORY[20:]

    return text, stats, errors, str(out_path), list(HISTORY)


with gr.Blocks(title="Speech Notes", theme=THEME, css=CSS) as demo:
    gr.HTML(HERO)

    with gr.Tabs():
        with gr.Tab("✨ Transcribe"):
            with gr.Row():
                with gr.Column(scale=5, elem_classes="panel"):
                    gr.HTML('<div class="sec"><i>1</i> Recording</div>')
                    audio = gr.Audio(sources=["upload", "microphone"], type="filepath",
                                     show_label=False)
                    gr.HTML('<div class="sec"><i>2</i> Language</div>')
                    language = gr.Dropdown(
                        choices=[("English", "en"), ("Auto-detect", "auto")],
                        value="en", show_label=False,
                    )
                    gr.HTML('<div class="sec"><i>3</i> Reference (optional)</div>')
                    reference = gr.Textbox(
                        lines=6, show_label=False,
                        placeholder="Paste the correct transcript to measure accuracy…",
                    )
                    run_btn = gr.Button("Transcribe ✨", elem_classes="primary-btn")

                with gr.Column(scale=6, elem_classes="panel"):
                    gr.HTML('<div class="sec"><i>✓</i> Result</div>')
                    stats = gr.HTML(EMPTY)
                    transcript = gr.Textbox(label="Transcript", lines=9, show_copy_button=True)
                    errors = gr.Textbox(label="Word-level errors", lines=4)
                    download = gr.File(label="Download .txt")

        with gr.Tab("🕘 History"):
            history = gr.Dataframe(
                headers=["Time", "Clip", "WER %", "Preview"],
                datatype=["str", "str", "str", "str"],
                value=[],
                interactive=False,
                wrap=True,
            )

    run_btn.click(
        run,
        inputs=[audio, reference, language],
        outputs=[transcript, stats, errors, download, history],
    )

if __name__ == "__main__":
    demo.launch()