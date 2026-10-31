"""Light formatting clean-up for raw Whisper output.

Whisper already returns punctuated text; this step only tidies spacing,
capitalisation and line breaks so the transcript reads like notes.
"""
import re

ABBREVIATIONS = ["mr", "mrs", "ms", "dr", "st", "jr", "sr", "prof"]

def clean_transcript(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ""
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)      # no space before punctuation
    for abbr in ABBREVIATIONS:                        # protect "Mr." etc. from splitting
        text = re.sub(rf"\b({abbr})\.", lambda m: m.group(1) + "\x00", text, flags=re.I)
    text = re.sub(r"([.!?])\s*(?=\S)", r"\1\n", text) # one sentence per line
    text = text.replace("\x00", ".")
    lines = []
    for line in text.split("\n"):
        line = line.strip()
        if line:
            lines.append(line[0].upper() + line[1:])
    result = "\n".join(lines)
    if result and result[-1] not in ".!?":
        result += "."
    return result
