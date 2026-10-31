"""One command for the whole prototype: transcribe, evaluate, build the report.

    python run_pipeline.py
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(script: str) -> None:
    print(f"\n=== {script} ===")
    subprocess.run([sys.executable, str(ROOT / "src" / script)], check=True)


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "src"))
    from check_env import check
    if not check():
        sys.exit(1)
    run("transcribe.py")
    run("evaluate.py")
    subprocess.run([sys.executable, str(ROOT / "report" / "make_report.py")], check=True)
    print("\nFinished. See results/ and report/speech_notes_report.pdf")
