"""Make sure an ffmpeg program is available to Whisper.

Whisper calls a program named `ffmpeg`. Two ways to get it:
  1. ffmpeg is already installed on the system (checked first), or
  2. the `imageio-ffmpeg` pip package, listed in requirements.txt, bundles
     an ffmpeg binary. We copy it to .tools/ffmpeg (ffmpeg.exe on Windows)
     and add that folder to PATH for this run.

So `pip install -r requirements.txt` is enough on most machines.
"""
import os
import shutil
import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parents[1] / ".tools"


def _use_bundled_ffmpeg() -> bool:
    try:
        import imageio_ffmpeg
    except ImportError:
        return False
    source = Path(imageio_ffmpeg.get_ffmpeg_exe())
    TOOLS_DIR.mkdir(exist_ok=True)
    target = TOOLS_DIR / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")
    if not target.exists():
        shutil.copy2(source, target)
        if os.name != "nt":
            target.chmod(0o755)
    os.environ["PATH"] = str(TOOLS_DIR) + os.pathsep + os.environ.get("PATH", "")
    return shutil.which("ffmpeg") is not None


def check() -> bool:
    if shutil.which("ffmpeg") is not None:
        return True
    if _use_bundled_ffmpeg():
        print("Using ffmpeg bundled with the imageio-ffmpeg package.")
        return True
    print("""
ffmpeg was not found. Install the requirements again (pip install -r requirements.txt),
or install ffmpeg yourself:
  Windows : winget install Gyan.FFmpeg
  macOS   : brew install ffmpeg
  Ubuntu  : sudo apt install ffmpeg
  Colab   : !apt install -y ffmpeg
""")
    return False


if __name__ == "__main__":
    sys.exit(0 if check() else 1)
