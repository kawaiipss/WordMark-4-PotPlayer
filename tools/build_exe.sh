#!/bin/sh
# Build the CPU-only release zip. Run from the project root with Git Bash.
#
#   tools/build_exe.sh <work dir> <FFmpeg LGPL shared build dir> <version>
#
# <work dir> holds a venv with faster-whisper sudachipy sudachidict-core jaconv
# pyinstaller installed and PyAV (av) UNINSTALLED. Its wheels contain GPL
# x264/x265, so the .exe uses the LGPL FFmpeg build with a placeholder av module.
set -e
WORK=$1
FF=$2
VERSION=$3
PY="$WORK/buildenv/Scripts/python"
APP=WordMark-4-PotPlayer
OUT="$WORK/build/dist/$APP"

mkdir -p "$WORK/stubs"
echo '"""Placeholder so faster_whisper imports without PyAV."""' > "$WORK/stubs/av.py"

PYTHONDONTWRITEBYTECODE=1 "$PY" -m PyInstaller --noconfirm --clean --windowed --name "$APP" \
  --paths "$WORK/stubs" --hidden-import av \
  --collect-all faster_whisper --collect-all ctranslate2 --collect-all sudachipy \
  --collect-all sudachidict_core --collect-data jaconv \
  --distpath "$WORK/build/dist" --workpath "$WORK/build/work" --specpath "$WORK/build" \
  video_bookmarker.py

# PyInstaller bundles Python's older runtime, which crashes CTranslate2 (std::mutex).
for f in msvcp140.dll vcruntime140.dll vcruntime140_1.dll; do
  cp "/c/Windows/System32/$f" "$OUT/_internal/$f"
done

mkdir -p "$OUT/_internal/ffmpeg"
cp "$FF"/bin/ffmpeg.exe "$FF"/bin/*.dll "$OUT/_internal/ffmpeg/"

cp LICENSE "$OUT/LICENSE.txt"
"$PY" tools/make_notices.py "$OUT/THIRD-PARTY-NOTICES.txt" "$FF/LICENSE.txt" \
  "$("$FF/bin/ffmpeg.exe" -hide_banner -version | head -1 | cut -d' ' -f3)"

mkdir -p dist
"$PY" -c "import shutil,sys; shutil.make_archive(sys.argv[1], 'zip', sys.argv[2], sys.argv[3])" \
  "dist/$APP-$VERSION-win64" "$WORK/build/dist" "$APP"
ls -la "dist/$APP-$VERSION-win64.zip"
