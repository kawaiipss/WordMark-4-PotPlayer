"""Write THIRD-PARTY-NOTICES.txt for the .exe. Run with the build environment's Python.

Usage: make_notices.py <output file> <FFmpeg LICENSE.txt> <FFmpeg version string>
"""
import os
import sys
from importlib import metadata

BUILD_ONLY = {"altgraph", "pefile", "pip", "pyinstaller", "pyinstaller-hooks-contrib",
              "pywin32-ctypes", "setuptools"}
LICENSE_HINTS = ("license", "licence", "copying", "notice", "authors")

HEADER = """WordMark-4-PotPlayer includes the third-party components listed below. Each
component remains under its own license. The restrictions in this program's
LICENSE file do not apply to them.
"""

FFMPEG = """FFmpeg {version}
License: GNU Lesser General Public License, version 3 or later (LGPL-3.0-or-later)
Files: _internal/ffmpeg/
Binary build: https://github.com/BtbN/FFmpeg-Builds
Source code: https://github.com/FFmpeg/FFmpeg and https://github.com/BtbN/FFmpeg-Builds
FFmpeg runs as a separate program and its DLLs can be replaced by the user.

{text}
"""

OTHERS = """Python
License: Python Software Foundation License. See https://docs.python.org/3/license.html

Tcl/Tk (GUI)
License: Tcl/Tk License (BSD style). See https://www.tcl.tk/software/tcltk/license.html

Microsoft Visual C++ Runtime (msvcp140.dll, vcruntime140.dll, vcruntime140_1.dll)
License: Microsoft Visual C++ Redistributable license terms.

Intel OpenMP Runtime (libiomp5md.dll, included with CTranslate2)
Redistributed as part of the CTranslate2 Python package.

PyInstaller bootloader
License: GPL-2.0-or-later with an exception permitting distribution of the
bootloader in generated executables under any terms.
See https://github.com/pyinstaller/pyinstaller/blob/develop/COPYING.txt

Whisper speech recognition models are not included. They are downloaded on first
use from Hugging Face and remain under their own licenses.
"""


def license_texts(dist):
    out = []
    for f in dist.files or []:
        name = f.name.lower()
        if any(h in name for h in LICENSE_HINTS) and not name.endswith((".py", ".pyc")):
            try:
                with open(f.locate(), encoding="utf-8", errors="replace") as fh:
                    out.append(f"--- {f.name} ---\n{fh.read().strip()}\n")
            except OSError:
                pass
    return out


def main() -> None:
    output, ffmpeg_license, ffmpeg_version = sys.argv[1:4]
    with open(ffmpeg_license, encoding="utf-8", errors="replace") as f:
        parts = [HEADER, FFMPEG.format(version=ffmpeg_version, text=f.read().strip()), OTHERS]
    missing = []
    for dist in sorted(metadata.distributions(), key=lambda d: d.metadata["Name"].lower()):
        meta = dist.metadata
        name = meta["Name"]
        if name.lower() in BUILD_ONLY:
            continue
        lic = meta.get("License-Expression") or (meta.get("License") or "").split("\n")[0]
        classifiers = [c.split("::")[-1].strip() for c in meta.get_all("Classifier", []) if "License" in c]
        texts = license_texts(dist)
        extra = os.path.join(os.path.dirname(__file__), "extra_licenses", name.lower() + ".txt")
        if not texts and os.path.exists(extra):  # 패키지에 라이선스 전문이 없는 경우
            with open(extra, encoding="utf-8") as fh:
                texts = [fh.read().strip() + "\n"]
        if not texts:
            missing.append(name)
        parts.append(f"{name} {meta['Version']}\nLicense: {lic or ', '.join(classifiers)}\n"
                     f"Project: {meta.get('Home-page') or meta.get('Project-URL') or ''}\n\n" + "\n".join(texts))
    sep = "\n" + "=" * 78 + "\n\n"
    with open(output, "w", encoding="utf-8", newline="\r\n") as f:
        f.write(sep.join(parts))
    print("packages without license files:", missing)


if __name__ == "__main__":
    main()
