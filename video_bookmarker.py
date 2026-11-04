"""
WordMark-4-PotPlayer
영상 음성에서 지정한 단어를 감지해 팟플레이어 책갈피(.pbf)를 생성하는 도구.

- 음성 인식: faster-whisper (단어 단위 타임스탬프)
- 언어: 한국어(ko) / 영어(en) / 일본어(ja)
- 일본어는 가나·한자뿐 아니라 로마자(예: arigatou, sugoi)로도 단어 입력 가능
- 결과: 영상과 같은 폴더에 숨김 파일 "영상이름.pbf" 생성 → 팟플레이어로 열면 책갈피 표시

사용법
  GUI : python video_bookmarker.py
  CLI : python video_bookmarker.py 영상또는폴더 [...] -w 단어1,단어2 -l ko
"""
from __future__ import annotations

import argparse
import ctypes
import glob
import os
import queue
import re
import sys
import threading
import unicodedata
from dataclasses import dataclass

VIDEO_EXTS = {
    ".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv", ".webm", ".m4v",
    ".ts", ".m2ts", ".mts", ".mpg", ".mpeg", ".3gp", ".ogv", ".asf",
}
LANGUAGES = {"ko": "한국어", "en": "English", "ja": "日本語"}
MODELS = ["large-v3", "large-v3-turbo", "medium", "small", "base"]
IS_FROZEN = getattr(sys, "frozen", False)  # .exe 로 실행 중인지 여부
DEFAULT_MODEL = "small" if IS_FROZEN else "large-v3"  # .exe 는 CPU 전용이라 가벼운 모델이 기본

# 콘솔 없는 .exe 에서는 stdout/stderr 가 None 이라 진행률 출력 라이브러리가 오류를 냄
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w", encoding="utf-8")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w", encoding="utf-8")
# 한국어 Windows 콘솔(cp949)에 없는 문자를 출력해도 멈추지 않도록 함
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(errors="replace")
GAP_RECHECK_SEC = 8.0  # 이 길이 이상 단어가 없는 구간은 VAD 없이 다시 인식
NEAR_BOOKMARK_MS = 1000  # 이 간격 안의 책갈피는 가장 빠른 것 하나만 남김


# ---------------------------------------------------------------------------
# CUDA DLL 경로 (pip 로 설치한 nvidia-cublas / nvidia-cudnn 사용)
# ---------------------------------------------------------------------------
def _add_cuda_dll_dirs() -> None:
    if os.name != "nt":
        return
    for base in sys.path:
        for bin_dir in glob.glob(os.path.join(base, "nvidia", "*", "bin")):
            try:
                os.add_dll_directory(bin_dir)
                os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
            except OSError:
                pass


_add_cuda_dll_dirs()


# ---------------------------------------------------------------------------
# 데이터 구조
# ---------------------------------------------------------------------------
@dataclass
class Word:
    start: float
    end: float
    text: str


@dataclass
class Hit:
    time: float
    keyword: str
    context: str


class CharStream:
    """정규화된 문자열 + 각 문자가 속한 단어 인덱스."""

    def __init__(self) -> None:
        self.chars: list[str] = []
        self.word_idx: list[int] = []

    def add(self, s: str, idx: int) -> None:
        for c in s:
            self.chars.append(c)
            self.word_idx.append(idx)

    @property
    def text(self) -> str:
        return "".join(self.chars)


# ---------------------------------------------------------------------------
# 텍스트 정규화
# ---------------------------------------------------------------------------
def norm_surface(s: str) -> str:
    """소문자화 + 글자/숫자만 남김 (공백·문장부호 제거)."""
    s = unicodedata.normalize("NFKC", s).lower()
    return "".join(c for c in s if unicodedata.category(c)[0] in ("L", "N"))


_MACRONS = str.maketrans("āīūēôâîûêōÂÎÛÊÔĀĪŪĒŌ", "aiueoaiueoaiueoaiueo")


def romaji_canon_part(s: str) -> str:
    """헵번식/훈령식 등 표기 차이를 흡수하는 로마자 정규형 (조각 단위)."""
    s = s.lower().translate(_MACRONS)
    s = re.sub(r"[^a-z]", "", s)
    s = s.replace("tch", "tty")
    s = s.replace("sh", "sy").replace("ch", "ty").replace("ts", "t")
    s = s.replace("j", "zy").replace("f", "h")
    s = s.replace("syi", "si").replace("tyi", "ti").replace("zyi", "zi")
    s = s.replace("di", "zi").replace("du", "zu").replace("wo", "o")
    s = re.sub(r"m(?=[bpm])", "n", s)
    s = s.replace("ou", "o").replace("ei", "e")
    return s


def collapse_repeats(stream: CharStream) -> CharStream:
    """연속 중복 문자 제거 (장음·촉음·ん 표기 차이 흡수)."""
    out = CharStream()
    prev = None
    for c, i in zip(stream.chars, stream.word_idx):
        if c != prev:
            out.add(c, i)
        prev = c
    return out


def romaji_keyword_regex(romaji: str) -> str:
    s = re.sub(r"(.)\1+", r"\1", romaji_canon_part(romaji))
    # 조사 は(wa)가 사전 읽기에서 ha 로 나오는 경우 대응
    return re.escape(s).replace("wa", "(?:wa|ha)")


# 사전 기본 읽기가 일상 회화의 읽기와 다른 단어
READING_OVERRIDES = {"明日": "アシタ"}
_tokenizer = None


def japanese_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        from sudachipy import Dictionary
        _tokenizer = Dictionary(dict="core").tokenizer()
    return _tokenizer


def readings(text: str) -> list[tuple[int, str]]:
    """일본어 문장을 형태소로 나눠 [(시작 문자 위치, 로마자)] 반환."""
    import jaconv
    from sudachipy import SplitMode

    out = []
    for m in japanese_tokenizer().tokenize(text, SplitMode.C):
        if m.part_of_speech()[0] in ("空白", "補助記号"):  # 공백은 읽기가 "キゴウ"로 나옴
            continue
        kana = READING_OVERRIDES.get(m.surface(), m.reading_form() or m.surface())
        kana = jaconv.kata2hira(kana).replace("ー", "").replace("っ", "")
        out.append((m.begin(), jaconv.kana2alphabet(kana)))
    return out


def to_romaji(text: str) -> str:
    return "".join(romaji for _, romaji in readings(text))


# ---------------------------------------------------------------------------
# 단어 검색
# ---------------------------------------------------------------------------
class Matcher:
    def __init__(self, keywords: list[str], language: str) -> None:
        self.keywords = [k.strip() for k in keywords if k.strip()]
        self.language = language

    # --- 스트림 구성 ---
    @staticmethod
    def surface_stream(words: list[Word]) -> CharStream:
        st = CharStream()
        for i, w in enumerate(words):
            st.add(norm_surface(w.text), i)
        return st

    @staticmethod
    def romaji_stream(words: list[Word]) -> CharStream:
        """단어들을 이어 붙여 형태소 분석(문맥 반영)한 뒤, 로마자를 원래 단어에 매핑."""
        st = CharStream()
        chunk_size = 200  # Sudachi 입력 길이 제한(약 49KB)을 넘지 않도록 단어 묶음 단위 처리
        for base in range(0, len(words), chunk_size):
            chunk = words[base:base + chunk_size]
            owner: list[int] = []
            raw = ""
            for i, w in enumerate(chunk):
                raw += w.text
                owner.extend([base + i] * len(w.text))
            for start, romaji in readings(raw):
                st.add(romaji_canon_part(romaji), owner[start])
        return collapse_repeats(st)

    # --- 검색 ---
    def find(self, words: list[Word]) -> list[Hit]:
        if not words:
            return []
        hits: list[Hit] = []
        surface = self.surface_stream(words)
        romaji = self.romaji_stream(words) if self.language == "ja" else None

        for kw in self.keywords:
            spans: list[tuple[int, int]] = []  # (첫 단어, 끝 단어)

            nk = norm_surface(kw)
            if nk:
                for m in re.finditer(re.escape(nk), surface.text):
                    s, e = m.start(), m.end() - 1
                    if self.language == "en" and not self._en_boundary(surface, s, e):
                        continue
                    spans.append((surface.word_idx[s], surface.word_idx[e]))

            if romaji is not None:
                # 로마자 입력이든 가나/한자 입력이든 읽기(로마자)로도 비교
                rk = kw if kw.isascii() else to_romaji(kw)
                pattern = romaji_keyword_regex(rk)
                if pattern:
                    for m in re.finditer(pattern, romaji.text):
                        spans.append((romaji.word_idx[m.start()], romaji.word_idx[m.end() - 1]))

            seen_first: set[int] = set()
            for first, last in sorted(set(spans)):
                if first in seen_first:
                    continue
                seen_first.add(first)
                ctx = "".join(w.text for w in words[max(0, first - 3):last + 4]).strip()
                hits.append(Hit(words[first].start, kw, ctx))
        return hits

    @staticmethod
    def _en_boundary(st: CharStream, s: int, e: int) -> bool:
        """영어: 단어 경계에서만 일치 (복수형 s/es 정도는 허용)."""
        wi = st.word_idx
        if s > 0 and wi[s - 1] == wi[s]:
            return False
        rest = []
        j = e + 1
        while j < len(wi) and wi[j] == wi[e]:
            rest.append(st.chars[j])
            j += 1
        return "".join(rest) in ("", "s", "es")


def merge_hits(hits: list[Hit], window: float) -> list[Hit]:
    """같은 단어가 window 초 이내에 반복되면 하나로 합침."""
    hits = sorted(hits, key=lambda h: h.time)
    out: list[Hit] = []
    last: dict[str, float] = {}
    for h in hits:
        if h.keyword in last and h.time - last[h.keyword] < window:
            continue
        last[h.keyword] = h.time
        out.append(h)
    return out


# ---------------------------------------------------------------------------
# 팟플레이어 책갈피 (.pbf)
# ---------------------------------------------------------------------------
def read_pbf(path: str) -> list[tuple[int, str, str]]:
    """기존 .pbf 의 책갈피 목록 [(ms, 제목, 나머지)]"""
    if not os.path.exists(path):
        return []
    data = open(path, "rb").read()
    for enc in ("utf-16", "utf-8-sig", "cp949"):
        try:
            text = data.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        return []
    items = []
    in_section = False
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("["):
            in_section = line.lower() == "[bookmark]"
            continue
        if not in_section or "=" not in line:
            continue
        _, value = line.split("=", 1)
        parts = value.split("*", 2)
        if not parts[0].strip().isdigit():
            continue
        title = parts[1] if len(parts) > 1 else ""
        rest = parts[2] if len(parts) > 2 else ""
        items.append((int(parts[0]), title, rest))
    return items


def drop_near_bookmarks(entries: list[tuple[int, str, str]]) -> list[tuple[int, str, str]]:
    """시간순 목록에서 바로 앞 책갈피와 너무 가까운 자동 생성 책갈피를 제거.
    자동 생성 제목은 "[단어] ..." 형식이며, 사용자가 직접 만든 책갈피는 지우지 않음."""
    kept: list[tuple[int, str, str]] = []
    for e in entries:
        if kept and e[0] - kept[-1][0] < NEAR_BOOKMARK_MS and e[1].startswith("["):
            continue
        kept.append(e)
    return kept


def load_audio(path: str):
    """.exe 는 번들된 LGPL FFmpeg 로 16kHz 모노 오디오를 추출. 소스 실행은 경로를 그대로 넘겨 PyAV 가 읽음.
    PyAV 배포본에는 GPL 인 x264/x265 가 들어 있어 .exe 에 포함하지 않음."""
    if not IS_FROZEN:
        return path
    import subprocess

    import numpy as np

    ffmpeg = os.path.join(sys._MEIPASS, "ffmpeg", "ffmpeg.exe")
    cmd = [ffmpeg, "-nostdin", "-v", "error", "-i", path, "-vn", "-ac", "1", "-ar", "16000", "-f", "f32le", "-"]
    result = subprocess.run(cmd, capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
    if result.returncode != 0 or not result.stdout:
        detail = result.stderr.decode("utf-8", "replace").strip()[-300:]
        raise RuntimeError(f"오디오를 읽을 수 없습니다: {detail}")
    return np.frombuffer(result.stdout, dtype=np.float32)


def set_hidden(path: str, hidden: bool) -> None:
    """Windows 숨김 속성 설정. 숨김 파일은 open(..., "w") 가 거부되므로 저장 전에 풀어야 함."""
    if os.name != "nt":
        return
    kernel32 = ctypes.windll.kernel32
    attrs = kernel32.GetFileAttributesW(path)
    if attrs == -1:  # 파일 없음
        return
    file_attribute_hidden = 0x2
    kernel32.SetFileAttributesW(path, attrs | file_attribute_hidden if hidden else attrs & ~file_attribute_hidden)


def write_pbf(video_path: str, hits: list[Hit], preroll: float, mode: str = "merge") -> tuple[str, int]:
    """책갈피 저장. mode: merge(기존 유지+추가) / replace(덮어쓰기). 반환: (경로, 추가된 수)"""
    pbf = os.path.splitext(video_path)[0] + ".pbf"
    existing = read_pbf(pbf) if mode == "merge" else []

    entries = list(existing)
    new_entries = []
    for h in hits:
        ms = max(0, int(round((h.time - preroll) * 1000)))
        title = make_title(h).replace("*", " ")
        if any(abs(ms - e[0]) < 500 and e[1] == title for e in entries):
            continue
        entries.append((ms, title, ""))
        new_entries.append(entries[-1])
    entries.sort(key=lambda e: e[0])
    entries = drop_near_bookmarks(entries)
    added = sum(e in entries for e in new_entries)

    lines = ["[Bookmark]"]
    for i, (ms, title, rest) in enumerate(entries):
        lines.append(f"{i}={ms}*{title}*{rest}")
    lines.append(f"{len(entries)}=")
    set_hidden(pbf, False)
    try:
        with open(pbf, "w", encoding="utf-16", newline="\r\n") as f:
            f.write("\n".join(lines) + "\n")
    finally:
        set_hidden(pbf, True)
    return pbf, added


def make_title(h: Hit) -> str:
    ctx = re.sub(r"\s+", " ", h.context)
    if len(ctx) > 40:
        ctx = ctx[:40] + "…"
    return f"[{h.keyword}] {ctx}"


def fmt_time(sec: float) -> str:
    sec = max(0, int(sec))
    return f"{sec // 3600:02d}:{sec % 3600 // 60:02d}:{sec % 60:02d}"


# ---------------------------------------------------------------------------
# 음성 인식
# ---------------------------------------------------------------------------
class Transcriber:
    def __init__(self, model_name: str, device: str, log) -> None:
        self.model_name = model_name
        self.device = device
        self.log = log
        self._model = None

    def model(self):
        if self._model is not None:
            return self._model
        from faster_whisper import WhisperModel
        tried = []
        if self.device in ("auto", "cuda"):
            tried.append(("cuda", "float16"))
        if self.device in ("auto", "cpu"):
            tried.append(("cpu", "int8"))
        last_err = None
        for dev, ct in tried:
            try:
                self.log(f"모델 로드 중: {self.model_name} ({dev}, {ct}) — 처음엔 다운로드로 시간이 걸릴 수 있습니다.")
                self._model = WhisperModel(self.model_name, device=dev, compute_type=ct)
                self.device = dev
                return self._model
            except Exception as e:  # noqa: BLE001
                last_err = e
                self.log(f"  {dev} 사용 불가: {e}")
        raise RuntimeError(f"모델을 불러올 수 없습니다: {last_err}")

    def transcribe(self, path: str, language: str, hotwords: str | None, progress, stop: threading.Event) -> list[Word]:
        try:
            return self._transcribe(path, language, hotwords, progress, stop)
        except RuntimeError as e:
            # CUDA 라이브러리 문제는 실제 추론 시점에 드러나는 경우가 있어 CPU 로 재시도
            if self.device == "cuda" and any(k in str(e).lower() for k in ("cublas", "cudnn", "cuda")):
                self.log(f"GPU 오류로 CPU 로 전환합니다: {e}")
                self._model, self.device = None, "cpu"
                return self._transcribe(path, language, hotwords, progress, stop)
            raise

    def _transcribe(self, path, language, hotwords, progress, stop) -> list[Word]:
        common = dict(
            language=language,
            word_timestamps=True,
            beam_size=5,
            hotwords=hotwords or None,
            condition_on_previous_text=False,
        )
        audio = load_audio(path)
        # 1차: VAD 로 음성 구간만 빠르게 인식
        segments, info = self.model().transcribe(audio, vad_filter=True, **common)
        duration = info.duration or 0
        words = self._collect(segments, duration, progress, stop)

        # 2차: 영상 끝까지 빠짐없이 검사 — 단어가 하나도 없는 구간(앞·중간·끝)을 VAD 없이 다시 인식
        gaps = self.find_gaps(words, duration)
        if gaps:
            total = sum(e - s for s, e in gaps)
            self.log(f"  인식 안 된 구간 {len(gaps)}곳({fmt_time(total)})을 다시 검사합니다...")
            clip = [t for g in gaps for t in g]
            segments, _ = self.model().transcribe(audio, vad_filter=False, clip_timestamps=clip, **common)
            extra = [
                w for w in self._collect(segments, duration, progress, stop)
                if any(s - 1.0 <= w.start and w.end <= e + 1.0 for s, e in gaps)
            ]
            if extra:
                self.log(f"  추가로 인식된 단어 {len(extra)}개")
                words = sorted(words + extra, key=lambda w: w.start)

        if duration:
            progress(1.0, duration, duration)
        self.log(f"  영상 끝까지 검사 완료 ({fmt_time(duration)})")
        return words

    @staticmethod
    def _collect(segments, duration, progress, stop) -> list[Word]:
        words: list[Word] = []
        for seg in segments:  # 제너레이터를 끝까지 소비해야 영상 끝까지 인식됨
            if stop.is_set():
                raise InterruptedError
            for w in seg.words or []:
                words.append(Word(w.start, w.end, w.word))
            if duration:
                progress(min(seg.end / duration, 1.0), seg.end, duration)
        return words

    @staticmethod
    def find_gaps(words: list[Word], duration: float, min_gap: float = GAP_RECHECK_SEC) -> list[tuple[float, float]]:
        """단어가 없는 구간 [(시작, 끝)] — 영상 처음부터 맨 끝(duration)까지."""
        gaps = []
        cur = 0.0
        for w in sorted(words, key=lambda w: w.start):
            if w.start - cur >= min_gap:
                gaps.append((cur, w.start))
            cur = max(cur, w.end)
        if duration - cur >= min(min_gap, 1.0) and duration > cur:
            gaps.append((cur, duration))  # 마지막 단어 이후 ~ 영상 끝은 짧아도 항상 재검사
        return [(round(s, 2), round(e, 2)) for s, e in gaps]


# ---------------------------------------------------------------------------
# 처리 파이프라인
# ---------------------------------------------------------------------------
@dataclass
class Options:
    keywords: list[str]
    language: str = "ko"
    model: str = "large-v3"
    device: str = "auto"
    preroll: float = 1.0
    merge_window: float = 0.0
    mode: str = "merge"
    save_transcript: bool = False


def collect_videos(paths: list[str]) -> list[str]:
    out = []
    for p in paths:
        if os.path.isdir(p):
            for root, _, files in os.walk(p):
                for f in sorted(files):
                    if os.path.splitext(f)[1].lower() in VIDEO_EXTS:
                        out.append(os.path.join(root, f))
        elif os.path.isfile(p) and os.path.splitext(p)[1].lower() in VIDEO_EXTS:
            out.append(p)
    seen = set()
    return [p for p in out if not (p in seen or seen.add(p))]


def process(videos: list[str], opt: Options, log, progress, stop: threading.Event) -> None:
    matcher = Matcher(opt.keywords, opt.language)
    tr = Transcriber(opt.model, opt.device, log)
    # 로마자 키워드는 인식 힌트로 쓰면 오히려 방해되므로 제외
    hint_words = [k for k in matcher.keywords if not (opt.language == "ja" and k.isascii())]
    hotwords = " ".join(hint_words)

    for n, path in enumerate(videos, 1):
        if stop.is_set():
            log("중지되었습니다.")
            return
        log(f"\n[{n}/{len(videos)}] {os.path.basename(path)}")
        try:
            words = tr.transcribe(
                path, opt.language, hotwords,
                lambda r, t, d: progress(n, len(videos), r, t, d), stop,
            )
        except InterruptedError:
            log("중지되었습니다.")
            return
        except Exception as e:  # noqa: BLE001
            log(f"  오류: {e}")
            continue

        if opt.save_transcript:
            txt = os.path.splitext(path)[0] + ".transcript.txt"
            with open(txt, "w", encoding="utf-8") as f:
                for w in words:
                    f.write(f"{fmt_time(w.start)}\t{w.text.strip()}\n")
            log(f"  인식 결과 저장: {txt}")

        hits = merge_hits(matcher.find(words), opt.merge_window)
        if not hits:
            log("  찾은 단어 없음")
            continue
        for h in hits:
            log(f"  {fmt_time(h.time)}  [{h.keyword}]  {h.context}")
        pbf, added = write_pbf(path, hits, opt.preroll, opt.mode)
        log(f"  → 책갈피 {added}개 추가: {pbf}")
    log("\n완료!")


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------
def run_gui() -> None:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    root = tk.Tk()
    root.title("WordMark-4-PotPlayer")
    root.geometry("780x620")

    msgs: queue.Queue = queue.Queue()
    stop = threading.Event()
    files: list[str] = []

    frm = ttk.Frame(root, padding=10)
    frm.pack(fill="both", expand=True)

    # 파일 목록
    ttk.Label(frm, text="영상 파일 / 폴더").grid(row=0, column=0, sticky="w")
    lst = tk.Listbox(frm, height=6, selectmode="extended")
    lst.grid(row=1, column=0, columnspan=4, sticky="nsew")
    btns = ttk.Frame(frm)
    btns.grid(row=2, column=0, columnspan=4, sticky="w", pady=(4, 8))

    def refresh():
        lst.delete(0, "end")
        for f in files:
            lst.insert("end", f)

    def add_files():
        sel = filedialog.askopenfilenames(
            title="영상 선택",
            filetypes=[("영상 파일", " ".join("*" + e for e in sorted(VIDEO_EXTS))), ("모든 파일", "*.*")],
        )
        for f in collect_videos(list(sel)):
            if f not in files:
                files.append(f)
        refresh()

    def add_folder():
        d = filedialog.askdirectory(title="폴더 선택 (하위 폴더 포함)")
        if d:
            for f in collect_videos([d]):
                if f not in files:
                    files.append(f)
            refresh()

    def remove_sel():
        for i in reversed(lst.curselection()):
            files.pop(i)
        refresh()

    ttk.Button(btns, text="파일 추가", command=add_files).pack(side="left")
    ttk.Button(btns, text="폴더 추가", command=add_folder).pack(side="left", padx=4)
    ttk.Button(btns, text="선택 삭제", command=remove_sel).pack(side="left")
    ttk.Button(btns, text="모두 삭제", command=lambda: (files.clear(), refresh())).pack(side="left", padx=4)

    # 옵션
    ttk.Label(frm, text="찾을 단어 (쉼표로 구분)").grid(row=3, column=0, sticky="w")
    kw_var = tk.StringVar()
    ttk.Entry(frm, textvariable=kw_var).grid(row=3, column=1, columnspan=3, sticky="ew")

    ttk.Label(frm, text="단어 언어").grid(row=4, column=0, sticky="w", pady=4)
    lang_var = tk.StringVar(value="ko")
    lf = ttk.Frame(frm)
    lf.grid(row=4, column=1, columnspan=3, sticky="w")
    for code, name in LANGUAGES.items():
        ttk.Radiobutton(lf, text=name, value=code, variable=lang_var).pack(side="left", padx=(0, 10))
    hint = ttk.Label(frm, text="", foreground="gray")
    hint.grid(row=5, column=1, columnspan=3, sticky="w")

    def on_lang(*_):
        hint.config(text="일본어는 로마자 입력 가능 (예: arigatou, sugoi, kawaii)" if lang_var.get() == "ja" else "")
    lang_var.trace_add("write", on_lang)

    of = ttk.Frame(frm)
    of.grid(row=6, column=0, columnspan=4, sticky="w", pady=6)
    ttk.Label(of, text="모델").pack(side="left")
    model_var = tk.StringVar(value=DEFAULT_MODEL)
    ttk.Combobox(of, textvariable=model_var, values=MODELS, width=15).pack(side="left", padx=(4, 12))
    ttk.Label(of, text="장치").pack(side="left")
    dev_var = tk.StringVar(value="cpu" if IS_FROZEN else "auto")  # .exe 는 CPU 전용
    ttk.Combobox(of, textvariable=dev_var, values=["auto", "cuda", "cpu"], width=6, state="readonly").pack(side="left", padx=(4, 12))
    ttk.Label(of, text="앞당김(초)").pack(side="left")
    pre_var = tk.DoubleVar(value=1.0)
    ttk.Spinbox(of, from_=0, to=10, increment=0.5, textvariable=pre_var, width=5).pack(side="left", padx=(4, 12))
    ttk.Label(of, text="중복 병합(초)").pack(side="left")
    win_var = tk.DoubleVar(value=0.0)
    ttk.Spinbox(of, from_=0, to=60, increment=1, textvariable=win_var, width=5).pack(side="left", padx=4)

    of2 = ttk.Frame(frm)
    of2.grid(row=7, column=0, columnspan=4, sticky="w")
    replace_var = tk.BooleanVar(value=False)
    ttk.Checkbutton(of2, text="기존 책갈피 덮어쓰기 (해제 시 기존 책갈피 유지 + 추가)", variable=replace_var).pack(side="left")
    tr_var = tk.BooleanVar(value=False)
    ttk.Checkbutton(of2, text="인식 텍스트 저장(.transcript.txt)", variable=tr_var).pack(side="left", padx=12)

    # 실행
    pf = ttk.Frame(frm)
    pf.grid(row=8, column=0, columnspan=4, sticky="ew", pady=8)
    run_btn = ttk.Button(pf, text="시작")
    run_btn.pack(side="left")
    stop_btn = ttk.Button(pf, text="중지", state="disabled", command=stop.set)
    stop_btn.pack(side="left", padx=4)
    pbar = ttk.Progressbar(pf, maximum=1.0)
    pbar.pack(side="left", fill="x", expand=True, padx=8)
    plabel = ttk.Label(pf, text="", width=24)
    plabel.pack(side="left")

    logbox = tk.Text(frm, height=14, wrap="word")
    logbox.grid(row=9, column=0, columnspan=4, sticky="nsew")

    frm.columnconfigure(1, weight=1)
    frm.rowconfigure(1, weight=1)
    frm.rowconfigure(9, weight=2)

    def start():
        kws = [k for k in re.split(r"[,，、\n]", kw_var.get()) if k.strip()]
        if not files:
            messagebox.showwarning("알림", "영상 파일을 추가하세요.")
            return
        if not kws:
            messagebox.showwarning("알림", "찾을 단어를 입력하세요.")
            return
        opt = Options(
            keywords=kws, language=lang_var.get(), model=model_var.get(), device=dev_var.get(),
            preroll=float(pre_var.get()), merge_window=float(win_var.get()),
            mode="replace" if replace_var.get() else "merge", save_transcript=tr_var.get(),
        )
        stop.clear()
        run_btn.config(state="disabled")
        stop_btn.config(state="normal")

        def worker():
            try:
                process(
                    list(files), opt,
                    lambda m: msgs.put(("log", m)),
                    lambda n, total, r, t, d: msgs.put(("prog", (n, total, r, t, d))),
                    stop,
                )
            except Exception as e:  # noqa: BLE001
                msgs.put(("log", f"오류: {e}"))
            msgs.put(("done", None))

        threading.Thread(target=worker, daemon=True).start()

    run_btn.config(command=start)

    def poll():
        try:
            while True:
                kind, data = msgs.get_nowait()
                if kind == "log":
                    logbox.insert("end", data + "\n")
                    logbox.see("end")
                elif kind == "prog":
                    n, total, r, t, d = data
                    pbar["value"] = r
                    plabel.config(text=f"{n}/{total}  {fmt_time(t)} / {fmt_time(d)}")
                elif kind == "done":
                    run_btn.config(state="normal")
                    stop_btn.config(state="disabled")
        except queue.Empty:
            pass
        root.after(100, poll)

    poll()
    root.mainloop()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def run_cli(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(description="영상에서 단어를 찾아 팟플레이어 책갈피(.pbf) 생성")
    ap.add_argument("paths", nargs="+", help="영상 파일 또는 폴더")
    ap.add_argument("-w", "--words", required=True, help="찾을 단어 (쉼표 구분)")
    ap.add_argument("-l", "--lang", choices=list(LANGUAGES), default="ko", help="단어 언어 (ko/en/ja)")
    ap.add_argument("-m", "--model", default="large-v3", help="Whisper 모델 (기본 large-v3)")
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    ap.add_argument("--preroll", type=float, default=1.0, help="책갈피를 몇 초 앞당길지 (기본 1.0)")
    ap.add_argument("--merge", type=float, default=0.0, help="같은 단어 반복 병합 간격(초, 0=병합 안 함)")
    ap.add_argument("--replace", action="store_true", help="기존 .pbf 책갈피를 덮어씀")
    ap.add_argument("--transcript", action="store_true", help="인식 텍스트를 .transcript.txt 로 저장")
    a = ap.parse_args(argv)

    videos = collect_videos(a.paths)
    if not videos:
        sys.exit("영상 파일을 찾지 못했습니다.")
    opt = Options(
        keywords=[k for k in re.split(r"[,，、]", a.words) if k.strip()],
        language=a.lang, model=a.model, device=a.device, preroll=a.preroll,
        merge_window=a.merge, mode="replace" if a.replace else "merge", save_transcript=a.transcript,
    )

    def prog(n, total, r, t, d):
        print(f"\r  진행 {r * 100:5.1f}%  {fmt_time(t)} / {fmt_time(d)}", end="", flush=True)

    def log(m):
        print("\r" + " " * 50 + "\r" + m, flush=True)

    process(videos, opt, log, prog, threading.Event())


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_cli(sys.argv[1:])
    else:
        run_gui()
