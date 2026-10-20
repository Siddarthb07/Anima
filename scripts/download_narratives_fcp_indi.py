"""
Download a laptop-sized Narratives subset from FCP-INDI public S3 and reshape
it into the BIDS layout expected by alignment/narratives_loader.py.

- stimuli/{story}.txt + {story}_words.json from gentle transcripts/align.json
- sub-*/func/*task-{story}_bold.nii.gz for a few multi-story subjects
- skips large audio wavs

Usage:
  python scripts/download_narratives_fcp_indi.py
"""

from __future__ import annotations

import json
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TARGET = ROOT / "data" / "ds002345"
BUCKET = "fcp-indi"
PREFIX = "data/Projects/narratives/"
STORIES = ("pieman", "tunnel", "lucy")
SUBJECTS = ("022", "026", "041")


def _local(tag: str) -> str:
    return tag.split("}", 1)[-1] if "}" in tag else tag


def list_keys(prefix: str, max_keys: int = 1000) -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []
    token = None
    while True:
        params = {"list-type": "2", "prefix": prefix, "max-keys": str(max_keys)}
        if token:
            params["continuation-token"] = token
        url = f"https://{BUCKET}.s3.amazonaws.com/?{urllib.parse.urlencode(params)}"
        with urllib.request.urlopen(url, timeout=120) as resp:
            root = ET.fromstring(resp.read())
        truncated = "false"
        token = None
        for el in root:
            name = _local(el.tag)
            if name == "Contents":
                key = None
                size = 0
                for c in el:
                    cn = _local(c.tag)
                    if cn == "Key":
                        key = c.text
                    elif cn == "Size":
                        size = int(c.text or 0)
                if key:
                    out.append((key, size))
            elif name == "IsTruncated":
                truncated = (el.text or "false").lower()
            elif name == "NextContinuationToken":
                token = el.text
        if truncated != "true" or not token:
            break
    return out


def download_key(key: str, dest: Path) -> None:
    import subprocess

    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size > 1_000_000:
        print(f"skip {dest.relative_to(TARGET)} ({dest.stat().st_size} B)", flush=True)
        return
    # drop incomplete stubs
    if dest.is_file() and dest.stat().st_size < 1_000_000:
        dest.unlink(missing_ok=True)
    url = f"https://{BUCKET}.s3.amazonaws.com/{urllib.parse.quote(key)}"
    print(f"GET {dest.relative_to(TARGET)}", flush=True)
    # curl is more reliable than urlretrieve on long Windows transfers
    tmp = dest.with_suffix(dest.suffix + ".part")
    cmd = [
        "curl",
        "-L",
        "--fail",
        "--retry",
        "5",
        "--retry-delay",
        "2",
        "--connect-timeout",
        "30",
        "--max-time",
        "1800",
        "-o",
        str(tmp),
        url,
    ]
    try:
        subprocess.run(cmd, check=True)
        tmp.replace(dest)
    except (FileNotFoundError, subprocess.CalledProcessError):
        # fallback: streaming urlopen with timeout
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=120) as resp, open(tmp, "wb") as f:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
        tmp.replace(dest)
    print(f"  -> {dest.stat().st_size} bytes", flush=True)


def build_stimuli_from_gentle() -> None:
    stim_dir = TARGET / "stimuli"
    stim_dir.mkdir(parents=True, exist_ok=True)
    for story in STORIES:
        base = f"{PREFIX}stimuli/gentle/{story}/"
        keys = dict(list_keys(base))
        tkey = base + "transcript.txt"
        akey = base + "align.json"
        if tkey not in keys or akey not in keys:
            raise FileNotFoundError(f"missing gentle stimuli for {story}")
        raw_t = TARGET / "_tmp" / f"{story}_transcript.txt"
        raw_a = TARGET / "_tmp" / f"{story}_align.json"
        download_key(tkey, raw_t)
        download_key(akey, raw_a)
        text = raw_t.read_text(encoding="utf-8", errors="ignore").strip()
        (stim_dir / f"{story}.txt").write_text(text + "\n", encoding="utf-8")
        align = json.loads(raw_a.read_text(encoding="utf-8"))
        words = []
        # gentle align.json is typically a list of {word, start, end} or nested
        entries = align if isinstance(align, list) else align.get("words") or align.get("segments") or []
        for e in entries:
            if not isinstance(e, dict):
                continue
            w = e.get("word") or e.get("alignedWord") or e.get("text")
            onset = e.get("onset_sec")
            if onset is None:
                onset = e.get("start") or e.get("start_time") or e.get("begin")
            if w is None or onset is None:
                continue
            words.append({"word": str(w), "onset_sec": float(onset)})
        if not words:
            # fallback: evenly space transcript tokens
            toks = text.split()
            words = [{"word": t, "onset_sec": round(i * 0.35, 3)} for i, t in enumerate(toks)]
            print(f"  warn: {story} align parse empty — synthetic onsets from transcript", flush=True)
        (stim_dir / f"{story}_words.json").write_text(json.dumps(words, indent=2) + "\n", encoding="utf-8")
        print(f"wrote stimuli/{story}.txt + {story}_words.json (n_words={len(words)})", flush=True)


def download_bold() -> int:
    n = 0
    for sub in SUBJECTS:
        for story in STORIES:
            pref = f"{PREFIX}sub-{sub}/func/"
            for key, size in list_keys(pref):
                if f"task-{story}" not in key:
                    continue
                if not (key.endswith("_bold.nii.gz") or key.endswith("_bold.json") or key.endswith("_events.tsv")):
                    continue
                # skip tiny annex pointers
                if key.endswith(".nii.gz") and size < 1_000_000:
                    print(f"skip tiny/annex {key} size={size}", flush=True)
                    continue
                rel = key[len(PREFIX) :]
                download_key(key, TARGET / rel)
                if key.endswith(".nii.gz"):
                    n += 1
    return n


def main() -> int:
    TARGET.mkdir(parents=True, exist_ok=True)
    # keep previously downloaded OpenNeuro metadata if present
    print("Building stimuli from FCP-INDI gentle transcripts...", flush=True)
    build_stimuli_from_gentle()
    print("Downloading BOLD for subjects", SUBJECTS, flush=True)
    n_bold = download_bold()
    print(f"BOLD nifti count: {n_bold}", flush=True)

    meta = {
        "source": "fcp_indi_narratives",
        "dataset": "ds002345",
        "stories": list(STORIES),
        "subjects": list(SUBJECTS),
        "note": "Subset from s3://fcp-indi/data/Projects/narratives/ (gentle stimuli + raw BOLD)",
    }
    (TARGET / "dataset_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    # cleanup tmp
    tmp = TARGET / "_tmp"
    if tmp.is_dir():
        import shutil

        shutil.rmtree(tmp, ignore_errors=True)

    import subprocess

    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "validate_narratives_root.py"), "--root", str(TARGET)],
        cwd=str(ROOT),
    )
    if r.returncode == 0:
        env_path = ROOT / ".env.benchmark"
        line = f"NARRATIVES_ROOT={TARGET.as_posix()}"
        if env_path.exists():
            text = env_path.read_text(encoding="utf-8")
            if "NARRATIVES_ROOT=" in text:
                lines = [ln if not ln.startswith("NARRATIVES_ROOT=") else line for ln in text.splitlines()]
                env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            else:
                env_path.write_text(text.rstrip() + "\n" + line + "\n", encoding="utf-8")
        print(f"Set NARRATIVES_ROOT={TARGET.resolve()}", flush=True)
    return r.returncode


if __name__ == "__main__":
    raise SystemExit(main())
