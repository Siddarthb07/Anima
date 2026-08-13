"""Fetch ds002345 laptop subset via OpenNeuro GraphQL + direct file URLs (no full-tree traversal)."""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TARGET = ROOT / "data" / "ds002345"
GRAPHQL = "https://openneuro.org/crn/graphql"
REQUIRED_STORIES = ("pieman", "tunnel", "lucy")
DEFAULT_SUBJECTS = ("001", "003", "004", "005", "006")


def gql(query: str, variables: dict | None = None) -> dict:
    payload = {"query": query}
    if variables:
        payload["variables"] = variables
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        GRAPHQL,
        data=data,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        out = json.loads(resp.read().decode("utf-8"))
    if out.get("errors"):
        raise RuntimeError(out["errors"])
    return out["data"]


def latest_tag() -> str:
    data = gql("{ dataset(id: \"ds002345\") { latestSnapshot { tag } } }")
    tag = data["dataset"]["latestSnapshot"]["tag"]
    print(f"latest snapshot tag: {tag}", flush=True)
    return tag


def list_tree(tag: str, prefix: str = "") -> list[dict]:
    """List files under a path prefix using snapshot file tree API."""
    # OpenNeuro snapshot files connection — try tree query used by web UI
    query = """
    query($datasetId: ID!, $tag: String!, $prefix: String!) {
      snapshot(datasetId: $datasetId, tag: $tag) {
        files(prefix: $prefix) {
          id
          filename
          size
          directory
          urls
        }
      }
    }
    """
    try:
        data = gql(query, {"datasetId": "ds002345", "tag": tag, "prefix": prefix})
        files = data["snapshot"]["files"]
        return files or []
    except Exception as exc:
        print(f"tree query failed for prefix={prefix!r}: {exc}", flush=True)
        return []


def download_url(url: str, dest: Path, retries: int = 4) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size > 0:
        print(f"skip existing {dest.relative_to(TARGET)}", flush=True)
        return
    for attempt in range(1, retries + 1):
        try:
            print(f"GET {dest.relative_to(TARGET)} ...", flush=True)
            urllib.request.urlretrieve(url, dest)
            print(f"  -> {dest.stat().st_size} bytes", flush=True)
            return
        except Exception as exc:
            print(f"  attempt {attempt} failed: {exc}", flush=True)
            time.sleep(2 * attempt)
    raise RuntimeError(f"failed to download {url}")


def openneuro_file_url(tag: str, path: str) -> str:
    # OpenNeuro CDN-style file endpoint
    enc = path.replace("/", ":")
    return f"https://openneuro.org/crn/datasets/ds002345/snapshots/{tag}/files/{enc}"


def try_download_path(tag: str, rel: str) -> bool:
    dest = TARGET / rel
    urls = [
        openneuro_file_url(tag, rel),
        f"https://s3.amazonaws.com/openneuro.org/ds002345/{rel}",
    ]
    # Also try urls from GraphQL if available
    files = list_tree(tag, prefix="/".join(rel.split("/")[:-1]) + "/" if "/" in rel else "")
    for f in files:
        if f.get("filename") == rel.split("/")[-1] or f.get("filename") == rel:
            for u in f.get("urls") or []:
                if u:
                    urls.insert(0, u)
    for url in urls:
        try:
            download_url(url, dest)
            return True
        except Exception as exc:
            print(f"  url failed ({url[:80]}...): {exc}", flush=True)
    return False


def main() -> int:
    TARGET.mkdir(parents=True, exist_ok=True)
    tag = latest_tag()

    # 1) stimuli + metadata
    meta_paths = [
        "dataset_description.json",
        "participants.tsv",
        "README",
    ]
    for story in REQUIRED_STORIES:
        meta_paths.append(f"stimuli/{story}.txt")
        meta_paths.append(f"stimuli/{story}_words.json")

    ok_meta = 0
    for rel in meta_paths:
        if try_download_path(tag, rel):
            ok_meta += 1
    print(f"metadata/stimuli downloaded: {ok_meta}/{len(meta_paths)}", flush=True)

    # 2) Discover subject func files via tree listing
    subjects = list(DEFAULT_SUBJECTS)
    # Expand from participants if present
    part = TARGET / "participants.tsv"
    if part.is_file():
        text = part.read_text(encoding="utf-8", errors="ignore")
        print(f"participants.tsv lines: {len(text.splitlines())}", flush=True)

    bold_ok = 0
    for sub in subjects:
        prefix = f"sub-{sub}/func/"
        files = list_tree(tag, prefix)
        if not files:
            # Fallback: try common BIDS filenames directly
            for story in REQUIRED_STORIES:
                candidates = [
                    f"sub-{sub}/func/sub-{sub}_task-{story}_bold.nii.gz",
                    f"sub-{sub}/func/sub-{sub}_task-{story}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz",
                    f"sub-{sub}/func/sub-{sub}_task-{story}_run-1_bold.nii.gz",
                ]
                for rel in candidates:
                    if try_download_path(tag, rel):
                        bold_ok += 1
                        # sidecar json optional
                        try_download_path(tag, rel.replace(".nii.gz", ".json"))
                        break
            continue

        for f in files:
            name = f.get("filename") or ""
            if f.get("directory"):
                continue
            if not any(f"task-{story}" in name for story in REQUIRED_STORIES):
                continue
            if not (name.endswith(".nii.gz") or name.endswith(".json") or name.endswith(".tsv")):
                continue
            rel = f"{prefix.rstrip('/')}/{name}" if not name.startswith("sub-") else name
            # Prefer urls from API
            urls = list(f.get("urls") or [])
            urls.append(openneuro_file_url(tag, rel))
            dest = TARGET / rel
            for url in urls:
                try:
                    download_url(url, dest)
                    if name.endswith(".nii.gz"):
                        bold_ok += 1
                    break
                except Exception as exc:
                    print(f"  skip {rel}: {exc}", flush=True)

    print(f"BOLD files downloaded: {bold_ok}", flush=True)

    meta = {
        "source": "openneuro_ds002345",
        "dataset": "ds002345",
        "tag": tag,
        "stories": list(REQUIRED_STORIES),
        "subjects": subjects,
        "note": "Laptop subset via GraphQL/direct URLs for Anima brain-probe retrain",
    }
    (TARGET / "dataset_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    # Validate
    import subprocess

    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "validate_narratives_root.py"), "--root", str(TARGET)],
        cwd=str(ROOT),
    )
    return r.returncode


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        raise
