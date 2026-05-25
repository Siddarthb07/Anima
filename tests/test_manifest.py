"""Benchmark manifest paths and schema."""

from benchmarks.manifest import MANIFEST_SCHEMA_VERSION, _probe_slug, write_manifest


def test_write_manifest_updates_latest_aliases(tmp_path, monkeypatch):
    import benchmarks.manifest as m

    monkeypatch.setattr(m, "reports_dir", lambda: tmp_path)
    entries = [{"tier": "internal", "benchmark": "smoke", "status": "ok"}]
    write_manifest("distilgpt2", entries, git_sha="abc")
    assert (tmp_path / "latest_manifest.json").is_file()
    assert (tmp_path / "latest_distilgpt2_manifest.json").is_file()
    payload = (tmp_path / "latest_distilgpt2_manifest.json").read_text(encoding="utf-8")
    assert "manifest_schema_version" in payload
    assert str(MANIFEST_SCHEMA_VERSION) in payload


def test_probe_slug():
    assert _probe_slug("hf-internal-testing/tiny-random-gpt2") == "tiny_random_gpt2"
    assert _probe_slug("distilgpt2") == "distilgpt2"
