from pathlib import Path

from benchmarks.manifest import _repo_relative, write_manifest


def test_repo_relative_fixture_path(tmp_path):
    root = Path(__file__).resolve().parent.parent
    inside = root / "benchmarks" / "fixtures" / "halueval_guard_sample.json"
    rel = _repo_relative(inside)
    assert rel.startswith("benchmarks/fixtures/")
    assert "\\" not in rel


def test_write_manifest_normalizes_fixture(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    reports = tmp_path / "benchmarks" / "reports"
    reports.mkdir(parents=True)

    import benchmarks.manifest as m

    monkeypatch.setattr(m, "reports_dir", lambda: reports)
    abs_fixture = str(Path(__file__).resolve())
    path = write_manifest(
        "test-model",
        [{"tier": "external_guard", "benchmark": "halueval", "fixture": abs_fixture, "status": "ok"}],
        git_sha="test",
    )
    data = path.read_text(encoding="utf-8")
    assert "tests" in data
    assert ":\\\\" not in data
