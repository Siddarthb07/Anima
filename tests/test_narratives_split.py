from probes.train import load_narratives_split


def test_load_narratives_split_matches_holdout_json():
    train, holdout = load_narratives_split()
    assert "lucy" in holdout
    assert "lucy" not in train
    assert set(train) & set(holdout) == set()
