from types import SimpleNamespace

from core.intervention import should_dampen


def test_should_dampen_true_on_large_swing():
    assert should_dampen([0.1, 0.5], threshold=0.25)


def test_should_dampen_false_on_small_swing():
    assert not should_dampen([0.1, 0.2], threshold=0.25)


def test_should_dampen_needs_two_samples():
    assert not should_dampen([0.5])
