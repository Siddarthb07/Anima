import torch

from probes.linear_probe import AffectProbe


def test_probe_shapes_and_predict():
    probe = AffectProbe(hidden_dim=768, num_layers=2)
    acts = {
        2: torch.randn(768),
        4: torch.randn(768),
    }
    out = probe(acts)
    assert set(out.keys()) == {"valence", "arousal", "uncertainty"}

    pred = probe.predict(acts)
    assert -1 <= pred["valence"] <= 1
    assert 0 <= pred["arousal"] <= 1
    assert 0 <= pred["uncertainty"] <= 1
    assert isinstance(pred["valence"], float)

    single = probe.predict(torch.randn(768))
    assert "valence" in single


def test_layer_weights_update_after_step():
    probe = AffectProbe(768, 2)
    acts = {2: torch.randn(768), 4: torch.randn(768)}
    before = probe.layer_weights.detach().clone()
    out = probe(acts)
    loss = out["valence"].sum() + out["arousal"].sum()
    loss.backward()
    torch.optim.SGD(probe.parameters(), lr=0.5).step()
    after = probe.layer_weights.detach().clone()
    assert not torch.allclose(before, after)
