import os

import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_HF_TESTS") != "1",
    reason="HF model tests: set RUN_HF_TESTS=1 (downloads weights; may require GPU/stable torch on Windows).",
)

from core.hooks import ActivationHook
from transformers import AutoModelForCausalLM

def test_hooks_distilgpt2_capture_and_clear():
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    model.eval()
    hook = ActivationHook(model, [2, 4])

    input_ids = torch.ones(1, 3, dtype=torch.long)
    with torch.no_grad():
        hook.clear()
        model(input_ids)

    assert set(hook.buffer.keys()) == {2, 4}
    assert hook.last_token(2).shape == torch.Size([768])

    hook.clear()
    assert hook.buffer == {}

    hook.remove()

    with torch.no_grad():
        hook.clear()
        model(input_ids)

    assert hook.buffer == {}
