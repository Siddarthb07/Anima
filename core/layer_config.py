LAYER_CONFIG = {
    # Tiny random GPT-2 — minimal RAM + safetensors (loads on older torch; CI / constrained Windows).
    "hf-internal-testing/tiny-random-gpt2": {
        "layers": [1, 3],
        "hidden_dim": 32,
        "has_sae": False,
    },
    "distilgpt2": {
        "layers": [2, 4],
        "hidden_dim": 768,
        "has_sae": False,
    },
    "meta-llama/Meta-Llama-3-8B": {
        "layers": [8, 16, 24, 28],
        "hidden_dim": 4096,
        "has_sae": True,
        "sae_release": "llama_scope_lxr_8x",
    },
    "meta-llama/Meta-Llama-3-8B-Instruct": {
        "layers": [8, 16, 24, 28],
        "hidden_dim": 4096,
        "has_sae": True,
        "sae_release": "llama_scope_lxr_8x",
    },
    "mistralai/Mistral-7B-v0.1": {
        "layers": [8, 16, 24],
        "hidden_dim": 4096,
        "has_sae": False,
    },
    "mistralai/Mistral-7B-Instruct-v0.2": {
        "layers": [8, 16, 24],
        "hidden_dim": 4096,
        "has_sae": False,
    },
    "Qwen/Qwen2-7B": {
        "layers": [7, 14, 22],
        "hidden_dim": 3584,
        "has_sae": False,
    },
    "Qwen/Qwen2-7B-Instruct": {
        "layers": [7, 14, 22],
        "hidden_dim": 3584,
        "has_sae": False,
    },
    "google/gemma-2-9b": {
        "layers": [9, 18, 28],
        "hidden_dim": 3584,
        "has_sae": True,
        "sae_release": "gemma-scope-9b-pt-res",
    },
    "google/gemma-2-9b-it": {
        "layers": [9, 18, 28],
        "hidden_dim": 3584,
        "has_sae": True,
        "sae_release": "gemma-scope-9b-pt-res",
    },
}
