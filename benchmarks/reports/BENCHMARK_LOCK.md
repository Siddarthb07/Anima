# Anima eval freeze (TMLR / AICS)

Generated: `2026-10-23T11:36:10.330841+00:00`

IEEE honesty: cite Protocol L live AUROC + CIs; fixture Protocol F is smoke only;
Protocol R is optimistic secondary; Protocol R' is held-out primary for thresholds.

| File | Bytes | SHA-256 |
|------|------:|---------|
| `halueval_live_distilgpt2.json` | 81761 | `5640148e2caff1441af0c57f2a55abb6d9c8dba69f65c2bbc380bcf353d991f6` |
| `halueval_live_qwen_qwen2_5-0_5b-instruct.json` | 87439 | `936e830c32897ddc71f87bbc25d87b56c1f48e92b7e890d8be934299dbbce7dc` |
| `halueval_live_tinyllama_tinyllama-1_1b-chat-v1_0.json` | 83191 | `f68c956c30b4e9c680b20031e3e62f327256f5d322cbe8fab373ecaf873225c6` |
| `truthfulqa_live_distilgpt2.json` | 77867 | `0e655b84234e32f6d90c310c72df5b4cd10d471a4096b8e136e69cef220795e0` |
| `truthfulqa_live_qwen_qwen2_5-0_5b-instruct.json` | 82944 | `295d6c0a297b1d62a245b1a337f7af77cdb92919380ff016c57174384e4786b9` |
| `truthfulqa_live_tinyllama_tinyllama-1_1b-chat-v1_0.json` | 78807 | `661877dc8f1a585838669791ac8659bc6067a2997e290614c49119d6c603d913` |
| `live_guard_distilgpt2.json` | 713 | `e5dce0d372fa5906dafc8130af2cfd6c74eb32596cc107984ecaef4147c9b09c` |
| `live_guard_qwen_qwen2_5-0_5b-instruct.json` | 1202 | `34bbf1d8c91fba599483281597971c4db7ece3aab88c76153a800fd0050b5fd3` |
| `live_guard_tinyllama_tinyllama-1_1b-chat-v1_0.json` | 1211 | `1517fb4fd80723a2d3d470796cc872208a1f6b72e8ead28f8d4334b674126db0` |
| `polarity_diagnostic_qwen_qwen2_5-0_5b-instruct.json` | 2712 | `699fc5fc449c40080566a7ac797daacb4d9fbb41d6653ccee6defb846e716906` |
| `polarity_diagnostic_tinyllama_tinyllama-1_1b-chat-v1_0.json` | 2730 | `6e6a1fcf15175e6ac9c86b0e4d3c453667cbdd0e4794d1bcae1720609a37f03b` |
| `protocol_r_prime_qwen_qwen2_5-0_5b-instruct.json` | 1812 | `782a9596e25facb39616fa70b122e9bff1286eb0e1a62913343a5f44331c8c29` |
| `protocol_r_prime_tinyllama_tinyllama-1_1b-chat-v1_0.json` | 1832 | `325949abbd220fbf217945d4787142c48f77d296b95ca0e58646d3b8911a684b` |
| `protocol_r_qwen_qwen2_5-0_5b-instruct.json` | 3720 | `8c97b4ea35944a4bd2b86f9a14f412c59159651e06478007a64416e6160ca89a` |
| `protocol_r_tinyllama_tinyllama-1_1b-chat-v1_0.json` | 3764 | `d5be6776a9f8b09bc9a9ae867f2fc63edb0625ed8245a896adc227f4dcb3b62e` |
| `semantic_entropy_matched_qwen_qwen2_5-0_5b-instruct.json` | 58310 | `85d13a6b9197410f5efd9f1969df4e3f7e0fe6a1fae73dc869210c886e191bcc` |
| `semantic_entropy_matched_tinyllama_tinyllama-1_1b-chat-v1_0.json` | 58489 | `da58f3dff87d51a5cbdcef0e30110ef380969a9dff370c8aa318c2230194e350` |
| `guard_ablation_qwen_qwen2_5-0_5b-instruct.json` | 4949 | `5313ad635d0a6e6032f299a4a018628c60039a05a248de466c5ffb57125b7b72` |
| `guard_ablation_tinyllama_tinyllama-1_1b-chat-v1_0.json` | 5191 | `87e2a8175b1ffce24627d50f4f8db2060a2727fce4b62be3a41f88b58a3b7e62` |
| `trust_integrity_adapted_qwen_qwen2_5-0_5b-instruct.json` | 2936 | `4188b3076e01a658cd95ae87f8598319e915563c849f2948c21bcdecf3983aed` |
| `trust_integrity_adapted_rollup.json` | 2136 | `cc655b006de94f8ec82fb37426ad9ccdee07afd500b9b6547fb3e8dbfaff8558` |
| `trust_integrity_adapted_tinyllama_tinyllama-1_1b-chat-v1_0.json` | 2960 | `02bfb7d7375c251bc8034c359cc8339c8dd4a851c4760001af6d7e5a6743fdd9` |
| `trust_integrity_qwen_qwen2_5-0_5b-instruct.json` | 3191 | `f44edec9e2befd12800d10edfc05a96adcd87c68e818962171aa99b206a07c13` |
| `trust_integrity_rollup.json` | 2180 | `7ca79f5019d7529a4de7fa9ba3822ec962e1a6ce7eaa20d05d1e91a18eb86176` |
| `trust_integrity_tinyllama_tinyllama-1_1b-chat-v1_0.json` | 3247 | `3763ed4e680f3aacbc1256de8b44edf69b40a0d8325d23d2b368a60a1849302b` |

## Checklist

- [ ] A0 polarity diagnostic logged
- [ ] Qwen regression (AUROC/CI/R) matches paper_numbers
- [ ] TinyLlama F/L/R + CIs frozen
- [ ] Matched SE table (both suites, both models)
- [ ] R' held-out for both models
- [ ] Ablations both models
- [ ] paper_numbers.json synced
- [ ] `pytest tests/test_eval_lift_a0_a5.py` green
