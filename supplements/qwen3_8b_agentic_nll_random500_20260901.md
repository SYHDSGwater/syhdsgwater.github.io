# Qwen3-8B agentic trajectory NLL — random/resolved 500 audit

Recorded 1 September 2026.

## Scoring protocol

- Student: `Qwen/Qwen3-8B`.
- Rendering: the checkpoint's official chat template, including original tools and message roles where present.
- History: historical assistant reasoning is retained whenever the source schema and Qwen3 template can represent it. AgentForge's plain `THOUGHT:` content was retained for all audited historical turns.
- Targets: assistant body, assistant tool call, and assistant end token only. System, user, and tool observations are context.
- Aggregation: token-weighted NLL over fully included assistant turns.
- Prompt cap: 40,959 tokens, leaving one token within Qwen3-8B's native 40,960-token context.
- Backend: vLLM prompt log-probabilities. A shared smoke trajectory matched the Transformers implementation within 0.13% relative NLL.

## Samples

- SWE-Lego: 500 deterministic samples from the 4,323-row `resolved` split of [`PrimeIntellect/SWE-Lego-Real-Data-Verified`](https://huggingface.co/datasets/PrimeIntellect/SWE-Lego-Real-Data-Verified), dataset revision `8b0d2bed6f04ef571ca04abebf737ea23cbceaf3`.
- Klear: 500 seeded uniform samples without replacement from the 65,994 trajectories in [`Kwai-Klear/SWE-smith-mini_swe_agent_plus-trajectories-66k`](https://huggingface.co/datasets/Kwai-Klear/SWE-smith-mini_swe_agent_plus-trajectories-66k), dataset revision `e06f33b1b54cf6727b077b54b5a71c81f2405f10`.
- Open-SWE: the existing fixed v1.2 random-500 sample from [`nvidia/Open-SWE-Traces`](https://huggingface.co/datasets/nvidia/Open-SWE-Traces).
- Every group produced 500 valid records and zero scoring errors.

## Results

| Trajectory family | Complete-turn target tokens | Token NLL | Perplexity | Turn-mean NLL | Truncated trajectories | Non-truncated-only NLL |
|---|---:|---:|---:|---:|---:|---:|
| SWE-Lego Real Data Verified resolved-500 | 4,387,082 | **0.7376** | 2.091 | 0.9233 | 283 / 500 (56.6%) | 0.7577 |
| Klear AgentForge / SWE-smith random-500 | 3,207,993 | **0.8119** | 2.252 | 1.2634 | 17 / 500 (3.4%) | 0.8242 |
| Open-SWE-Traces v1.2 random-500 | 7,604,303 | **1.2289** | 3.417 | 1.2761 | 315 / 500 (63.0%) | 1.2164 |

## Interpretation boundary

These measurements compare a fixed student with candidate trajectory distributions. They do not isolate the causal contribution of these 500 examples to a trained checkpoint. The published SWE-Lego and Klear recipes use larger mixtures, different harnesses, and additional training choices. The results support a prospective NLL-bucketing experiment; they do not establish a universal optimum or a hard threshold.
