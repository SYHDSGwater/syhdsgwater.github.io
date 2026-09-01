# v14.d data-protocol and loss-mask audit

Audit date: 1 September 2026. This note records the checks used to distinguish a healthy but ineffective SFT run from a silent data or template failure.

## Frozen data provenance

v14.c produced 16,124 training trajectories and 256 issue-disjoint validation trajectories. The full-corpus audit accepted both splits and recorded:

| Split | Rows | Assistant turns | Native balanced think blocks |
|---|---:|---:|---:|
| Train | 16,124 | 745,175 | 745,175 |
| Validation | 256 | 11,624 | 11,624 |

The selection preflight, benchmark blocklist, split topology, and file hashes were frozen before formal training.

## v14.c → v14.d protocol alignment

The original Orchard first-user prompt allowed “at least one command,” while the exported mini-swe-agent protocol accepted one action per assistant turn. v14.d was materialized from the frozen v14.c files with a single transformation:

- require exactly one `<think>...</think>` block;
- require exactly one outer bash block containing exactly one shell action;
- change only the first user message in every record;
- preserve issue selection and every assistant and observation body.

The materializer pinned the v14.c source SHA-256 values, failed closed when a prompt did not match the known Orchard structure, rewrote all 16,124/256 rows, and rejected any remaining multi-action wording.

## Training chat template and loss mask

The formal run used the installed MS-SWIFT template with:

- `preserve_thinking=true`;
- `add_non_thinking_prefix=false`;
- `loss_scale=default`;
- `max_length=65536` and deletion rather than truncation for over-length examples;
- no packing.

The template audit selected the first 16 and the 16 largest serialized training rows. It rendered the frozen ChatML independently, tokenized it with offsets, and compared it against MS-SWIFT training-mode output. All 32 rows passed exact checks for input IDs, supervised-token positions, and label-token identity.

The resulting objective treated system, user, and tool observations as context. It supervised every assistant turn's reasoning and action body plus the assistant `<|im_end|>\n` terminator. Historical assistant turns remained in the serialized context.

## The separate evaluation-template bug

An older evaluation template omitted historical assistant reasoning from later-turn context. That bug affected rollout evaluation, not the v14.d training serialization or training loss mask. The repaired evaluation preserved all historical assistant reasoning; low submission and high turn-cap behavior remained.

## Optimization audit

The completed v14.d W&B run used Qwen3.5-4B text FullFT, 8×B300, BF16, Flash Attention 2, DeepSpeed ZeRO-3, gradient checkpointing, 64K context, two epochs, global batch size 32, cosine decay, and peak learning rate `1e-5`.

Across 1,008 updates, every recorded loss and gradient norm was finite. Mean update loss was 0.243; the final 50-update mean was 0.214. Held-out loss fell from 0.26464 to 0.25405. Independent teacher forcing also fell from 0.307 to 0.211 on train-300 and from 0.300 to 0.252 on validation-256.

These checks do not prove that the data or recipe was optimal. They rule out the specific hypotheses that the run silently changed the selected data, retained a train/deployment action-contract mismatch, dropped historical reasoning during training, supervised the wrong role classes, omitted assistant terminators, or suffered numerical optimizer failure.

## Local provenance

The audit is derived from the frozen project records and executable checks:

- `docs/qwen35_4b_sft_v14c_fullft_64k_zh.md`
- `docs/qwen35_4b_sft_v14d_nll_ppl_weight_interpretability_zh.md`
- `docs/sft_v14d_wandb_run_20260831_zh.md`
- `scripts/agentic_materialize_sft_v14d_promptfix.py`
- `scripts/verda_h200/audit_sft_v14d_formal.py`
- `scripts/verda_h200/audit_ms_swift_v14c_export_protocol.py`
