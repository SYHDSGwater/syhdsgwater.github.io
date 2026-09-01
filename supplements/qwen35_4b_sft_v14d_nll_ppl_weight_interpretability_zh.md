# v14.d SFT 可解释性实验总结：NLL、困惑度与权重差

日期：2026-08-16

本文汇总 Tmax-SFT 对照、v14.d 教师轨迹 NLL、字/字节级困惑度、权重差、embed
替换消融，以及 Verified50 失败模式。数字均来自本地 `artifacts/` 中已同步的
JSON；逐条 NLL 在 `artifacts/nll_shards/`。

一句话：**SFT 在似然上学会了 MiniMax-M2.5，在 SWE 闭环上忘掉了 4B 自己的短程交卷策略。**
崩溃不是「Qwen3 不能 SFT」，也不是词表膨胀把 token PPL 算歪了。

## 1. 要回答的问题

1. Qwen3.5-9B 相对 Qwen3-8B，在同一套 agent 轨迹上 token NLL 低很多，是不是 15 万→24 万词表造成的假象？
2. v14.d FullFT 相对 Qwen3.5-4B 基座，SWE-bench Verified 前 50 为何从 submit 74% / strict 42% 掉到 submit 32% / strict 约 24%？
3. 基座在教师轨迹上的 NLL 分布如何？SFT 是学会了老师，还是只是把权重搅乱？
4. 权重差是否集中在 embedding？把 SFT 的 tied embed 换回基座，NLL 会不会回到基座？
5. 文献里 AgentForge / FrogMini 对 Qwen3 SFT 能涨，和 v14.d 差在哪？

## 2. 协议

| 项 | 设定 |
|---|---|
| v14.d 学生 | Qwen3.5-4B instruct，FullFT，2 epoch，peak LR `1e-5`（run id `...-v14d-promptfix-lr1e5`） |
| 训练数据 | Orchard MiniMax-M2.5 `scaleswe` + `rebench`，16,124 条 resolved 全长轨迹；选样与 v14.c 相同，只改了首条 user 的 action contract |
| 损失 | assistant 内容 + `<\|im_end\|>` 的 token-mean NLL；ChatML 保留 `<think>`；不做 error mask / 课程 |
| 评测 | mini-swe-agent，Verified 前 50，max 150 turn |
| Tmax 对照 | `allenai/tmax-sft` `thinking_all`，seed `20260816`，n=1000，`max_length=32768` |
| v14.d NLL | train-300 / val-256 / Verified50 轨迹，`max_length=65536` |
| 字/字节 | 仅 assistant target；字 = Unicode code point；字节 = UTF-8；BPB = byte-mean NLL / ln 2 |

官方 chat template 实验用各模型自己的 `apply_chat_template`（`enable_thinking=True`，从 tool_calls 推断 bash tools）。Qwen3 为 JSON tool_call，Qwen3.5 为 XML。mask 在渲染文本上用 `<|im_start|>assistant`…`<|im_end|>` + offset mapping，不用增量 apply（Qwen3.5 会因「No user query」失败）。

机器：数据机 `pro-781584222457`（Tmax、Qwen3-8B）；RL 机 `seetacloud-rl`（4B NLL 与权重差）。

## 3. Tmax-SFT 1k：词表不是 token PPL 差距的原因

模型：Qwen3.5-9B（词表 248,044）vs Qwen3-8B（151,643）。

### 3.1 Token NLL

| 模板 | 9B token-mean | 9B PPL | 8B token-mean | 8B PPL | 8B/9B | 截断 |
|---|---:|---:|---:|---:|---:|---|
| 共享手写 ChatML | 0.317 | 1.374 | 0.800 | 2.225 | 2.52× | 24 / 22 |
| 官方 `apply_chat_template` | 0.274 | 1.315 | 0.792 | 2.208 | 2.89× | 19 / 20 |

官方模板上 9B 更低，差距更大，不是「手写 ChatML 偏袒 Qwen3.5」。

### 3.2 字 / 字节级（官方模板）

| | 词表 | 字/token | nats/字 | 字 PPL | BPB |
|---|---:|---:|---:|---:|---:|
| Qwen3.5-9B | 248,044 | 3.48 | 0.0788 | 1.082 | 0.114 |
| Qwen3-8B | 151,643 | 3.53 | 0.2243 | 1.251 | 0.323 |
| 比值 | 1.64× | 1.02× | **2.84×** | — | 2.84× |

共享 ChatML 上 nats/字比值仍为 **2.48×**（0.091 vs 0.225），字/token 同样几乎一样（约 3.50 vs 3.55）。

这份英文/代码轨迹上，更大词表没有把序列切得更短。token PPL 差距是模型对这段分布更熟，不是分词单位不同造成的假象。

产物：`artifacts/nll_tmax_1k.json`、`nll_tmax_official.json`、`nll_tmax_char_byte.json`。

## 4. v14.d：似然成功，策略失败

同一套 ChatML、keep think、`max_length=65536`。train-300 / val-256 **0 截断**。

### 4.1 教师轨迹：SFT 学会了 MiniMax

| 文本 | 基座 4B | v14.d SFT | 含义 |
|---|---:|---:|---|
| train-300 token-mean | 0.307（PPL 1.36） | **0.211**（PPL 1.23） | BC 成功 |
| val-256 | 0.300 | 0.252 | 同分布泛化，不是只背 300 条 |
| train-300 rebench | 0.258 | 0.152 | 较短/较干净的教师更易学 |
| train-300 scaleswe | 0.323 | 0.230 | 仍下降，但绝对 NLL 更高 |

基座 train-300 逐条分位：P10/P25/**P50**/P75/P90 = 0.222 / 0.257 / **0.319** / 0.370 / 0.409。没有「大部分极低、少数极高」的双峰；教师分布对 4B 是中等难度的克隆目标，不是不可学。

长度方向上基座 NLL 几乎平坦（0–16k 0.334，48–64k 0.301）。SFT 把短轨迹压得更低（0–16k **0.169**），说明它在拟合教师的短程写法，而不只是长上下文。

### 4.2 交叉打分：忘掉自己的成功策略，连自己的评测轨迹都不如基座

| 文本 | 打分模型 | token-mean | seq-mean |
|---|---|---:|---:|
| 基座自己的 Verified50 轨迹 | 基座 | **0.074** | 0.153 |
| 同上 | SFT | 0.109 | 0.246 |
| 其中 0–16k 短轨迹 | 基座 → SFT | **0.189 → 0.315** | 0.219 → 0.371 |
| v14.d 自己的 Verified50 轨迹 | 基座 | **0.086** | 0.120 |
| 同上 | SFT | 0.117 | 0.178 |

两件事同时成立：

- 在教师轨迹上 SFT ≪ 基座（学会老师）。
- 在基座解对过的短轨迹上 SFT ≫ 基座（忘掉短程交卷）。
- 在 SFT 自己滚出来的评测轨迹上，**基座的 NLL 仍然更低**。SFT 没有进入一个「只对自己 rollout 更确定」的新模式，而是把质量函数扭向了 MiniMax 长程探索，评测时自己也不如基座会写那些变长、少 edit、不交卷的句子。

SFT 评测轨迹更长（token 中位 49.6k vs 基座 20.7k），15/50 触到 64k 截断；NLL 被长尾重复动作拉低，不能当成「SFT 更会写」。

产物：`artifacts/nll_v14d_base.json`、`nll_v14d_sft.json`、`nll_v14d_sft_onpolicy.json`、`nll_v14d_base_on_sfttraj.json`、`nll_v14d_val_base.json`、`nll_v14d_val_sft.json`。

## 5. Qwen3-8B × v14.d train-300

口径与 4B 相同：共享 ChatML keep think，0 截断。

| | token-mean | token PPL | nats/字 | 字 PPL | BPB | 字/token |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3.5-4B 基座 | 0.307 | 1.36 | — | — | — | — |
| Qwen3-8B | **0.668** | **1.95** | 0.171 | 1.186 | 0.246 | 3.91 |

8B/4B token NLL ≈ **2.17×**。8B 参数更多，但不贴 MiniMax/Qwen3.5 这条分布。字/token 3.91，高于 Tmax 的 ~3.5，说明 v14.d 轨迹更「碎」（更多短 token），不是 8B 切得更短。

逐条 P10–P90：0.512 / 0.615 / **0.731** / 0.844 / 0.917。

产物：`artifacts/nll_v14d_train300_qwen3_8b_char_byte.json`。

## 6. 权重差：全局很小，浅层 MLP 与 embed 方向不对齐

### 6.1 v14.d FullFT vs Qwen3.5-4B

723 个 tensor，全局相对 L2 **0.75%**（`‖ΔW‖ / ‖W₀‖`），余弦 −0.008。

| family | rel | cos(ΔW, W₀) |
|---|---:|---:|
| 全局 | 0.75% | −0.008 |
| `embed_tokens` | 1.02% | **−0.115** |
| `up_proj` / `down_proj` | 1.83% / 1.83% | ≈ −0.002 |
| `gate_proj` | 1.68% | −0.003 |
| `norm` | 0.012% | ≈ 0 |
| 层相对变化中位 | **0.003%** | — |

相对变化最大的 12 个张量全是 **layer 0–6 的 MLP up/down**，rel ≈ **2.05–2.16%**。层中位几乎为 0，说明多数深层几乎没动，更新集中在浅层 MLP 和 embedding。

`embed_tokens` 的余弦为负：embedding 更新方向与原向量相反，不只是同方向缩放。

对照 v11（Qwen3.5-9B LoRA r64，AgentForge-plus 20k）：LoRA 张量上全局 rel 1.64%，相对全模型 1.00%；层中位 1.62%，更均匀；无 embed 更新。v14.d 是「浅层大、深层几乎不动」的 FullFT，不是均匀的小适配。

产物：`artifacts/weight_delta_v14d.json`、`weight_delta_v11.json`。

### 6.2 把 SFT 的 tied embed 换回基座

SFT 权重 + 基座 `embed_tokens`（tied lm_head）。embed 相对 SFT 自身的 rel 为 1.02%，与上面 `embed_tokens` 一致。

| 文本 | SFT 原权重 | SFT + 基座 embed |
|---|---:|---:|
| train-300 | 0.211 | **0.212** |
| val-256 | 0.252 | 0.253 |
| 基座 Verified50 轨迹 | 0.109 | 0.107 |
| SFT Verified50 轨迹 | 0.117 | 0.117 |

NLL 几乎不动。灾难不在 embedding 数值本身，而在浅层（及连带的）策略表征。换 embed 救不回来。

产物：`artifacts/nll_sft_base_embed.json`。

## 7. Verified50：基座会当 agent，SFT 把停机策略弄丢

### 7.1 评测对照

同一 50 题、同一 scaffold：

| | 基座 4B | v14.d |
|---|---:|---:|
| submit | **74%**（37/50） | 32% |
| strict | **42%**（21/50） | 约 24%（12/50；丢掉 11，新解开 2） |
| 撞 150-turn | 22% | 66% |
| 动作里真正 edit | 13.2% | 2.7% |
| 首次 edit 中位轮次 | 15 | 28 |
| 空 patch | 10% | 40% |
| git history hacking | 2% | 12% |
| malformed bash | 14% | 3% |

协议更干净（bash 更合法、`<think>` 完整），闭环从「改完就交」变成「一直逛、很少改、几乎不交」。

### 7.2 基座 29 道失败不是「不会当 agent」

| 模式 | 题数 | 占失败 |
|---|---:|---:|
| 交卷但隐藏 F2P=0 | **16** | 55% |
| 撞上限，仓库里已有 diff | 8 | 28% |
| 撞上限，空 patch | 2 | 7% |
| 全程零 edit | 3 | 10% |

停机：`finish` 37、`LimitsExceeded` 11、`ContextLimitExceeded` 2。基座**会交**；交错是第一大失败。含 `django-10097`：438 个 F2P 全过但没 submit。

官方难度：

| 难度 | 题数 | 解出 |
|---|---:|---:|
| `<15 min` | 18 | 13（72%） |
| `15 min – 1 hour` | 26 | 8（31%） |
| `1–4 hours` | 6 | 0 |

弱点是中等题修不准 + 少数停机失败，不是 scaffold 不会用。SFT 丢掉的 11 道基座已解对题里，10 道是烧满 150 turn。

产物：`artifacts/verified50_qwen35_4b_base_run1/summary.json`。

## 8. 和 AgentForge / FrogMini 为什么结论相反

文献能涨，通常同时满足若干条；v14.d 几乎全反着做。

| | 文献里能涨的 Qwen3 SFT | v14.d |
|---|---|---|
| 学生起点 | 弱 base 或更大模型（如 8B base ~8% → 38%；FrogMini 14B） | **已有 42% strict 的 4B instruct** |
| 教师 | 同构 scaffold、同量级或专门合成的短/中轨迹 | MiniMax-M2.5（约 230B）全长成功轨迹 |
| 损失 | 常截断、mask 失败步、课程、只学关键 action | 全长 BC，无 error mask |
| 后处理 | 常接 RL | 只 SFT |
| 失败模式 | AgentForge 也报告：把 agent 数据 SFT 到 reasoning 模型会「想满上下文、永不 task-complete」 | 同构：烧满 150 turn、几乎不 submit |

v14.d 不是「SFT 无效」，而是 **用过强老师的长程探索，覆盖了学生已经会的短程停机**。似然下降（0.307→0.211）与 SWE 崩溃同时出现，是行为克隆的典型错配，不是优化没跑起来。

训练数据本身 16,124 条最后一步都是 submit、空 patch 0%、edit 占比 8.2%、首次 edit 中位 13。学生学到的是「教师那种很长的探索过程」，评测时 4B 走不完、也不知道该停。

## 9. 结论

1. **词表**：Tmax 上 8B/9B 字/token 几乎相同，nats/字比值 2.5–2.8×，token PPL 差距是真实的分布匹配，不是 24 万词表虚低。
2. **v14.d SFT 在 NLL 上成功**：train-300 0.307→0.211，val-256 同步下降。
3. **在策略上失败**：基座短程成功轨迹 NLL 变差（尤其 0–16k 0.189→0.315）；评测 submit/strict 崩溃；SFT 连自己的评测轨迹都不如基座会建模。
4. **权重**：全局只动 0.75%，但浅层 MLP ~2.1%、embed 余弦 −0.115；换回基座 embed 几乎不改变 NLL。
5. **8B 也不贴 MiniMax**：同一 train-300 上 token NLL 0.668，约为 4B 基座的 2.17×。
6. **基座不是废物**：Verified50 已 42% strict；该补的是中等题修准和偶发停机，不是用 230B 全长成功轨迹做 2 epoch FullFT。

## 10. 产物索引

汇总 JSON（以 `artifacts/` 为准）：

| 文件 | 内容 |
|---|---|
| `nll_tmax_1k.json` | Tmax 1k 共享 ChatML |
| `nll_tmax_official.json` | Tmax 1k 官方 template |
| `nll_tmax_char_byte.json` | 字/字节 PPL 与词表 |
| `nll_v14d_base.json` | 4B 基座 train-300 + 基座 Verified50 |
| `nll_v14d_sft.json` | v14.d 同上 |
| `nll_v14d_sft_onpolicy.json` | SFT 打自己的 Verified50 |
| `nll_v14d_base_on_sfttraj.json` | 基座打 SFT 的 Verified50 |
| `nll_v14d_val_base.json` / `nll_v14d_val_sft.json` | val-256 |
| `nll_v14d_train300_qwen3_8b_char_byte.json` | Qwen3-8B × train-300 |
| `nll_sft_base_embed.json` | SFT 权重 + 基座 tied embed |
| `weight_delta_v14d.json` | 4B FullFT vs 基座 |
| `weight_delta_v11.json` | 9B LoRA v11 对照 |
| `nll_shards/` | 逐条 NLL jsonl |
| `v14d_train_sample_300.jsonl` | train-300 输入样本 |
| `verified50_qwen35_4b_base_run1/` | 基座 Verified50 评测 |

未纳入本文的远程大文件：Tmax 原始 1k jsonl、v14.d val/SFT 评测轨迹全文（仍在远程）；RL 上未完成的 `shards/train1k_*` 不是正式结果。
