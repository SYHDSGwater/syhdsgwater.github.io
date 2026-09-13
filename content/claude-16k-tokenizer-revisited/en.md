<callout>
**TL;DR:** We initially suspected that Claude’s vocabulary reduction from roughly 49K to 16K might relieve an LM-head / softmax bottleneck that worsens with looped depth. Two subsequent experiments failed to support that story; their direction was actually the opposite. A more plausible explanation now is: **a smaller vocabulary produces more tokens, buying stronger performance and better-looking financial numbers.**

![](/images/claude-tokenizer/anthropic-arr-tokenizer-2026.en.svg)
</callout>

The tokenizer introduced after Claude Opus 4.7 is an unusual design. According to [Sander Land’s reverse engineering of the Claude tokenizer](https://www.tokenize.rs/claude), its vocabulary shrank dramatically, from about 49K to about 16K. Anthropic’s own [token-counting documentation](https://platform.claude.com/docs/en/build-with-claude/token-counting) explicitly warns that Claude 4.7 and later use a new tokenizer that produces **approximately 30% more tokens** for the same input text.

In [our previous article](/en/notes/claude-16k-tokenizer-reasoning-architecture/), we proposed a more ambitious explanation: a small vocabulary might work together with recurrent / looped depth, relieving the softmax bottleneck by shrinking the output space. It was an elegant, falsifiable hypothesis. So we tested it. The results did not support it.

# 1. An elegant hypothesis fails: small vocab does not relieve a bottleneck that grows with depth

The classical softmax bottleneck comes from the final layer’s low-rank constraint. A standard LM head is

$$
z = Wh,\qquad W\in\mathbb{R}^{V\times D}
$$

Stacking hidden states across contexts gives a logit matrix with $`\mathrm{rank}(L)\le D`$. When $`V\gg D`$, a natural concern follows: as the backbone becomes more capable and hidden states carry more complex information, a linear-softmax decoder of fixed width $`D`$ may struggle increasingly to map that information onto a huge vocabulary. In 2026, Godey & Artzi extended this argument to the backward pass: $`g_h=W^Tg_z`$ projects vocabulary-space gradients into a subspace of dimension at most $`D`$. They reported that 95–99% of the logit-gradient norm may lie outside the LM-head feedback subspace. [Lost in Backpropagation](https://arxiv.org/abs/2603.10145)

If Claude also uses a looped Transformer, the story becomes more tempting: recurrent depth adds latent compute while decoder rank stays fixed, potentially creating a **depth–decoder mismatch**. A small vocabulary looks like a reasonable patch.

We ran EXP-001/001b on Ouro-1.4B. Ouro supports native recurrent depth, allowing us to extract hidden states at $`T=1,2,3,4`$ from the same checkpoint. Alongside its native linear LM head, we trained a rank-expanding residual head:

$$
z(h)=Wh+U\operatorname{GELU}(Ah)
$$

The additional nonlinear feature basis raises the logit family’s rank upper bound from approximately $`D`$ to $`D+r`$. If deeper recurrent states are increasingly constrained by linear softmax, the rich head’s marginal benefit should grow with $`T`$.

The results went in the opposite direction. EXP-001b froze native $`W`$ and added matched linear and nonlinear residuals:

| Depth | $`G_{linear}`$ | $`G_{nonlin}`$ |
| --- | --- | --- |
| T1 | +0.3039 | +0.0546 |
| T2 | +0.0111 | +0.00283 |
| T3 | ≈0 | ≈0 |
| T4 | 0 | 0 |

Here $`G_{linear}`$ is the CE reduction obtained by a linear residual over the frozen native head; $`G_{nonlin}`$ is the additional gain of the matched nonlinear residual over the linear residual. We found that **the entire opportunity for post-hoc decoder adaptation was disappearing**. In the original EXP-001, even fully refitting a roughly 100M-parameter linear head produced no held-out CE gain at T2/T3/T4. The best validation checkpoints at those depths all occurred around the first update; the native head performed better. With frozen hidden states, multinomial logistic regression is convex in $`W`$, making “getting stuck in nonconvex optimization” an unconvincing explanation on its own. A more natural interpretation is that recurrent refinement moves hidden states toward the readout geometry already served by the native LM head.

We initially expected

$$
T\uparrow\Rightarrow\text{decoder pressure}\uparrow
$$

The observations were closer to

$$
T\uparrow\Rightarrow\text{decoder adaptation opportunity}\downarrow
$$

This does not establish a universal law of recurrent decoder alignment. But on Ouro, the forward softmax bottleneck did not worsen with recurrent depth. [Experiment code and results](https://github.com/SYHDSGwater/depth-decoder-mismatch)

We then ran EXP-003C using a tiny shared-body Transformer with D=32 and four layers. We held the byte vocabulary fixed at 256 and added never-target classes only to the output head, expanding $`V_{out}`$ from 256 to 4096, then compared T1 with T4. Define the active-vocabulary penalty

$$
P_{active}(T)=CE_{active}(T,4096)-CE_{active}(T,256)
$$

and the interaction

$$
I_{active}=P_{active}(4)-P_{active}(1)
$$

If recurrence amplifies the optimization burden of a larger output space, we should observe $`I_{active}>0`$. Instead, $`P_{active}(1)=+0.0254`$ and $`P_{active}(4)=+0.0010`$, giving $`I_{active}=-0.0244`$, with a five-seed 95% interval of $`[-0.0853,+0.0364]`$. This is neither a statistically significant negative effect nor evidence of equivalence to zero. It provides no positive evidence for the hypothesis; four of five seeds had negative interactions. The result also agrees with Murugan’s 2026 causal study: geometric gradient compression exists, yet expanding never-target output classes from 256 to 4096 does not consistently harm learning. [Does the LM Head Create a Harmful Gradient Bottleneck?](https://arxiv.org/abs/2608.16671)

# 2. A more direct challenge: a severe bottleneck should have changed the LM head by now

Both experiments have limitations. Outside the experiments, however, there is a simpler piece of revealed-preference evidence. From early GPT architectures through recent open-weight frontier models, attention, FFNs, positional encoding, normalization, MoE, routing, optimizers, and KV caches have all been extensively reworked. Yet the final

$$
h\rightarrow Wh\rightarrow\operatorname{softmax}
$$

remains largely intact. Mixture of Softmaxes, DOC, Sigsoftmax, and other nonlinear or high-rank output families have all been proposed. The softmax bottleneck was already a mature research topic in the RNN language-model era. What stands out is that these methods never became standard components of large models.

This industry evidence matters because the LM head is **one of the easiest places to run a local A/B test**. Changing attention affects kernels, caches, parallelism, and the serving stack. A richer decoder carries much less engineering risk. If a rank-expanding head reliably delivered substantial loss improvements in frontier-scale pretraining, it would be difficult to explain why every major lab missed it for a decade. We cannot see closed labs’ internal ablations; my inference is that they have systematically tested this direction. The more plausible reading is that the classical softmax rank constraint exists mathematically but is usually **not the binding constraint in modern large models**. During end-to-end training, the backbone learns representation geometry suited to a linear head, while the marginal benefit of higher output rank fails to offset additional parameters, communication, and serving costs.

This also explains why $`V\gg D`$ alone does not imply a severe bottleneck. The target conditional distribution must have a sufficiently large **effective log-probability rank**; the vocabulary’s nominal dimension cannot establish that. A real distribution can be full-rank yet have a rapidly decaying singular spectrum, allowing a $`D`$-dimensional approximation to capture most CE-relevant structure. Modern Transformers also often have widths in the thousands or tens of thousands, a very different regime from early RNN language models with a few hundred hidden dimensions.

Reconsidering Claude’s 16K vocabulary, I now assign a low posterior to “relieving the classical softmax bottleneck.”

# 3. Measured comparison: how many more tokens does Claude 16K use than o200k_base?

Before discussing compute reallocation, consider the actual difference between Claude’s new tokenizer and OpenAI’s `o200k_base`. In 2026, Playcode compared identical byte inputs across **16 real fixtures**. It used Anthropic’s official `count_tokens` endpoint on the Claude side and `tiktoken`’s `o200k_base` on the OpenAI side, cross-checking against actual API `usage` from GPT-5.1/5.5/5.6 Sol. This measures tokenization itself, without mixing in output length, reasoning budgets, or agent trajectories.

| Identical content | Claude 4.7+ / o200k_base token ratio |
| --- | --- |
| TypeScript | **1.73×** |
| Rust | **1.58×** |
| JavaScript | **1.52×** |
| Python | **1.50×** |
| HTML | **1.36×** |
| English prose | **1.40×** |
| Chinese prose | **1.44×** |
| Chinese chat | **1.53×** |

Source: [The Same TypeScript Costs 73% More on Claude Than on GPT](https://playcode.io/blog/real-price-of-frontier-models). One sample makes the difference concrete: the same 2,888-character TypeScript file uses **681 tokens** with `o200k_base` and **1,178 tokens** with Claude’s new tokenizer. Differences between Claude’s own old and new tokenizers are also concentrated in English and code: within the same fixtures, English prose rose 34%, TypeScript 31%, Rust 29%, and an agent system prompt 39%, while Chinese prose was nearly unchanged. Anthropic’s “approximately 30% more tokens” is therefore a workload average. For coding and agent workloads central to Claude, the gap against `o200k_base` frequently reaches **1.5–1.7×**.

An independent vocabulary experiment points in the same direction. Using an exactly countable Claude 4.7+ reconstruction and `o200k_base`, the `novocab` project measured the share of complete words in running text represented as a single piece: **38.3% vs 88.1%** for English, **42.3% vs 88.9%** for code identifiers, and a median of **4.6% vs 67.9%** across 11 other Latin-script languages. [novocab tokenizer analysis](https://github.com/wdhwg001/novocab) This is not a performance benchmark, but it directly exposes the different inductive biases: `o200k_base` stores many common lexical units in its vocabulary; Claude 4.7+ more often decomposes the same string into reusable subpieces.

The first-order effect of 16K is concrete: **Claude performs more token-level state transitions on identical text, especially in code and agent workloads.** We can then ask whether that extra token computation buys capability, and why Anthropic accepts the cost.

Anthropic supplies an important fact: the Claude 4.7+ tokenizer produces approximately **30% more tokens** for identical text on average, and its current pricing documentation says the tokenizer “contributes to improved performance on a wide range of tasks.” [Anthropic pricing](https://platform.claude.com/docs/en/about-claude/pricing) That statement does not explain the mechanism, but it does establish that Anthropic accepts worse compression from finer tokenization in exchange for performance.

The compute allocation is straightforward. During generation, if the same text grows from $`L`$ tokens to roughly $`1.3L`$, the autoregressive decoder must execute about 1.3 times as many sequential decoding steps. Each token receives another Transformer state update. If the model also has recurrent depth, sequence steps multiply with latent-depth compute. This effect holds even without assuming Claude uses a looped Transformer:

$$
L_{token}\uparrow\quad\Rightarrow\quad\text{sequential Transformer compute}\uparrow
$$

Inputs allow parallel prefill, but longer sequences still mean more representation slots, more attention / FFN computation, and a finer compositional interface. A direct consequence of small vocab is to return some composition that the tokenizer had merged away to the neural network.

For a frontier model at the scale of several trillion parameters, the compute saved by narrowing the LM head is a small part of total inference FLOPs. Additional full Transformer forwards dominate.

$$
\text{finer tokenization}\rightarrow\text{more Transformer forwards}\rightarrow\text{more inference compute}
$$

On the output side, needing 1.4–1.7× as many tokens for the same text approximately means 1.4–1.7× as many autoregressive Transformer forwards. Every step traverses attention, FFNs/MoE, and the rest of the backbone. That is the main computational cost of the 16K tokenizer. Combined with Anthropic’s stated performance benefit, a direct explanation is that **more token-level computation is itself a performance cost Claude is willing to pay.**

# 4. A more plausible explanation: an effective price increase improves the numbers around an IPO

Claude’s core API and enterprise services charge by tokens. Anthropic’s token-counting documentation says identical input produces approximately 30% more tokens and explicitly warns that **billing reflects this tokenizer’s counts**. Meanwhile, the standard API list prices for Opus 4.6, Opus 4.7, Opus 4.8, and Opus 5 remain **$5 / MTok input and $25 / MTok output**. [Anthropic token counting](https://platform.claude.com/docs/en/build-with-claude/token-counting) / [Opus 4.8](https://www.anthropic.com/news/claude-opus-4-8)

In the simplest approximation, if fixed text becomes 1.3 times as many tokens under the 4.7 tokenizer and the price per million tokens does not fall, billed units rise about 30%. Anthropic explicitly asks developers to reassess costs in its migration documentation. Technically, the model gets more sequence-level compute. Commercially, that compute maps naturally onto the token usage customers see, so Anthropic need not absorb all of its additional cost.

![](/images/claude-tokenizer/anthropic-arr-tokenizer-2026.en.svg)

**Figure: Anthropic’s 2026 ARR and a tokenizer-normalized scenario.** The orange line shows officially announced and market-reported annualized revenue run rates. After the April 16 launch of Opus 4.7, the teal dashed line divides public values by 1.30. Using the reported end-July threshold of more than $65 billion, the equivalent in old-tokenizer units is approximately $50 billion, a difference of about 23.1%.

This scenario assumes that all revenue after launch is exposed to the new tokenizer and extends the official approximately 30% inflation for identical **input** text to all billed units, holding text volume and unit prices fixed. Actual migration rates and revenue composition are unknown. If the affected share of reported revenue is only q, the conversion factor is `(1 − q) + q / 1.30`. The dashed line is a unit-sensitivity analysis, not a financial restatement or a causal estimate of the tokenizer’s contribution to growth. Its break at the switch does not represent a revenue decline.

Data and definitions: the start-of-year value carries forward approximately $9 billion from year-end 2025; February 12 is $14 billion; April 6 is above $30 billion; the May 28 announcement says the run rate had exceeded $47 billion that month; the end-July value exceeded $65 billion, reported on August 17. Connecting lines are visual guides, with no extrapolation into August or September. ARR here means annualized revenue run rate, not realized full-year revenue. Sources: [Anthropic, February 12](https://www.anthropic.com/news/anthropic-raises-30-billion-series-g-funding-380-billion-post-money-valuation), [April 6](https://www.anthropic.com/news/google-broadcom-partnership-compute), [May 28](https://www.anthropic.com/news/series-h), [Reuters, August 17](https://www.investing.com/news/stock-market-news/anthropic-revenue-run-rate-tops-65-billion-source-says-4864031); [Opus 4.7 launch](https://www.anthropic.com/news/claude-opus-4-7) and [token-counting documentation](https://platform.claude.com/docs/en/build-with-claude/token-counting).

This suggests a natural chain:

$$
\text{smaller vocab}\rightarrow\text{more tokens}\rightarrow\text{more Transformer steps}\rightarrow\text{higher billed token usage}
$$

I now prefer to view 16K as **architecture × product economics co-design**. Anthropic can exchange a finer discrete interface for more neural computation while retaining a simple per-million-token pricing interface. Customers see unchanged nominal prices, while the token measurement unit itself changes underneath.

The commercial argument can go further: **LLM inference may have substantially more margin than the near-cost business many people imagine.** In a leaked transcript of a May 20, 2026 investor meeting, Liang Wenfeng described DeepSeek’s API pricing rule as recovering the cost of a batch of equipment in roughly ten months. He then said that “sixfold profit” might sound high but was not high given current AI efficiency; even fourfold or threefold returns could remain substantial. He also described DeepSeek’s business and consumer online services as byproducts of AGI research: a lightweight API maintenance team could support a business with hundreds of millions of dollars in ARR. [Full investor-meeting transcript](https://chinaresearchcollective.substack.com/p/liang-wenfeng-investor-meeting-may) This is not an Anthropic margin disclosure, and DeepSeek’s cost structure cannot be transferred directly to Claude. It does provide an industry reference: **frontier-model inference unit economics may be much stronger than outsiders assume.**

If each billed token already has a stable positive contribution margin, and demand does not fall proportionately while model prices stay fixed, splitting identical content into more tokens mechanically raises revenue and can raise profit. The profit increase depends on the marginal cost of the extra Transformer computation. The key variable is

$$
\Delta \Pi \approx \Delta N_{token}\times (P_{token}-C_{marginal\ token})
$$

As long as $`P_{token}>C_{marginal\ token}`$, additional billed tokens bring additional contribution profit. The business logic does not require savings in the LM head. Additional full Transformer forwards are the main new cost for a frontier model. With a substantial positive inference contribution margin, Anthropic can increase computation per request, improve capability, and earn more revenue and profit through token-based billing. In this reading, 16K deliberately increases inference compute while the billing system covers the increment.

The timing makes this interpretation especially relevant. Public reporting in 2026 indicates that Anthropic is preparing a potentially enormous IPO, bringing revenue growth, gross margin, and long-term infrastructure costs under direct capital-market scrutiny. [Reuters](https://www.reuters.com/technology/artificial-intelligence/ai-models-capabilities-leap-comes-with-new-safety-warnings-2026-09-09/) / [Financial Times](https://www.ft.com/content/9536c7b9-c600-48ec-8fe2-453b0ca187e9) At this stage, a tokenizer that improves performance and increases token usage without an explicit price increase on the rate card has obvious commercial appeal. **This remains my personal interpretation; Anthropic would presumably reject the characterization of an effective price increase.**

# 5. What else might 16K optimize?

Some benefits cannot be explained by more forwards alone. Reverse engineering suggests that Claude’s tokenizer is closer to a minimum-piece / PathPiece design and explicitly uses word-boundary state. Its changes go beyond cutting conventional BPE merges down to 16K. Anthropic is also increasing structural reuse per vocabulary slot: storing fewer `" token"` forms, surface variants, and long lexical chunks, while leaving more composition to the model. That direction could help code identifiers, tool schemas, URLs, long-tail strings, and rare words, as discussed in our previous article.

After these experiments, I would no longer connect those observations into a story that “16K repairs the LM head.” A more robust, simpler account is that **Claude reduces lexical memorization in the tokenizer and leaves more work and computation to the Transformer; longer token sequences are both a performance cost and the billing unit.** Anthropic confirms approximately 30% more tokens and attributes part of the performance improvement to the tokenizer. Our experiments found no evidence that recurrent depth amplifies the softmax / output-space bottleneck. Together, these observations explain reality better than the original, more elegant bottleneck story.

My current summary of Claude’s 16K design is this: 16K may make the Transformer work harder, and when an API charges by tokens, that technical choice has a natural commercial payoff.

---

# Appendix: experimental recipes and statistical definitions

Full code, configurations, and per-run records: [depth-decoder-mismatch](https://github.com/SYHDSGwater/depth-decoder-mismatch).

## A. EXP-001 / EXP-001b: does recurrent depth increase the decoder bottleneck?

| Item | EXP-001 | EXP-001b |
| --- | --- | --- |
| Core question | As Ouro’s recurrent depth increases, does a rank-expanding rich head gain more over a refitted linear head? | After removing optimization drift from full-head refitting, does the nonlinear residual’s gain over a matched linear residual still grow with depth? |
| Backbone | `ByteDance/Ouro-1.4B`; 24 shared layers; hidden size 2048; vocabulary 49,152; native recurrence T=1..4 | Same checkpoint and cached hidden states |
| Data | FineWeb-Edu `sample-10BT`; 31,250 documents in the primary 1M run; 800k / 100k / 100k train/validation/test targets; document splitting before windowing | Reuses the same 1M cached-state dataset: a post-hoc mechanistic audit, not an independent replication |
| Sampling | seq_len=1024; predictor positions ≥128 within each document; 32 targets/window; T1..T4 fully paired on identical targets | Identical |
| Native head | $`z=W_{native}h`$; diagnostic only | $`W_{native}`$ frozen throughout, forming the shared base for every residual arm |
| Linear family | Full bias-free $`Wh`$, initialized from native $`W`$ and refitted; approximately 100.7M trainable parameters | $`z=W_{native}h+U(Ah+b)`$; low-rank linear residual, width=512 |
| Rich family | $`z=Wh+U\mathrm{GELU}(Ah)`$; extra nonlinear feature basis, theoretically expanding logit-family rank from approximately $`D`$ to $`D+r`$ | $`z=W_{native}h+U\mathrm{GELU}(Ah+b)`$; same residual shape and parameter count as the linear residual |
| Optimization | FP32 AdamW; wd=0; batch=256; constant LR=1e-4; probe seeds 11/29/47; select the best validation checkpoint, then evaluate test once | FP32 AdamW; wd=0; batch=256; LR grid `{3e-5,1e-4,3e-4}` tuned on seed11; report seeds 29/47/83; fixed 2000-step budget; step0 included in selection, with denser early validation |
| Primary metric | $`R_{head}(T)=CE^*_{linear}(T)-CE^*_{rich}(T)`$; DDM predicts $`dR/dT>0`$ | $`G_{nonlin}(T)=CE_{linear\ residual}(T)-CE_{nonlinear\ residual}(T)`$ |
| Key results | 1M run: T1 regret +0.0779, T2/T3/T4 ≈0; slope **-0.02340**, 95% document-bootstrap CI `[-0.02407,-0.02275]`. Full linear refitting also has no held-out gain at T2–T4; all best checkpoints are approximately step1 | $`G_{nonlin}`$: **0.0546 → 0.00283 → ≈0 → 0**; $`G_{linear}`$: **0.3039 → 0.0111 → 0 → 0**. Every T4 residual arm selects step0 |
| Supported conclusion | At Ouro’s native recurrent depths, we do not observe a worsening linear-softmax forward bottleneck. Post-hoc decoder adaptation opportunities instead disappear rapidly with depth, consistent with decoder alignment / iterative refinement. | |
| Main limitations | Full-W refitting is highly overparameterized; pilot/1M runs are sequential scaling, not independent confirmatory replications | Reuses an already inspected test set; low-rank residuals cover only one class of richer decoder; results cannot establish the absence of softmax bottlenecks in all models |

## B. EXP-003C: does recurrence amplify training damage from output-vocabulary inflation?

| Item | Recipe |
| --- | --- |
| Core question | Does Murugan’s never-target output-class null reverse when weight-shared recurrence is added? Does larger $`V_{out}`$ hurt active-token learning more at T4 than T1? |
| Model | Murugan-style tiny Transformer: D=32, four layers, four heads, FFN=128, pre-LN, GELU, zero dropout. Repeat the full four-layer body T times as one shared recurrent block. Add absolute position only once at the initial input; apply final LayerNorm only at the final exit |
| Tokenizer / active vocabulary | WikiText-2 raw bytes; identity byte IDs 0..255; input and valid-target vocabularies remain fixed at 256 |
| 2×2 intervention | T ∈ {1,4} × $`V_{out}`$ ∈ {256,4096}; the 4096 arm appends 3840 trainable output rows that are never inputs or targets |
| Pairing | Within each seed, all four arms have identical initial embeddings, recurrent body, and active output rows, with paired batch order. The 4096 arm only appends dummy rows |
| Training objective | Compute **raw full-softmax CE** once at the final recurrent state. No intermediate-loop loss at T4, avoiding confounding depth with the number of supervision events |
| Training budget | 600 optimizer steps/run; batch=32, seq_len=64, totaling 1,228,800 supervised bytes/fit; AdamW betas=(0.9,0.95), wd=0.1, 60-step warmup + cosine, gradient clipping=1.0, FP32; five paired report seeds |
| LR policy | Dedicated tuning seed83; select from `{0.01,0.02,0.04}` using only the V=256 control, then fix that LR for the V=4096 arm at the same depth. Selected T1=.01, T4=.04 |
| Evaluation | Fixed 16,384-byte validation sample; trajectory recorded at steps `[0,20,50,100,200,400,600]`; primary endpoint fixed at step600; test split never accessed |
| Loss decomposition | $`CE_{raw}`$ uses the 4096-way softmax; $`CE_{active}`$ renormalizes away dummy probability mass; $`CE_{competition}=CE_{raw}-CE_{active}=-\log(1-m_{dummy})`$, separating a mechanical denominator penalty from damage to active-token learning |
| Primary statistic | $`P_{active}(T)=CE_{active}(T,4096)-CE_{active}(T,256)`$; $`I_{active}=P_{active}(4)-P_{active}(1)`$. The original hypothesis requires $`I_{active}>0`$ |
| Results | $`P_{active}(1)=+0.02538`$, $`P_{active}(4)=+0.00096`$, hence $`I_{active}=-0.02442`$. Five-seed paired Student-t 95% CI `[-0.08525,+0.03640]`; four of five seed interactions are negative. Endpoint dummy mass is only about 1e-4, making raw and active interactions nearly identical |
| Interpretation | No evidence that recurrence amplifies the output-space inflation penalty. The point estimate is negative, but the CI crosses zero, supporting neither a significant negative interaction nor equivalence to zero. Under the preregistered triage rule, expensive Ouro-scale output-vocabulary CPT was stopped |
| Main limitations | T4 has worse absolute CE than T1 in this 600-step tiny-model regime, and its best LR hits the grid ceiling. This is not a miniature reproduction of mature Ouro/Claude; never-target classes are not equivalent to a real tokenizer vocabulary change |
| Actual compute | Six tuning fits + 20 primary fits; total GPU runtime 522.85 s on an RTX 5090. Its role is low-cost hypothesis triage, not estimation of frontier-scale effect sizes |

Together, these experiments address a narrow question: **can we find evidence that recurrent depth makes the LM-head / output-space bottleneck worse, providing a direct mechanism for Claude’s 16K vocabulary?** The current answer is no. They neither establish that softmax bottlenecks are absent in every LLM nor identify Claude’s actual design motivation. They lower the posterior of the mechanism we initially found most attractive.
