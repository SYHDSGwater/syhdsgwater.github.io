<callout color="gray_bg">
	**TL;DR: **我们起初猜测 Claude 把 vocabulary 从约 49K 压到约 16K，可能是在缓解 looped depth 下愈发严重的 LM-head / softmax bottleneck。后续两组实验没有支持这个故事，方向甚至相反。现在更可能的解释是：**更小的 vocabulary，更多的tokens，以此换取**
		- **更强的性能**
		- **更好看的财务数据**
	![](/images/claude-tokenizer/anthropic-arr-tokenizer-2026.svg)
</callout>
Claude Opus 4.7 之后的 tokenizer 是一个很反常的设计。根据 [Sander Land 对 Claude tokenizer 的逆向重建](https://www.tokenize.rs/claude)，它的 vocabulary 从此前大约 49K 大幅缩小到约 16K；Anthropic 自己也在 [token counting 文档](https://platform.claude.com/docs/en/build-with-claude/token-counting) 中明确提醒：Claude 4.7 及之后的模型使用新 tokenizer，同样输入文本会产生**约 30% 更多 tokens**。
我们在上一篇文章 <mention-page url="https://app.notion.com/p/3d06b290b3b4802a9863e5a11407b239"/> 里提出过一个更激进的解释：small vocab 可能与 recurrent / looped depth 协同，通过缩小输出空间缓解 softmax bottleneck。这个假说很漂亮，也足够可证伪。于是我们真的去测了它。结果并不支持。
<table_of_contents/>
# 1. 漂亮但失败的假说：small vocab 并没有缓解一个随 depth 增长的 bottleneck
经典 softmax bottleneck 来自最后一层的低秩约束。标准 LM head 写成
$$
z = Wh,\qquad W\in\mathbb{R}^{V\times D}
$$
把不同 context 的 hidden states 堆起来后，logit matrix 的 rank 满足 $`\mathrm{rank}(L)\le D`$。当 $`V\gg D`$ 时，一个很自然的担忧是：随着 backbone 变得更强、hidden state 承载更复杂的信息，固定宽度 $`D`$ 的 linear-softmax decoder 会越来越难把这些信息映射到一个巨大 vocabulary 上。2026 年 Godey & Artzi 又把这个问题推进到 backward pass：$`g_h=W^Tg_z`$ 只能把 vocabulary-space gradient 投回一个至多 $`D`$ 维的子空间，并报告 95–99% 的 logit-gradient norm 可能落在 LM-head feedback subspace 之外。[Lost in Backpropagation](https://arxiv.org/abs/2603.10145)
如果 Claude 又恰好采用 looped Transformer，这个故事会变得更诱人：recurrent depth 增加 latent compute，但 decoder rank 没变，于是可能出现一种 **depth–decoder mismatch**。small vocab 看起来就像一种合理补丁。
我们用 Ouro-1.4B 做了 EXP-001/001b。Ouro 有原生 recurrent depth，因此可以在同一个 checkpoint 上读取 $`T=1,2,3,4`$ 的 hidden states。除了原生 linear LM head，我们训练了一个额外的 rank-expanding residual head：
$$
z(h)=Wh+U\operatorname{GELU}(Ah)
$$
它额外增加了一组 nonlinear feature basis，因此 logit-family 的 rank upper bound 可以从约 $`D`$ 扩到 $`D+r`$。如果 deeper recurrent states 越来越受 linear-softmax 限制，那么 rich head 的边际收益应该随 $`T`$ 增长。
实际结果正好相反。EXP-001b 冻结 native $`W`$，分别加入 matched linear residual 与 nonlinear residual，得到：
<table fit-page-width="true" header-row="true">
<tr>
<td>Depth</td>
<td>$`G_{linear}`$</td>
<td>$`G_{nonlin}`$</td>
</tr>
<tr>
<td>T1</td>
<td>+0.3039</td>
<td>+0.0546</td>
</tr>
<tr>
<td>T2</td>
<td>+0.0111</td>
<td>+0.00283</td>
</tr>
<tr>
<td>T3</td>
<td>≈0</td>
<td>≈0</td>
</tr>
<tr>
<td>T4</td>
<td>0</td>
<td>0</td>
</tr>
</table>
这里 $`G_{linear}`$ 表示相对 frozen native head，额外 linear residual 能降低多少 CE；$`G_{nonlin}`$ 则是 matched nonlinear residual 相对 linear residual 的增益。我们发现甚至**整个 post-hoc decoder adaptation opportunity 都在消失**。原始 EXP-001 里完整重拟合一个约 100M 参数的 linear head，在 T2/T3/T4 上也没有得到 held-out CE gain；这些深度的最佳 validation checkpoint 全部出现在第一个更新点附近，native head 反而更好。对冻结 hidden states，multinomial logistic regression 关于 $`W`$ 本身是凸问题，因此这个现象很难简单归因于“非凸优化卡住”。更自然的解释是：随着 recurrent refinement 推进，hidden state 本身越来越接近 native LM head 已经适配好的 readout geometry。
我们最初预期
$$
T\uparrow\Rightarrow\text{decoder pressure}\uparrow
$$
实验观察更接近
$$
T\uparrow\Rightarrow\text{decoder adaptation opportunity}\downarrow
$$
这不证明一个普遍的“recurrent decoder alignment”定律，但至少在 Ouro 上，forward softmax bottleneck 没有随着 recurrent depth 恶化。[实验代码与结果](https://github.com/SYHDSGwater/depth-decoder-mismatch)
我们随后又做了 EXP-003C，使用 D=32、4-layer 的 tiny shared-body Transformer，固定 byte vocabulary 为 256，只在 output head 里加入 never-target classes，把 $`V_{out}`$ 从 256 扩到 4096，再比较 T1 与 T4。定义 active-vocabulary penalty
$$
P_{active}(T)=CE_{active}(T,4096)-CE_{active}(T,256)
$$
以及 interaction
$$
I_{active}=P_{active}(4)-P_{active}(1)
$$
如果 recurrence 会放大 output-space optimization burden，我们应该看到 $`I_{active}>0`$。实际 $`P_{active}(1)=+0.0254`$，$`P_{active}(4)=+0.0010`$，所以 $`I_{active}=-0.0244`$，5-seed 95% interval 为 $`[-0.0853,+0.0364]`$。它不是一个显著的负效应，也不能证明 equivalence to zero，但至少没有任何正向证据；5 个 seed 中 4 个方向为负。这个结果和 Murugan 2026 的 causal study 也一致：几何上的 gradient compression 确实存在，但把 never-target output classes 从 256 扩到 4096 并不会稳定损害学习。[Does the LM Head Create a Harmful Gradient Bottleneck?](https://arxiv.org/abs/2608.16671)
# 2. 更直接的反证：如果 softmax bottleneck 很严重，不可能十年没人改过 LM head
上文的两个实验存在一些不完善之处。但实验之外有一个更朴素的 revealed-preference evidence。从 GPT 早期架构到近两代的 open-weight frontier，attention、FFN、position encoding、normalization、MoE、routing、optimizer、KV cache 几乎都被重写过，但最后的
$$
h\rightarrow Wh\rightarrow\operatorname{softmax}
$$
仍然基本保留原样。Mixture of Softmaxes、DOC、Sigsoftmax 以及各种 nonlinear / high-rank output families 并不是没人提出；相反，softmax bottleneck 早在 RNN LM 时代就是一个成熟研究话题。真正反常的是：这些方法始终没有成为大模型标准件。
这条行业证据很强，因为 LM head 恰好是一个**最容易做局部 A/B 的部位**。换 attention 会改变 kernel、cache、并行和整条 serving stack；换一个 richer decoder 的工程风险小得多。如果一个 rank-expanding head 能在 frontier-scale pretraining 上稳定带来可观 loss gain，几乎无法解释为什么十年里所有大型实验室都错过它。我们看不到闭源 lab 的内部 ablation，但它们肯定系统测试过这个方向。更合理的推断是：经典 softmax rank constraint 在数学上存在，却通常**不是现代大模型的 binding constraint**；backbone 会在 end-to-end training 中主动学习一个对 linear head 友好的 representation geometry，而提高 output rank 的边际收益不足以抵消额外的参数、通信和 serving cost。
这也解释了为什么 $`V\gg D`$ 本身并不能推出“严重 bottleneck”。要构成严重 bottleneck，target conditional distribution 的 **effective log-probability rank** 需要足够大；vocabulary 的表面维度本身还不足以说明这一点。真实分布可以 full-rank，但如果 singular spectrum 衰减很快，一个 $`D`$ 维 approximation 已经足够覆盖绝大多数 CE-relevant structure。现代 Transformer 的 $`D`$ 又往往是数千甚至上万，和早期几百维 RNN LM 所处的 regime 已经完全不同。
所以，在重新审视 Claude 16K 时，我会把“为了缓解经典 softmax bottleneck”降到一个很低的 posterior。
# 3. 实验对比：Claude 16K 相比 o200k_base 到底多出多少 tokens？
在讨论 compute reallocation 之前，先直接看 Claude 新 tokenizer 和 OpenAI `o200k_base` 的实际差异。Playcode 在 2026 年用 **16 个真实 fixtures** 做了同字节输入对比：Claude 侧调用 Anthropic 官方 `count_tokens` endpoint，OpenAI 侧用 `tiktoken` 的 `o200k_base`，并用 GPT-5.1/5.5/5.6 Sol 的真实 API `usage` 做交叉验证。这个实验测的是 tokenizer 本身，不混入模型输出长度、reasoning budget 或 agent trajectory。
<table fit-page-width="true" header-row="true">
<tr>
<td>相同内容</td>
<td>Claude 4.7+ / o200k_base token ratio</td>
<td>直观含义</td>
</tr>
<tr>
<td>TypeScript</td>
<td>**1.73×**</td>
<td>Claude 多 73%</td>
</tr>
<tr>
<td>Rust</td>
<td>**1.58×**</td>
<td>Claude 多 58%</td>
</tr>
<tr>
<td>JavaScript</td>
<td>**1.52×**</td>
<td>Claude 多 52%</td>
</tr>
<tr>
<td>Python</td>
<td>**1.50×**</td>
<td>Claude 多 50%</td>
</tr>
<tr>
<td>HTML</td>
<td>**1.36×**</td>
<td>Claude 多 36%</td>
</tr>
<tr>
<td>English prose</td>
<td>**1.40×**</td>
<td>Claude 多 40%</td>
</tr>
<tr>
<td>Chinese prose</td>
<td>**1.44×**</td>
<td>Claude 多 44%</td>
</tr>
<tr>
<td>Chinese chat</td>
<td>**1.53×**</td>
<td>Claude 多 53%</td>
</tr>
</table>
来源：[The Same TypeScript Costs 73% More on Claude Than on GPT](https://playcode.io/blog/real-price-of-frontier-models)。其中一个具体样本更直观：同一个 2,888-character TypeScript file，`o200k_base` 为 **681 tokens**，Claude 新 tokenizer 为 **1,178 tokens**。Claude 新旧 tokenizer 自身的差异也集中在 English/code：同一组 fixtures 中 English prose +34%、TypeScript +31%、Rust +29%、agent system prompt +39%，而 Chinese prose 几乎不变。这说明 Anthropic 官方所说“约 30% 更多 tokens”只是 workload-average；对于 Claude 最重要的 coding/agent workload，和 `o200k_base` 的差距经常已经到 **1.5–1.7×**。
另一个独立的词表实验从不同角度得到相同方向。`novocab` 项目基于可精确计数的 Claude 4.7+ reconstruction 与 `o200k_base`，统计 running text 中一个完整 word 被 tokenizer 当作 single piece 的比例：English 为 **38.3% vs 88.1%**，code identifiers 为 **42.3% vs 88.9%**，11 个其他 Latin languages 的 median 为 **4.6% vs 67.9%**。[novocab tokenizer analysis](https://github.com/wdhwg001/novocab) 这不是性能 benchmark，却很直接地说明两个 tokenizer 的 inductive bias：`o200k_base` 大量把常见 lexical units 直接存进 vocabulary；Claude 4.7+ 更经常把同一个字符串拆成 reusable subpieces。
因此 16K 的第一阶效果并不抽象：**Claude 在相同文本上真的执行更多 token-level state transitions，尤其是在 code / agent 场景。** 接下来才值得问，这些额外 token computation 是否可能换来更好的模型能力，以及为什么 Anthropic 愿意承担它。
Anthropic 官方已经给了一个关键事实：Claude 4.7+ 的新 tokenizer 对同样文本平均产生约 **30% 更多 tokens**，并在当前 pricing 文档中直接写道，这个 tokenizer “contributes to improved performance on a wide range of tasks”。[Anthropic pricing](https://platform.claude.com/docs/en/about-claude/pricing) 这句话没有解释机制，但至少说明 Anthropic 明确愿意接受更细粒度 tokenization 带来的压缩率退化，以此换取性能。
从计算分配的角度看，它其实很直观。对于生成阶段，同样一段文本如果从 $`L`$ 个 tokens 变成大约 $`1.3L`$，那么 autoregressive decoder 至少要执行大约 1.3 倍的 sequential decode steps。每个 token 都获得一次新的 Transformer state update；如果模型内部还存在 recurrent depth，那么 sequence steps 与 latent-depth compute 还会进一步相乘。即使完全不假设 Claude 使用 looped Transformer，这个效果也成立：
$$
L_{token}\uparrow\quad\Rightarrow\quad\text{sequential Transformer compute}\uparrow
$$
输入侧虽然可以 parallel prefill，但更长 token sequence 同样意味着更多 representation slots、更高的 attention / FFN compute，以及更细粒度的 compositional interface。换句话说，small vocab 的一个直接结果，是把原本被 tokenizer “提前 merge” 掉的一部分组合工作重新交回 neural network。
对于几 T 参数量级的 frontier model，这里不需要讨论 LM-head 变窄节省了多少计算：它在整体 inference FLOPs 中只是很小的一部分。真正占主导的是更多 token 带来的完整 Transformer forward。
$$
\text{finer tokenization}\rightarrow\text{more Transformer forwards}\rightarrow\text{more inference compute}
$$
对于输出阶段，如果同样文本需要 1.4–1.7× tokens，就近似意味着 1.4–1.7× 次 autoregressive Transformer forward。每一步都要经过整套 attention、FFN/MoE 与其他 backbone computation；这才是 16K tokenizer 的主要计算代价。Anthropic 官方又明确表示新 tokenizer contributes to improved performance，因此一个更直接的解释是：**更多 token-level computation 本身就是 Claude 愿意支付的性能成本。**
# 4. 更有可能的解释：IPO时期，变相涨价让财务数据更好看
Claude的核心 API 、企业服务都是按 token 计费。Anthropic 的官方 token-counting 文档不仅写明新 tokenizer 让同样输入文本产生约 30% 更多 tokens，还明确提醒用户：**billing reflects this tokenizer's counts**。更关键的是，Anthropic 的 list price 显示，Opus 4.6 与 Opus 4.7、Opus 4.8、Opus 5的标准 API 标价都保持在 **\$5 / MTok input、\$25 / MTok output。**[Anthropic token counting](https://platform.claude.com/docs/en/build-with-claude/token-counting) / [Opus 4.8](https://www.anthropic.com/news/claude-opus-4-8)
因此，在最简单的近似下，如果一段固定文本在 4.7 tokenizer 下变成 1.3 倍 tokens，而每 million tokens 的价格没有同步下降，那么它对应的 billed units 也大约增加 30%。这不是隐藏规则，Anthropic 已经在 migration docs 里明确要求开发者重新测算成本。技术上，模型获得了更多 sequence-level compute；商业上，这部分新增 compute 自然映射到了客户看到的 token usage，Anthropic 无需独自吸收全部新增成本。
![](/images/claude-tokenizer/anthropic-arr-tokenizer-2026.svg)
**图：Anthropic 2026 年 ARR 与 tokenizer 归一化情景。** 橙线为官方及市场报道的年化收入运行率；青色虚线在 4 月 16 日 Opus 4.7 上线后按公开值 ÷ 1.30 折算。按 7 月底超过 650 亿美元的报道阈值计算，旧 tokenizer 单位等值约为 500 亿美元，差距约 23.1%。
这一情景假设上线后全部收入均受新 tokenizer 影响，并把官方针对同文本**输入侧**约 +30% 的 token 膨胀延伸到全部计费单位，固定文本量与单价。实际迁移率和收入构成未知；若受影响部分仅占当期公开收入的 q，折算系数应为 `(1 − q) + q / 1.30`。因此虚线是单位敏感性分析，不能当作财务重述或 tokenizer 对增长的因果贡献；切换处的断口也不代表收入下跌。
数据与口径：年初沿用 2025 年末约 90 亿美元；2/12 为 140 亿、4/6 超过 300 亿、5/28 公告称当月已超过 470 亿；7 月底超过 650 亿，由 8/17 报道披露。连线只作视觉引导，不外推 8–9 月。这里 ARR 指 annualized revenue run rate，而非已实现全年收入。来源：[Anthropic 2/12](https://www.anthropic.com/news/anthropic-raises-30-billion-series-g-funding-380-billion-post-money-valuation)、[4/6](https://www.anthropic.com/news/google-broadcom-partnership-compute)、[5/28](https://www.anthropic.com/news/series-h)、[Reuters 8/17](https://www.investing.com/news/stock-market-news/anthropic-revenue-run-rate-tops-65-billion-source-says-4864031)；[Opus 4.7 发布](https://www.anthropic.com/news/claude-opus-4-7)与[token counting 文档](https://platform.claude.com/docs/en/build-with-claude/token-counting)。
这形成了很自然的逻辑推测：
$$
\text{smaller vocab}\rightarrow\text{more tokens}\rightarrow\text{more Transformer steps}\rightarrow\text{higher billed token usage}
$$
所以我现在更愿意把 16K 理解成一次 **architecture × product economics co-design**。它允许 Anthropic 用更细的离散接口换更多 neural computation，同时仍然维持一个非常简单的“每百万 token 定价”产品界面。用户看到的是 nominal price 没变，系统内部发生的却是 token 计量单位本身发生了变化。
这里可以把商业逻辑再推进一步：**LLM inference 本身并不像很多人直觉中那样是一个接近成本价的薄利业务。** 2026 年 5 月 20 日流出的梁文锋投资者交流会录音转写里，他解释 DeepSeek API 的定价标准是“一批设备大约十个月收回成本”；随后又提到，在当前 AI efficiency 下“六倍利润”看起来高、其实并不高，未来即使下降到四倍、三倍仍然会有可观利润。他还强调 DeepSeek 的 B/C 端在线服务是 AGI 研发的副产品，API 只需很轻的维护团队就能形成数亿美元 ARR 的商业基础。[梁文锋投资者交流会完整转写](https://chinaresearchcollective.substack.com/p/liang-wenfeng-investor-meeting-may) 这当然不是 Anthropic 的 margin disclosure，DeepSeek 的成本结构也不能直接套到 Claude；但它提供了一个重要的行业参照：**frontier-model inference 的单位经济可能比外界想象得厚得多。**
在这种条件下，如果每个 billed token 已经具有稳定的正 contribution margin，那么在模型价格不变、用户需求没有同比例下降的情况下，同样内容被切成更多 tokens 会机械地增加 revenue，也会增加 profit，只是利润增加幅度取决于新增 Transformer compute 的边际成本。换句话说，关键变量是
$$
\Delta \Pi \approx \Delta N_{token}\times (P_{token}-C_{marginal\ token})
$$
只要 $`P_{token}>C_{marginal\ token}`$，更多 billed tokens 就对应更多 contribution profit。这里不需要依赖 LM-head 节省来闭合商业逻辑。对于 frontier model，更多 tokens 带来的完整 Transformer forward 才是主要新增成本；如果 inference 本身具有可观的正 contribution margin，那么 Anthropic 可以同时增加每次请求的计算量、提升模型能力，并因为 token-based billing 获得更多收入与利润。换句话说，16K 更像是一个愿意主动增加 inference compute、同时又让新增 compute 被计费体系覆盖的设计。
商业时点也让这个解释更值得注意。2026 年公开报道显示 Anthropic 正在推进可能规模巨大的 IPO，资本市场开始直接审视其 revenue growth、gross margin 与长期基础设施成本。[Reuters](https://www.reuters.com/technology/artificial-intelligence/ai-models-capabilities-leap-comes-with-new-safety-warnings-2026-09-09/) / [Financial Times](https://www.ft.com/content/9536c7b9-c600-48ec-8fe2-453b0ca187e9) 在这种阶段，一个同时能提高模型表现、增加 token usage、又不需要在价目表上显式“涨价”的 tokenizer 设计当然具有商业吸引力。**不过这一点仅代表我的个人观点，因为官方肯定不会同意变相涨价这种解释。**
# 5. 16K 还可能在优化什么？
还有一部分收益不能被“更多 forwards”解释。Claude 新 tokenizer 的逆向结果显示，它更接近 minimum-piece / PathPiece 风格，并显式利用 word-boundary state，设计上的改动超出了简单地把传统 BPE merge 数量砍到 16K。换句话说，Anthropic 同时在提高 vocabulary slot 的结构复用率：少记一些 `" token"`、surface variants 和长 lexical chunks，把更多组合交给模型完成。这个方向对代码 identifier、tool schema、URL、长尾字符串和 rare words 都有潜在价值，我们在上一篇文章已经详细讨论过。
但总之经过这轮实验后，我不会再把这些现象串成“16K 是为了修复 LM head”。一个更稳健、也更简单的叙事是：**Claude 正在主动减少 tokenizer 的 lexical memorization，把更多工作和计算留给 Transformer；更长的 token sequence 是性能成本，同时也是计费单位。** Anthropic 官方已经确认新 tokenizer 会产生约 30% 更多 tokens，并认为它参与了性能提升；我们的实验则没有发现 recurrent depth 会放大 softmax / output-space bottleneck。两条证据放在一起，比最初那个更漂亮的 bottleneck 故事更能解释现实。
最终我会把 Claude 16K 的设计概括为：16K 可能让 Transformer 工作得更多，而当 API 恰好按 token 收费时，这个技术选择还获得了一个非常自然的商业闭环。
<empty-block/>
---
# Appendix：实验配方与统计口径
完整代码、配置和逐 run 记录见 [depth-decoder-mismatch](https://github.com/SYHDSGwater/depth-decoder-mismatch)。
## A. EXP-001 / EXP-001b：recurrent depth 是否增加 decoder bottleneck？
<table fit-page-width="true" header-row="true">
<tr>
<td>项目</td>
<td>EXP-001</td>
<td>EXP-001b</td>
</tr>
<tr>
<td>核心问题</td>
<td>随着 Ouro recurrent depth 增加，rank-expanding rich head 相对重新拟合 linear head 的 gain 是否增大？</td>
<td>去掉 full-head refit 的 optimization drift 后，nonlinear residual 相对 matched linear residual 的 gain 是否仍随 depth 增大？</td>
</tr>
<tr>
<td>Backbone</td>
<td>`ByteDance/Ouro-1.4B`，24 shared layers，hidden size 2048，vocab 49,152，native recurrent T=1..4</td>
<td>同一 checkpoint、同一组缓存 hidden states</td>
</tr>
<tr>
<td>数据</td>
<td>FineWeb-Edu `sample-10BT`；1M primary run 共 31,250 documents，800k / 100k / 100k train/val/test targets；document split 先于 windowing</td>
<td>复用同一 1M cached-state dataset，因此是 post-hoc mechanistic audit，不是独立 replication</td>
</tr>
<tr>
<td>采样</td>
<td>seq_len=1024；每 document 取 predictor positions ≥128；32 targets/window；T1..T4 对同一 target 完全 paired</td>
<td>完全相同</td>
</tr>
<tr>
<td>Native head</td>
<td>$`z=W_{native}h`$，仅 diagnostic</td>
<td>$`W_{native}`$ 全程 frozen，作为所有 residual arms 的共同 base</td>
</tr>
<tr>
<td>Linear family</td>
<td>完整 bias-free $`Wh`$，从 native $`W`$ 初始化后重新拟合；约 100.7M trainable params</td>
<td>$`z=W_{native}h+U(Ah+b)`$；low-rank linear residual，width=512</td>
</tr>
<tr>
<td>Rich family</td>
<td>$`z=Wh+U\mathrm{GELU}(Ah)`$；额外 nonlinear feature basis，理论 logit-rank family 从约 $`D`$ 扩到 $`D+r`$</td>
<td>$`z=W_{native}h+U\mathrm{GELU}(Ah+b)`$；与 linear residual 相同 residual shape / parameter count</td>
</tr>
<tr>
<td>优化</td>
<td>FP32 AdamW，wd=0，batch=256，constant LR=1e-4；probe seeds 11/29/47；best validation checkpoint 再做单次 test eval</td>
<td>FP32 AdamW，wd=0，batch=256；LR grid `{3e-5,1e-4,3e-4}`，seed11 tuning；report seeds 29/47/83；固定 2000-step budget；step0 纳入 model selection，early validation 更密</td>
</tr>
<tr>
<td>Primary metric</td>
<td>$`R_{head}(T)=CE^*_{linear}(T)-CE^*_{rich}(T)`$；DDM 预测 $`dR/dT>0`$</td>
<td>$`G_{nonlin}(T)=CE_{linear\ residual}(T)-CE_{nonlinear\ residual}(T)`$</td>
</tr>
<tr>
<td>关键结果</td>
<td>1M run：T1 regret +0.0779，T2/T3/T4 ≈0；slope = **-0.02340**，95% document-bootstrap CI `[-0.02407,-0.02275]`。完整 linear refit 在 T2–T4 也没有 held-out gain，全部最佳 checkpoint≈step1</td>
<td>$`G_{nonlin}`$ = **0.0546 → 0.00283 → ≈0 → 0**；$`G_{linear}`$ = **0.3039 → 0.0111 → 0 → 0**；T4 的所有 residual arms 最佳都是 step0</td>
</tr>
<tr>
<td>能支持的结论</td>
<td>在 Ouro 的 native recurrent depths 上，没有观察到“depth 越深，linear-softmax decoder 越成为 forward bottleneck”；相反，post-hoc decoder adaptation opportunity 随 depth 快速消失，更符合 decoder alignment / iterative refinement。</td>
<td></td>
</tr>
<tr>
<td>主要 limitation</td>
<td>full-W refit 极度 overparameterized；pilot/1M 是 sequential scaling 而非独立 confirmatory replication</td>
<td>复用已看过的 test set；low-rank residual 只覆盖一类 richer decoder；结果不能推出所有模型都不存在 softmax bottleneck</td>
</tr>
</table>
## B. EXP-003C：recurrent depth 是否放大 output-vocabulary inflation 的训练伤害？
<table fit-page-width="true" header-row="true">
<tr>
<td>项目</td>
<td>配方</td>
</tr>
<tr>
<td>核心问题</td>
<td>Murugan 的 never-target output-class null 在加入 weight-shared recurrence 后是否会翻转？即 larger $`V_{out}`$ 是否在 T4 比 T1 更伤 active-token learning？</td>
</tr>
<tr>
<td>模型</td>
<td>Murugan-style tiny Transformer：D=32，4 layers，4 heads，FFN=128，pre-LN，GELU，zero dropout；完整 4-layer body 作为一个 shared recurrent block 重复 T 次；absolute position 只在初始输入加入一次，final LayerNorm 只在最终 exit 使用</td>
</tr>
<tr>
<td>Tokenizer / active vocab</td>
<td>WikiText-2 raw bytes；identity byte IDs 0..255；输入和合法 target vocabulary 永远固定 256</td>
</tr>
<tr>
<td>2×2 intervention</td>
<td>T ∈ \{1,4\} × $`V_{out}`$ ∈ \{256,4096\}；4096 arm 额外加入 3840 个 trainable、never-input、never-target output rows</td>
</tr>
<tr>
<td>Pairing</td>
<td>同一 seed 下四个 arms 的 embedding、recurrent body 与 active output rows 初始值完全相同；batch order paired；4096 arm 只追加 dummy rows</td>
</tr>
<tr>
<td>训练 objective</td>
<td>只在最终 recurrent state 上计算一次 **raw full-softmax CE**；T4 不加 intermediate-loop loss，避免监督次数与 depth 混杂</td>
</tr>
<tr>
<td>训练预算</td>
<td>600 optimizer steps / run；batch=32，seq_len=64，即 1,228,800 supervised bytes / fit；AdamW betas=(0.9,0.95)，wd=0.1，60-step warmup + cosine，grad clip=1.0，FP32；5 paired report seeds</td>
</tr>
<tr>
<td>LR policy</td>
<td>dedicated tuning seed83；只在 V=256 control 上从 `{0.01,0.02,0.04}` 选 LR，再把该 LR 固定给同 depth 的 V=4096 arm。最终 T1=.01，T4=.04</td>
</tr>
<tr>
<td>Evaluation</td>
<td>固定 16,384-byte validation sample；step `[0,20,50,100,200,400,600]` 记录 trajectory；primary endpoint 固定为 step600；未访问 test split</td>
</tr>
<tr>
<td>Loss decomposition</td>
<td>$`CE_{raw}`$ 使用 4096-way softmax；$`CE_{active}`$ 将 dummy probability mass renormalize 掉；$`CE_{competition}=CE_{raw}-CE_{active}=-\log(1-m_{dummy})`$，从而区分 mechanical denominator penalty 与 active-token learning damage</td>
</tr>
<tr>
<td>Primary statistic</td>
<td>$`P_{active}(T)=CE_{active}(T,4096)-CE_{active}(T,256)`$；$`I_{active}=P_{active}(4)-P_{active}(1)`$。原假说要求 $`I_{active}>0`$</td>
</tr>
<tr>
<td>结果</td>
<td>$`P_{active}(1)=+0.02538`$，$`P_{active}(4)=+0.00096`$；因此 $`I_{active}=-0.02442`$。5-seed paired Student-t 95% CI `[-0.08525,+0.03640]`；seed interactions 4/5 为负。endpoint dummy mass 仅约 1e-4，因此 raw 与 active interaction 几乎相同</td>
</tr>
<tr>
<td>解释</td>
<td>没有发现 recurrence 放大 output-space inflation penalty 的证据；point estimate 反而为负，但 CI 跨零，因此不能宣称显著 negative interaction 或 equivalence-to-zero。按 preregistered triage rule，停止昂贵的 Ouro-scale output-vocab CPT</td>
</tr>
<tr>
<td>主要 limitation</td>
<td>T4 在 600-step tiny-model regime 下 absolute CE 比 T1 更差，且最佳 LR 命中 grid 上界，因此它不是 mature Ouro/Claude 的 miniature reproduction；never-target classes 也不等价于真实 tokenizer vocab change</td>
</tr>
<tr>
<td>实际 compute</td>
<td>6 个 tuning fits + 20 个 primary fits，总 GPU runtime 522.85 s（RTX 5090），因此它的角色是低成本 hypothesis triage，而非 frontier-scale effect-size estimate</td>
</tr>
</table>
这两组实验共同回答的是一个很窄的问题：**我们是否能找到证据，证明 recurrent depth 会让 LM-head / output-space bottleneck 更严重，从而为 Claude 16K 提供直接机制解释？** 当前答案是否定的。它们没有证明 softmax bottleneck 在所有 LLM 上不存在，也没有证明 Claude 16K 的真实设计动机；它们只是把原先最吸引人的机制假说降到了很低的 posterior。
<empty-block/>
