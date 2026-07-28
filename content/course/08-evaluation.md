# 第 8 章：reward 上升以后，怎样证明模型真的变好了

## 8.1 本章符号说明

| 符号 | English | 本章含义 |
|---|---|---|
| \(n\) | sampled candidates | 每道题实际生成的候选数 |
| \(c\) | correct candidates | 其中通过 verifier 的候选数 |
| \(k\) | attempt budget | 评估允许使用的候选数 |
| \(\widehat p\) | empirical success rate | 单次采样成功率的样本估计 |
| \(\Delta\) | paired improvement | 同一题上新旧模型指标之差 |
| CI | confidence interval | 由有限评估样本带来的不确定范围 |

## 8.2 本章目标

读完本章，你应该能够：

1. 区分训练 reward、验证 reward 与真实目标；
2. 为 RL 实验建立不可被训练污染的 baseline 与数据划分；
3. 将 parse rate、条件正确率、端到端正确率分开报告；
4. 正确解释 pass@1、pass@k 与采样预算；
5. 用配对评估、置信区间和多随机种子报告不确定性；
6. 设计能回答因果问题的 ablation；
7. 系统检查 reward hacking 与通用能力回退。

## 8.3 本章主线

第 7 章保证：

> token 对齐、mask、ratio 和 reward 管线至少通过了已知测试。

但训练日志里的 `mean_reward` 上升，只能说明模型越来越擅长优化**当前实现的 scorer**。它不自动推出：

- 未见题上也更正确；
- 在相同采样预算下更好；
- 推理过程更可靠；
- 原有语言能力没有退化；
- 没有利用 parser 或测试漏洞。

因此评估必须从结论倒推证据：

\[
\text{声称模型更好}
\rightarrow
\text{定义“好”}
\rightarrow
\text{锁定评估协议}
\rightarrow
\text{与正确 baseline 配对比较}
\rightarrow
\text{报告不确定性与失败样本}.
\]

## 8.4 本章新增概念

| 名词 | 中文直觉 | 本章中解决的问题 |
|---|---|---|
| end-to-end metric | 从原始输出到任务结果的总成功率 | 不让解析失败从分母中消失 |
| conditional accuracy | 已成功解析样本中的正确率 | 区分格式问题与推理问题 |
| pass@k | \(k\) 次尝试至少一次成功 | 衡量采样搜索能力 |
| paired evaluation | 同一批题比较两个模型 | 消除题目难度差异造成的噪声 |
| confidence interval | 有限测试集下的可能波动范围 | 防止把小随机差异写成结论 |
| ablation | 只改变一个因素的对照 | 判断改进究竟来自哪一部分 |
| reward hacking | 提高 scorer 而未完成真实目标 | 检查优化器是否利用规则漏洞 |

## 8.5 先写清楚你要证明的命题

一个可检验的实验命题应类似：

> 在固定模型规模、SFT 初始化、训练题量和生成预算下，使用 group-relative RLVR 训练后，held-out 数学题的 greedy exact-match 从 X 提升到 Y，同时通用问答回退不超过 Z。

它明确：

- 变了什么：训练方法；
- 固定什么：模型、数据、预算；
- 在哪里测：held-out；
- 用什么解码：greedy；
- 主指标是什么：exact-match；
- 允许什么代价：通用能力回退上限。

“reward 从 0.3 涨到 0.7”不是完整命题，因为 scorer、采样温度、题目集合和回答长度都可能同时变化。

## 8.6 数据划分必须先于训练

至少分成：

| 集合 | 用途 | 能否用于选择 checkpoint/超参数 |
|---|---|---:|
| train | rollout 与梯度更新 | 是 |
| validation | 早停、超参数、reward 设计迭代 | 是 |
| test | 最终一次性报告 | 否 |

对程序生成数学题，随机拆分仍可能泄漏模板。更严格的划分应按：

- 题型模板；
- 数值范围；
- 推理深度；
- 数据来源；
- 时间；
- 工具/API 版本。

若 train 与 test 只更换数字，结论是“学会模板泛化”，不能夸大为一般推理能力。

## 8.7 baseline 必须使用完全相同的评估协议

至少比较：

1. 初始 SFT/reference checkpoint；
2. 训练步数为 0 的同代码路径；
3. RL 后 checkpoint；
4. 必要时相同数据量的额外 SFT 或 rejection sampling baseline。

所有模型必须共享：

- prompt 模板；
- tokenizer 与停止条件；
- max tokens；
- temperature、top-p；
- 每题采样数；
- parser 与 verifier 版本；
- 测试题顺序。

否则提升可能只来自更大的采样预算或更宽松的 parser。

## 8.8 把一个“准确率”拆成三个数

设总测试题数为 \(M\)：

\[
\text{parse rate}
=
\frac{\#\text{成功解析}}
{M},
\]

\[
\text{conditional accuracy}
=
\frac{\#\text{解析且正确}}
{\#\text{成功解析}},
\]

\[
\text{end-to-end accuracy}
=
\frac{\#\text{解析且正确}}
{M}.
\]

例如：

| 模型 | parse rate | 解析后正确率 | 端到端正确率 |
|---|---:|---:|---:|
| baseline | 80% | 75% | 60% |
| RL model | 99% | 64% | 63% |

端到端提升了 3 个点，但真正变化主要来自格式，而不是推理。只报告 63% 会掩盖这个事实；只在可解析样本上报告 64% 又会得到相反印象。

## 8.9 pass@1 与 pass@k 回答不同问题

### 8.9.1 pass@1

一次生成成功的比例。它最接近真实单次调用质量，但必须注明解码策略：

- greedy pass@1；
- temperature sampling pass@1；
- 固定 seed 或多 seed 平均。

### 8.9.2 pass@k

每题允许 \(k\) 次独立尝试，只要至少一个成功就算通过。若单次成功率为 \(p\)，理想独立条件下：

\[
\operatorname{pass@}k
=
1-(1-p)^k.
\]

它衡量“模型加采样搜索”的系统能力，不等于单次回答更可靠。比较 pass@k 时必须固定 \(k\)、temperature 和总 token 预算。

当每题已采样 \(n\) 个候选，其中 \(c\) 个正确，无放回估计为：

\[
\widehat{\operatorname{pass@}k}
=
1-
\frac{
\binom{n-c}{k}
}{
\binom{n}{k}
},
\quad n\ge k.
\]

若 \(c=0\)，估计为 \(0\)；若 \(n-c<k\)，至少会抽到一个正确答案，估计为 \(1\)。

### 8.9.3 还要报告效率

同样的 pass@k，模型可能生成更长回答、消耗更多算力。建议同时报告：

- 平均 completion tokens；
- 成功样本的 tokens；
- 每题 wall-clock；
- verifier/tool 调用次数；
- 每个成功样本的平均计算成本。

## 8.10 用配对比较减少噪声

不要让 baseline 测一批题、RL model 测另一批题。对同一个测试题 \(j\)，记录：

\[
\Delta_j
=
m_{\text{new},j}
-
m_{\text{base},j}.
\]

总体提升：

\[
\bar\Delta
=
\frac1M
\sum_{j=1}^{M}\Delta_j.
\]

配对设计自动抵消“某批题更难”的波动。对采样评估，还应：

- 对同一 prompt 使用预先固定的 seed 集；
- 或为每题采足够候选后计算统一的 pass@k；
- 对不同训练 seed 分别训练和报告。

## 8.11 不确定性不能省略

测试集只有有限样本。若 200 道题从 60% 升到 62%，可能只是几道题的随机变化。

可使用：

- 二项比例 Wilson interval；
- 按题 bootstrap；
- 配对 bootstrap 估计 \(\bar\Delta\) 的 CI；
- 多训练 seed 的均值与标准差。

推荐至少报告：

```text
end-to-end accuracy: 63.2%
paired improvement: +3.1 percentage points
95% bootstrap CI: [+1.0, +5.2]
training seeds: 3
```

CI 若跨过 \(0\)，正确结论是“当前证据不足以确认提升”，不是“几乎有效”。

## 8.12 ablation 要对应一个因果问题

对 GRPO/RLVR，可以设计：

| 对照 | 只改变什么 | 回答的问题 |
|---|---|---|
| \(G=4\) vs \(G=16\) | group size | 改进是否来自更稳的组 baseline/更多探索 |
| 无标准化 vs 标准化 | advantage scale | 组内缩放是否必要 |
| 单次更新 vs 多 epoch | rollout reuse | 提升是否来自样本复用 |
| 无 clipping vs clipping | trust region surrogate | 稳定性是否来自裁剪 |
| \(\beta=0\) vs \(\beta>0\) | reference KL | 通用能力与格式稳定是否靠 KL 保持 |
| task reward only vs + format reward | shaping | 提升究竟来自正确率还是格式 |
| random reward | 反馈语义 | 管线是否会“无论什么 reward 都上涨” |

每次只改一个关键因素，并尽量匹配：

- optimizer steps；
- rollout tokens；
- 总训练 FLOPs；
- 验证频率。

否则不能把结论归因给算法结构。

## 8.13 reward hacking 的系统检查

### 8.13.1 parser hacking

- 输出多个 `\boxed{}`，让 parser 选中有利答案；
- 在答案附近插入特殊字符；
- 利用正则的贪婪/非贪婪差异；
- 输出 NaN、无穷或数值溢出。

### 8.13.2 verifier hacking

- 只通过公开测试；
- 利用浮点容差；
- 访问不该读取的文件或网络；
- 修改环境状态而非完成任务。

### 8.13.3 reward composition hacking

- 用极长文本积累格式分；
- 通过拒答规避负分；
- 重复关键词获取 RM 偏好；
- 牺牲主任务换取容易优化的辅助项。

### 8.13.4 检查方法

1. 保存 reward 最高与提升最快的样本；
2. 用独立 parser/verifier 复核；
3. 对隐藏测试、扰动格式、同义 prompt 重测；
4. 人工盲评随机样本与高 reward 样本；
5. 将 reward 各分量与真实主指标分别画图；
6. 对失败样本建立类型表，而不是只看均值。

## 8.14 通用能力与行为回退

reference KL 小不等于所有能力都被保留。训练特定数学任务后，还应根据用途检查：

- 通用知识与指令遵循；
- 安全拒答；
- 多语言能力；
- 输出多样性与重复；
- 校准与不确定性表达；
- 长上下文；
- 与工具无关的正常聊天。

建立 regression suite，并在训练前锁定最低允许阈值。若主任务 +3%，通用能力 -10%，这不是无条件的成功。

## 8.15 一页实验报告模板

```text
目标命题
  在相同 1× 采样预算下提高 held-out 算术准确率。

固定条件
  base checkpoint / prompt template / decoding / parser / verifier / token budget

训练变化
  GRPO, G=8, epsilon=0.2, beta=0.01, 2 update epochs

主结果
  greedy end-to-end accuracy
  baseline: 60.1%
  trained:  64.0%
  paired Δ: +3.9 pp, 95% CI [+2.0, +5.8]

拆分指标
  parse rate / conditional accuracy / answer length / KL / zero-variance groups

消融
  no KL / no clipping / G=2 / random reward

回退
  general instruction suite: -0.4 pp

失败分析
  arithmetic slip 42%, parse failure 18%, premature stop 15%, other 25%

结论边界
  只支持同模板外数值泛化；尚未证明跨题型推理提升。
```

## 8.16 贯穿案例的可信结论

对 `23×17` 类任务，下面两种说法区别很大：

不充分：

> 训练 reward 从 0.4 升到 0.9，所以模型学会了数学推理。

更可信：

> 在按运算结构隔离的 2,000 道 held-out 题上，固定 greedy 解码与 128-token 预算后，端到端准确率从 61.3% 提升到 67.8%；提升同时来自 parse rate +1.2 pp 和解析后正确率 +5.8 pp。配对 bootstrap 95% CI 为 [+4.7,+8.3] pp，三个训练 seed 均为正；独立 verifier 复核未发现明显格式漏洞。该结果只证明当前题型分布上的单次回答改进。

科学性主要来自限定了结论，而不是用了更复杂的指标。

## 8.17 本章自测

1. 为什么训练 reward 上升不能直接证明 held-out 正确率上升？
2. parse rate、conditional accuracy 与 end-to-end accuracy 分别揭示什么？
3. pass@k 提高可能来自哪些与单次模型质量无关的因素？
4. 为什么相同题目的配对比较比两批独立题更稳？
5. 设计一个只检验 reference KL 作用的 ablation。
6. reward 最高的样本为什么应该优先人工检查？

## 8.18 本章之后还缺什么

目前的贯穿案例是单轮问答：

```text
prompt → 连续生成 token → 最终 verifier
```

工具 Agent 会多次与外部环境交互：

```text
观察 → 选择工具 → 环境变化 → 新观察 → 再决策
```

第 9 章不会重新发明一套强化学习，而是逐项替换第 1 章的状态、动作、转移和 reward，并说明工具错误、安全约束与长程信用分配怎样进入现有框架。

## 8.19 对应资料

- [Evaluating Large Language Models Trained on Code](https://arxiv.org/abs/2107.03374)
- [HELM: Holistic Evaluation of Language Models](https://arxiv.org/abs/2211.09110)
- [RewardBench](https://arxiv.org/abs/2403.13787)
- [Sutton & Barto, Section 2.4: Incremental Implementation](http://incompleteideas.net/book/RLbook2020.pdf)
