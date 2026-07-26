# 第四周：从一次成功训练到可信实验与 Agent RL

> 最后一周不再增加新的算法缩写。目标是把“模型跑起来了”升级为“我知道它为什么有效、何时失效、能否迁移到业务环境”。

## 本周最终产出

完成一份一页实验报告，至少包含：

- 明确假设与单变量消融；
- train reward、held-out task metric、KL、clip fraction 和长度；
- 多个 seed 或不确定性说明；
- 失败样例与 reward hacking 分析；
- 计算成本与样本效率；
- 把 verifier 替换成业务环境时的风险清单。

整周主线：

```text
建立可复现实验基线
→ 做有因果解释的消融
→ 区分训练 reward 与真实能力
→ 主动攻击 verifier
→ 将同一框架迁移到工具调用和 Agent RL
```

---

# 第一章：先定义“实验成功”

## 1.1 不要用训练 reward 定义成功

训练优化的是代理目标：

\[
\max_\theta
\mathbb E_{\pi_\theta}[r_{\mathrm{train}}(x,y)].
\]

真正关心的是：

\[
\max_\theta
\mathbb E_{(x,y^*)\sim\mathcal D_{\mathrm{test}}}
[m_{\mathrm{real}}(x,y,y^*)].
\]

当 \(r_{\mathrm{train}}\neq m_{\mathrm{real}}\) 时，reward 上升不保证能力上升。

必须同时报告：

| 层级 | 示例 | 用途 |
|---|---|---|
| 训练代理 | shaped reward | optimizer 直接看到 |
| 任务指标 | held-out exact match | 判断任务能力 |
| 行为约束 | KL、长度、格式、安全率 | 判断副作用 |
| 定性证据 | 生成样例、失败分类 | 解释模型学到了什么 |

## 1.2 明确实验假设

坏问题：

> group size 会怎样？

可检验假设：

> 在固定总 completion 预算下，group size 从 4 增至 8 会降低全同 reward 组比例，从而改善早期 advantage 信噪比；但 prompt 多样性下降，收益可能在后期饱和。

每个实验都应写：

```text
改变什么
保持什么不变
预期机制
支持该机制的指标
可能的替代解释
```

---

# 第二章：为 math-hard 建立可信基线

## 2.1 基线先于 RL

在任何训练前记录：

- 固定 test subset 的 exact match；
- train prompt rollout reward；
- 输出长度、格式解析成功率；
- 每题 \(G\) 次采样的 pass@1 与 pass@G；
- 高置信错误样例。

如果 baseline 本身异常，后续提升没有解释价值。

## 2.2 parser 是评估系统的一部分

对数学任务：

```python
raw_text
→ extract boxed answer
→ canonicalize
→ compare with target
```

分别记录：

```text
parse success
semantic exact match conditional on parse success
overall exact match
```

否则“格式失败”和“数学错误”会混成一个数字。

## 2.3 pass@k 与训练 reward

单次正确概率为 \(p\)，独立采样 \(k\) 次至少一次成功的概率：

\[
\operatorname{pass@}k
=1-(1-p)^k.
\]

pass@k 上升可能只是分布尾部仍包含正确答案，不代表 greedy/pass@1 提升。RL 训练通常希望把成功轨迹从尾部搬到高概率区域，因此要同时看：

- temperature sampling pass@k；
- greedy 或低温 pass@1；
- 平均 log-prob 与答案多样性。

---

# 第三章：四个最有解释力的消融

## 3.1 Group size

### 机制

更大的 \(G\) 为同 prompt baseline 提供更多样本：

\[
\hat A_{i,j}
=\frac{r_{i,j}-\bar r_i}
{s_i+\varepsilon}.
\]

### 预期

- 优点：更可能同时看到成功与失败，降低全同 reward 组比例；
- 缺点：固定总 rollout 数时，prompt 多样性下降；
- 成本：生成长度近似相同时，rollout 计算随 \(G\) 线性增长。

### 应看指标

```text
fraction_all_correct_groups
fraction_all_wrong_groups
advantage_std
unique_prompts_per_update
eval accuracy per generated token
```

## 3.2 KL 系数 \(\beta\)

目标近似：

\[
\mathcal L
=\mathcal L_{\mathrm{policy}}
+\beta D_{\mathrm{KL}}
(\pi_\theta\|\pi_{\mathrm{ref}}).
\]

| \(\beta\) | 预期行为 | 风险 |
|---|---|---|
| 太小 | reward 学得快、KL 大 | reward hacking、能力遗忘、风格漂移 |
| 合适 | task metric 与 reward 同步上涨 | 需要根据模型和任务调节 |
| 太大 | KL 很低 | 策略几乎不学习 |

不要只比较最终 reward；画出 `eval metric vs KL`，寻找相同偏移预算下更有效的设置。

## 3.3 PPO epochs 与 clip \(\epsilon\)

更多 epoch 提高同一批 rollout 的样本效率，但数据更陈旧。

更大的 \(\epsilon\) 允许 ratio 离 \(1\) 更远：

\[
\rho_t\in[1-\epsilon,1+\epsilon].
\]

联动解释：

| 现象 | 解释 |
|---|---|
| epochs 增加，eval 提升且 KL 可控 | 有效数据复用 |
| epochs 增加，reward 升而 eval 降 | 过拟合 rollout/verifier |
| clip fraction 长期接近 0 | 更新很小或 clip 基本没作用 |
| clip fraction 很高且 KL 大 | 更新过猛 |
| clip fraction 很高但参数变化小 | 大量梯度被截断 |

## 3.4 Reward shaping 与长度限制

总 reward：

\[
r_{\mathrm{total}}
=r_{\mathrm{correct}}
+\alpha r_{\mathrm{format}}
-\lambda_{\mathrm{len}}\operatorname{length}(y).
\]

每个 shaping 项都会创造新的攻击面。

必须报告 reward 分量，而不是只报告和：

```text
correct_reward
format_reward
length_penalty
total_reward
```

---

# 第四章：用受控实验比较 GRPO 与 GR-REINFORCE

## 4.1 公平比较

保持一致：

- 模型初始化；
- prompt batch 与 group size；
- 每轮生成的 completion 数；
- 学习率、KL 系数与最大长度；
- seed 与评估集。

主要区别：

```text
GR-REINFORCE：每批 rollout 只产生一次 on-policy 更新
GRPO：同批 rollout 做少量 clipped PPO epoch
```

## 4.2 两种效率

**样本效率**：

\[
\frac{\text{eval improvement}}
{\text{generated completion tokens}}.
\]

**计算效率**：

\[
\frac{\text{eval improvement}}
{\text{GPU hours or FLOPs}}.
\]

GRPO 可能用相同 rollout 获得更高样本效率，但多次 forward/backward 会增加优化计算。只报告 step 数会掩盖这一点。

## 4.3 建议图表

至少画：

1. eval exact vs generated tokens；
2. eval exact vs wall-clock/GPU time；
3. train reward 与 eval exact 同图；
4. KL、clip fraction 与 gradient norm；
5. response length 与 parse success；
6. 全错 group 比例。

---

# 第五章：系统化发现 reward hacking

## 5.1 四类漏洞

### 解析器漏洞

- 输出多个答案，让 parser 取到有利的一个；
- Unicode、科学计数法、前导零或溢出；
- 在标签后追加指令；
- 利用解析失败的默认值。

### 代理目标漏洞

- 反复输出格式 token 获得 shaping reward；
- 用更长回答换取模型 judge 偏好；
- 只优化容易样本，牺牲困难样本；
- 输出记忆模板而非真正推理。

### 学到的 reward model 漏洞

- 迎合措辞、谄媚和固定风格；
- 利用训练分布中的长度/礼貌相关性；
- 生成分布外文本使 RM 过度自信；
- 与 judge 模型共享偏差。

### 环境与工具漏洞

- 绕过 API、伪造工具输出；
- 修改测试文件而不是修复代码；
- 重试直到偶然成功；
- 使用未授权信息或外部状态。

## 5.2 红队 verifier

在训练前建立攻击集：

```text
正确但格式不同
格式正确但答案错误
多个互相矛盾的答案
极长回答
提示注入
Unicode/空白/编码边界
超时、异常、工具失败
能通过旧 parser 但语义错误的样例
```

训练后再次运行同一攻击集，并新增模型实际发现的漏洞。

## 5.3 双通道评估

```text
训练通道：便宜、快速、可微或可大量调用的 verifier
审计通道：独立实现、较慢、不可被训练直接查询
```

两者差距扩大是 reward hacking 的早期信号。

---

# 第六章：不要忽略统计不确定性

## 6.1 二项指标的标准误

exact match 在 \(n\) 个样本上的估计为 \(\hat p\)，近似标准误：

\[
\operatorname{SE}(\hat p)
\approx
\sqrt{\frac{\hat p(1-\hat p)}{n}}.
\]

例如 \(\hat p=0.40,n=100\)：

\[
\operatorname{SE}\approx
\sqrt{\frac{0.4\times0.6}{100}}
\approx0.049.
\]

提升两个百分点很可能只是噪声。

## 6.2 配对评估

比较两个 checkpoint 时，使用同一批 prompt 和尽量一致的采样设置。记录每题的成功/失败差异，再做 paired bootstrap，比比较两个独立均值更有统计效率。

## 6.3 Seed

至少区分：

- 数据/mini-batch seed；
- rollout sampling seed；
- 模型初始化或 adapter seed；
- 评估 sampling seed。

预算有限时，与其只跑一个超长实验，不如先跑多个较短 seed 判断趋势是否稳定。

---

# 第七章：把 verifier 换成 Agent 环境

## 7.1 LLM Agent 的 MDP

以代码修复 Agent 为例：

| RL 元素 | Agent 对应物 |
|---|---|
| 状态 \(s_t\) | issue、仓库、已读文件、工具返回、历史消息 |
| 动作 \(a_t\) | 文本、搜索、编辑、运行测试、提交 |
| 转移 \(P\) | 工具或环境执行动作后的新状态 |
| reward | 测试通过、静态检查、成本、安全约束 |
| 轨迹 | 从接收任务到最终补丁的全部交互 |

与单轮数学题相比，Agent RL 更难：

- 状态部分可观察；
- 工具可能失败或非确定；
- 轨迹更长，credit assignment 更困难；
- 动作具有真实副作用；
- reward 更容易被环境漏洞利用。

## 7.2 业务任务设计模板

在训练前填写：

```text
任务：
初始状态：
允许动作：
禁止动作：
成功条件：
失败条件：
成本：
终止条件：
训练 verifier：
独立审计器：
可能的投机方式：
恢复/回滚方式：
```

## 7.3 Reward 应包含成本与安全

示例：

\[
r=
r_{\mathrm{success}}
-\lambda_c\cdot\text{tool cost}
-\lambda_t\cdot\text{latency}
-\lambda_s\cdot\text{safety violation}.
\]

但不要以为加惩罚就自动安全。对于不可逆操作，应通过权限和环境隔离硬约束，而不是仅靠负 reward。

## 7.4 从离线到在线的安全顺序

```text
离线轨迹回放
→ 沙箱中的 imitation / preference optimization
→ 可回滚环境中的小规模在线 RL
→ 人工审计的灰度任务
→ 才考虑真实业务动作
```

---

# 第八章：一页实验报告模板

## 问题与假设

一句话说明改变什么、为什么可能有效。

## 设置

```text
model / adapter
dataset split
algorithm
batch × group size
learning rate
KL coefficient
clip epsilon
PPO epochs
max length
seed
generated tokens / GPU hours
```

## 主要结果

| Run | Eval | Train reward | KL | Clip frac | Length | Cost |
|---|---:|---:|---:|---:|---:|---:|
| baseline | | | | | | |
| ablation A | | | | | | |

## 机制证据

展示能支持原假设的中间指标，而不只是最终分数。

## 失败样例

至少给出一条高 reward 失败、一条低 reward 但实际合理的回答。

## 结论与限制

说明结果能支持什么、不能支持什么，以及下一步最小实验。

---

# 第九章：完成检查

1. 为什么 train reward 不能定义实验成功？
2. 固定总 rollout 预算时，增大 group size 有什么交易？
3. KL 系数过小和过大分别会怎样？
4. 为什么要同时报告样本效率和计算效率？
5. 怎样设计独立审计通道？
6. 二项 exact match 的不确定性如何估计？
7. Agent RL 为什么不能只靠负 reward 约束危险动作？
8. 一个合格消融必须保持哪些变量不变？

<details>
<summary><strong>展开查看答案要点</strong></summary>

1. 训练 reward 是可被优化的代理，可能与 held-out 真实指标错位。
2. baseline 估计和成功/失败混合概率改善，但 prompt 多样性下降，生成成本按 group 增长。
3. 太小会漂移和投机；太大会阻止学习。
4. 数据复用可能提高每 token 收益，却增加每批优化 FLOPs。
5. 使用训练时不可直接查询、不同实现或不同模型的评估器，并保留对抗集。
6. 可用 \(\sqrt{\hat p(1-\hat p)/n}\) 近似，模型比较优先配对 bootstrap。
7. reward 不能保证策略永不尝试危险动作；不可逆行为需要权限、沙箱和审批硬约束。
8. 模型、数据、seed、预算、评估和除目标变量外的超参数。

</details>

---

# 附录：可选原始资料

- [CS285 Homework 4 · diagnostics and ablations](https://rail.eecs.berkeley.edu/deeprlcourse/static/homeworks/hw4.pdf)
- [Berkeley Advanced LLM Agents](https://rdi.berkeley.edu/adv-llm-agents/sp25)
- [CS224R · RL for Reasoning](https://cs224r.stanford.edu/spring_2025/slides/10_cs224r-rl_for_reasoning_lecture.pdf)

