# 第 5 章：模型的分数从哪里来

## 5.1 本章符号说明

| 符号 | English | 本章含义 |
|---|---|---|
| \(x\) | prompt | 用户问题或指令 |
| \(y\) | response | 模型的完整回答 |
| \(y_w,y_l\) | preferred / rejected response | 同一 prompt 下更受偏好与较不受偏好的回答 |
| \(r_\psi(x,y)\) | reward model score | 参数为 \(\psi\) 的模型给回答打出的标量 |
| \(\pi_{\text{SFT}}\) | supervised fine-tuned policy | 由高质量示范训练出的初始策略 |
| \(\pi_{\text{ref}}\) | reference policy | 对齐阶段使用的固定参考策略 |
| \(\beta\) | KL temperature | 偏离 reference 的代价；在 DPO 中也控制偏好 margin 尺度 |
| \(\sigma(z)\) | sigmoid | 将任意实数变成 \(0\) 到 \(1\) 之间的概率 |
| \(Z(x)\) | partition function | 只依赖 prompt、用于让最优策略概率和为 \(1\) 的归一化常数 |

## 5.2 本章目标

读完本章，你应该能够：

1. 先按反馈接口而不是算法名字组织大模型对齐方法；
2. 区分 demonstration、preference 和 verifiable outcome；
3. 说明 SFT 学到的是“像示范”，而不是直接最大化任务成功率；
4. 从成对偏好数据推出 reward model 的排序损失；
5. 说清 RLHF 中 policy、reference、reward model、value model 四个角色；
6. 从 KL 正则化的最优策略推导 DPO 的训练 logit；
7. 区分 DPO、RLHF 与 RLVR 各自需要的数据和能解决的问题。

## 5.3 本章主线

前四章解决的是：

> 已经有 reward 时，怎样把它变成稳定、受控的模型更新。

但 `23×17` 的 \(0/1\) 从哪里来？开放问答的“更有帮助”又怎样变成数字？

在算法之前，先看我们真正能收集到什么：

| 反馈形态 | 一条训练记录长什么样 | 人或程序回答的问题 |
|---|---|---|
| 示范 | \((x,y^\*)\) | “理想回答应该怎样写？” |
| 成对偏好 | \((x,y_w,y_l)\) | “这两个回答哪个更好？” |
| 可验证结果 | \((x,y,R)\) | “这个回答是否通过明确规则？” |

三种数据接口不同，后面的训练方法才不同：

\[
\text{示范}\rightarrow\text{SFT},
\quad
\text{偏好}\rightarrow
\begin{cases}
\text{RM + RL}\\
\text{DPO}
\end{cases},
\quad
\text{可验证结果}\rightarrow\text{RLVR}.
\]

## 5.4 本章新增概念

| 名词 | 中文直觉 | 先记住什么 |
|---|---|---|
| SFT | 模仿高质量答案 | 对给定答案做 token 级最大似然 |
| preference data | 回答间的相对比较 | 不要求标注者给绝对分数 |
| reward model | 把偏好压缩成可泛化标量 | 先学 scorer，再给新 rollout 打分 |
| RLHF | 用人类偏好训练 reward，再用 RL 优化 policy | 可以在线探索，但系统角色多 |
| DPO | 直接让 preferred 相对 reference 更可能 | 不显式训练 reward model，不在线 rollout |
| verifier | 根据明确规则核验结果 | 适合数学、代码、工具状态等可判定任务 |
| RLVR | 用可验证 reward 做在线强化学习 | reward 可靠且便宜时很有吸引力 |

## 5.5 第一种反馈：直接给理想回答

若数据是高质量示范 \((x,y^\*)\)，最直接的训练目标是：

\[
\mathcal L_{\text{SFT}}
=
-
\sum_{t=1}^{T}
\log\pi_\theta
\left(
y_t^\*
\mid
x,y_{<t}^\*
\right).
\]

它与普通语言模型交叉熵相同：在每个位置提高示范 token 的概率。

对贯穿案例，示范可能是：

```text
23×17 = 23×(10+7) = 230+161 = 391。
\boxed{391}
```

SFT 的优点：

- 监督密集，每个 token 都有目标；
- 训练稳定、实现简单；
- 能快速教会输出格式、基本行为与任务模板。

它的边界也很清楚：

- 只能模仿数据中已有的行为；
- 每个 prompt 通常只有少量示范，未展示的正确推理不会被直接鼓励；
- 示范中偶然的措辞也会被当作目标；
- 无法自然利用“模型自己尝试了多个答案，其中一个成功”。

所以 SFT 常是对齐起点，不等于已经解决了序列决策问题。

## 5.6 第二种反馈：人更容易比较，而不是打绝对分

对开放问题，让标注者回答“这个答案值 0.73 分”很难；比较两个答案通常容易：

```text
prompt: 请解释为什么月亮不会掉到地球上。

回答 A: 给出轨道运动、引力与切向速度的连贯解释。
回答 B: 只说“因为月亮在太空中”，且包含事实错误。

标注: A 优于 B。
```

数据记录为 \((x,y_w,y_l)\)。它只告诉我们排序，不直接提供两个回答的绝对 reward。

接下来有两条路线：

1. 学一个能给任意回答打分的 reward model，再使用第 2–4 章的 RL；
2. 不显式产生 reward 分数，直接用偏好对更新 policy。

先看第一条。

## 5.7 reward model 怎样从比较中学会打分

令 \(r_\psi(x,y)\) 输出一个实数。我们希望 preferred 回答的分数更高。最常用的 Bradley–Terry 假设是：

\[
P_\psi(y_w\succ y_l\mid x)
=
\sigma
\left(
r_\psi(x,y_w)
-
r_\psi(x,y_l)
\right).
\]

这里只关心分数差：

- 差为 \(0\)，模型认为两者胜率各为 \(50\%\)；
- 差为正，preferred 的预测胜率超过 \(50\%\)；
- 差越大，模型越确信排序。

训练损失为偏好标签的负对数似然：

\[
\mathcal L_{\text{RM}}(\psi)
=
-
\mathbb E
\left[
\log\sigma
\left(
r_\psi(x,y_w)
-
r_\psi(x,y_l)
\right)
\right].
\]

reward model 学会后，可以给从未见过的新回答 \(y\) 输出 \(r_\psi(x,y)\)。这就把稀疏的人工比较扩展成了可大批量调用的 scorer。

### 5.7.1 reward model 的分数不是真理

RM 只是在偏好数据分布上拟合人类比较。policy 会主动搜索高分输出，可能找到标注数据没有覆盖的漏洞：

- 用冗长或特定格式骗取高分；
- 强烈迎合错误前提；
- 生成 reward model 偏爱的词但不真正完成任务；
- 进入 RM 的分布外区域。

因此 RM accuracy 高不代表 policy 优化一定安全。需要 reference KL、在线评估、对抗测试和定期补充偏好数据。

## 5.8 RLHF 的完整角色分工

典型 RLHF 可以按时间顺序理解：

1. 用 demonstrations 训练 \(\pi_{\text{SFT}}\)；
2. 用 preference pairs 训练 \(r_\psi\)；
3. 从当前 policy 在线生成回答；
4. reward model 给完整回答打分；
5. 加入偏离 reference 的 KL 代价；
6. 用 PPO 等方法更新 policy 和 value model。

训练时常见四个角色：

| 角色 | 是否更新 | 输出 |
|---|---|---|
| policy / actor | 是 | token 分布 |
| value model / critic | 是 | 前缀的未来 reward 预期 |
| reward model | 否 | 完整回答偏好分数 |
| reference policy | 否 | 原始语言行为的 log-prob |

其中：

- reward model 回答“完整回答在人类偏好下值多少”；
- value model 回答“从当前前缀继续生成，预计最终能拿多少 RM reward”。

二者都输出标量，但训练数据、输入位置和职责不同，不能混用。

RLHF 的优势是 policy 能在线探索偏好模型认为更好的新回答。代价是系统复杂、rollout 昂贵，并可能优化 RM 漏洞。

## 5.9 不显式训练 reward：从最优策略推到 DPO

现在考虑第二条路线。我们仍从一个明确目标开始，而不是直接给 DPO 公式。

对固定 prompt \(x\)，假设存在未知真实偏好 reward \(r(x,y)\)。希望 policy 获得高 reward，同时不要偏离 reference 太远：

\[
\max_\pi
\mathbb E_{y\sim\pi(\cdot\mid x)}
\left[
r(x,y)
\right]
-
\beta
D_{\mathrm{KL}}
\left(
\pi(\cdot\mid x)
\|
\pi_{\text{ref}}(\cdot\mid x)
\right).
\]

这个优化问题的最优策略满足：

\[
\pi^\*(y\mid x)
=
\frac{1}{Z(x)}
\pi_{\text{ref}}(y\mid x)
\exp
\left(
\frac{r(x,y)}{\beta}
\right),
\]

其中 \(Z(x)\) 只是让所有回答概率和为 \(1\) 的归一化常数。移项：

\[
r(x,y)
=
\beta
\left[
\log\pi^\*(y\mid x)
-
\log\pi_{\text{ref}}(y\mid x)
\right]
+
\beta\log Z(x).
\]

关键观察：在同一个 prompt 下比较 \(y_w\) 与 \(y_l\) 时，\(\beta\log Z(x)\) 会相减消失：

\[
\begin{aligned}
r(x,y_w)-r(x,y_l)
=\beta\Bigg[
&
\log\frac{\pi^\*(y_w\mid x)}
{\pi_{\text{ref}}(y_w\mid x)}
\\
-
&
\log\frac{\pi^\*(y_l\mid x)}
{\pi_{\text{ref}}(y_l\mid x)}
\Bigg].
\end{aligned}
\]

将未知最优策略 \(\pi^\*\) 用正在训练的 \(\pi_\theta\) 表示，再代入 Bradley–Terry 偏好概率，得到 DPO loss：

\[
\mathcal L_{\text{DPO}}(\theta)
=
-
\mathbb E
\left[
\log\sigma
\left(
\beta
\left[
\log\frac{\pi_\theta(y_w\mid x)}
{\pi_{\text{ref}}(y_w\mid x)}
-
\log\frac{\pi_\theta(y_l\mid x)}
{\pi_{\text{ref}}(y_l\mid x)}
\right]
\right)
\right].
\]

## 5.10 DPO 的工程含义

定义回答相对 reference 的提升：

\[
m_\theta(x,y)
=
\log\pi_\theta(y\mid x)
-
\log\pi_{\text{ref}}(y\mid x).
\]

DPO 就是在推动：

\[
m_\theta(x,y_w)
>
m_\theta(x,y_l).
\]

即：

> 相比 reference，让当前 policy 对 preferred 回答的相对偏好提升得更多。

reference 很重要。若只最大化 \(\log\pi_\theta(y_w)-\log\pi_\theta(y_l)\)，模型可能通过无约束地压低 rejected 回答或破坏原分布来完成排序；reference 提供了比较坐标系。

DPO 训练不需要：

- 在线 rollout；
- 单独的 reward model；
- value model；
- PPO 的 old/current ratio。

但它也不会探索数据集之外的新回答。训练信号被固定偏好对限制，本质上更接近离线偏好优化。

## 5.11 第三种反馈：结果可以被程序核验

对数学、代码、结构化查询、工具调用等任务，常能构造 verifier：

| 任务 | verifier 检查 |
|---|---|
| 数学 | 解析最终答案，与标准答案或符号计算结果比较 |
| 代码 | 编译、单元测试、隐藏测试、安全规则 |
| SQL | 在隔离数据库执行并比较结果 |
| 工具 Agent | 检查环境最终状态是否满足约束 |
| 格式生成 | JSON schema、语法或约束检查 |

对 `23×17`，parser 从 `\boxed{391}` 提取 `391`，verifier 与标准答案比较：

\[
R(x,y)
=
\mathbf 1
\left[
\operatorname{parse}(y)=391
\right].
\]

这类 reward 不需要猜人类的主观偏好，通常便宜、稳定、可重复。使用可验证 reward 在线训练称为 reinforcement learning with verifiable rewards，简称 **RLVR**。

RLVR 的优势：

- 可以从当前 policy 在线采样并探索新解法；
- reward 的含义明确；
- 不需要 reward model 泛化到分布外回答。

它的边界：

- 只适用于可可靠核验的维度；
- verifier 或 parser 有漏洞时，policy 会直接优化漏洞；
- 结果正确并不保证推理过程安全、简洁或忠实；
- 稀疏 \(0/1\) reward 仍需要足够探索。

## 5.12 一张表选清训练路线

| 你拥有的数据/环境 | 自然起点 | 是否在线采样 | 是否需要 critic | 主要风险 |
|---|---|---:|---:|---|
| 高质量标准回答 | SFT | 否 | 否 | 受示范覆盖限制 |
| 固定偏好对 | DPO | 否 | 否 | 无法探索数据集外行为 |
| 偏好对 + 希望在线探索 | RM + RLHF/PPO | 是 | 通常是 | reward model 被利用 |
| 程序可核验结果 | RLVR + policy optimization | 是 | 取决于算法 | verifier 漏洞、奖励稀疏 |

这些方法常按阶段组合，而不是互斥：

\[
\text{预训练}
\rightarrow
\text{SFT}
\rightarrow
\text{DPO 或 RLHF}
\rightarrow
\text{特定可验证任务上的 RLVR}.
\]

## 5.13 回到贯穿案例

训练 `23×17` 类型的算术模型时，可以逐级利用：

1. **SFT**：提供若干正确分解示范，先让模型学会基本格式和运算模式；
2. **DPO**：给同题下更清楚/更可靠与较差的回答对，训练回答偏好；
3. **RLVR**：让当前模型自己尝试大量新题，只按最终答案是否正确给 reward；
4. **PPO 或下一章的 group-relative 方法**：将在线 reward 变成受控更新。

若最终答案有标准解，再训练一个主观 RM 去猜“算对没有”通常是绕路；直接 verifier 更可靠。若目标是“解释友好、不过度冗长”，程序 verifier 又不足，需要偏好数据补充。

## 5.14 本章容易混淆的结论

| 容易误解成 | 正确理解 |
|---|---|
| SFT 也是 RL | SFT 对固定示范做最大似然，没有由当前策略采样与回报加权 |
| reward model 就是 critic | RM 评价完整回答的偏好，critic 预测当前 policy 从前缀出发的未来回报 |
| DPO 等价于先训练一个显式 RM 再跑 PPO | DPO 直接在固定偏好对上优化相对 log-prob |
| verifier 一定比人类偏好更全面 | 它只可靠覆盖可形式化检查的维度 |
| RLVR 不会 reward hacking | parser、测试或环境规则同样可能被利用 |
| 选出算法再找数据 | 应先确定反馈接口，再选择与之匹配的目标 |

## 5.15 本章自测

1. 为什么 pairwise preference 往往比绝对打分更容易标注？
2. reward model 和 value model 都输出标量，它们的监督数据和职责分别是什么？
3. DPO 推导中，归一化常数 \(Z(x)\) 为什么能消失？
4. DPO 为什么需要 reference policy，却不需要 old policy？
5. 数学题已有可靠标准答案时，RM 与 verifier 哪个更自然？为什么？
6. 何时应把 SFT、DPO 与 RLVR 串成多个阶段，而不是只选一个？

## 5.16 本章之后还缺什么

对可验证推理任务，一个 prompt 往往一次采样多条回答。同组回答天然提供了“这道题当前通常能拿多少分”的参照。

第 6 章会由这个同题组内参照推出：

- group-relative advantage；
- 不使用 critic 的 GR-REINFORCE；
- 需要重复利用 rollout 时的 GRPO；
- reference KL 的采样估计。

这样 GRPO 不会以一个庞大公式突然出现，而是从本章的 RLVR 数据形态自然长出来。

## 5.17 对应论文与资料

- [InstructGPT](https://arxiv.org/abs/2203.02155)
- [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)
- [Direct Preference Optimization](https://arxiv.org/abs/2305.18290)
- [DeepSeekMath](https://arxiv.org/abs/2402.03300)
- [Let’s Verify Step by Step](https://arxiv.org/abs/2305.20050)
