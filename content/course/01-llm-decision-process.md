# 第 1 章：把一次 LLM 生成写成顺序决策问题

## 1.1 本章符号说明

| 符号 | English | 本章含义 |
|---|---|---|
| \(x\) | prompt | 输入题目 |
| \(y_t\) | generated token | 回答中的第 \(t\) 个 token |
| \(y_{<t}\) | generated prefix | 第 \(t\) 个 token 之前的回答前缀 |
| \(s_t\) | state | \(x\) 与 \(y_{<t}\) 拼成的完整上下文 |
| \(a_t\) | action | 当前采样的 token，等于 \(y_t\) |
| \(\pi_\theta(a_t\mid s_t)\) | policy | 当前语言模型的 next-token 概率 |
| \(r_t\) | reward | 第 \(t\) 步后立刻得到的外部评分 |
| \(\tau\) | trajectory / rollout | 从 prompt 到回答结束的一次完整生成 |
| \(\gamma\) | discount factor | 后续 reward 每向远处一步的折扣比例 |
| \(R(\tau)\) | trajectory return | 整条轨迹累计得到的回报 |
| \(J(\theta)\) | objective | 当前模型在题目分布上的平均回报 |

## 1.2 本章目标

读完本章，你应该能够：

1. 把 LLM 的 state、action、policy 和 transition 逐项说清；
2. 写出一条回答的生成概率；
3. 区分即时 reward、整条轨迹的 return 和训练目标；
4. 解释为什么 verifier 给出的 0/1 分数不是 token 标签；
5. 说明 EOS 终止与最大长度截断为什么不是一回事。

## 1.3 本章主线

我们的项目只有一个外部接口：语言模型生成回答，判题程序在回答结束后给 0 或 1 分。

在讨论算法前，必须先回答：**模型到底在控制什么？一条回答的概率从哪里来？训练希望哪个量变大？**

本章只建立这个问题。还不讨论怎样求梯度。

## 1.4 本章新增概念

| 名词 | 中文 | 在当前项目中解决什么问题 |
|---|---|---|
| state | 状态 | 记录模型下一步决策所需的完整前缀 |
| action | 动作 | 表示模型当前可以改变的一个选择 |
| policy | 策略 | 给每个可能的下一个 token 分配概率 |
| transition | 状态转移 | 说明采样 token 后怎样得到新前缀 |
| reward | 奖励 | 外部系统对一步或整条回答的即时反馈 |
| return | 回报 | 从某个时刻开始累计的后续 reward |
| trajectory | 轨迹 | 一次从开始到结束的交互记录 |
| rollout | 采样轨迹 | 强调轨迹由当前策略实际生成 |
| MDP | 马尔可夫决策过程 | 把顺序决策压成统一数学接口 |

## 1.5 先跟完一次真实生成

prompt 是：

```text
计算 23 × 17，并把最终答案放在 \boxed{} 中。
```

为了便于阅读，下面把若干真实 tokenizer token 合成三个片段：

| 时刻 | 模型已经看到的内容 | 本次生成 |
|---:|---|---|
| 1 | prompt | `23×10=230；` |
| 2 | prompt + 第一段 | `23×7=161；` |
| 3 | prompt + 前两段 | `所以 \boxed{391}<eos>` |

每一行都执行相同的循环：

```text
读取完整上下文
→ 输出 next-token 概率
→ 从概率分布采样
→ 把 token 拼到上下文
→ 继续生成
```

模型真正控制的不是最终答案字符串，而是每一步的 next-token 分布。

## 1.6 把生成过程中的对象逐一命名

### 1.6.1 State：模型下一步决策前已经知道什么

第 \(t\) 次采样前，模型看到：

\[
s_t=(x,y_{<t}).
\]

这里：

- \(x\) 是 prompt；
- \(y_{<t}\) 是此前已经生成的全部 token；
- \(s_t\) 是两者拼成的完整上下文。

为什么 state 必须包含完整前缀？因为相同的生成位置，在不同前缀下会产生完全不同的 next-token 分布。

### 1.6.2 Action：模型当前做出的一个选择

模型在状态 \(s_t\) 下采样 token \(y_t\)。把这次选择记作：

\[
a_t=y_t.
\]

action 在这里没有神秘含义，就是“下一个 token 选了什么”。

### 1.6.3 Policy：谁给 action 分配概率

参数为 \(\theta\) 的语言模型给出：

\[
\pi_\theta(a_t\mid s_t).
\]

从左到右读：

> 当前参数为 \(\theta\) 的模型，在前缀 \(s_t\) 下，给本次 token \(a_t\) 分配的概率。

\(\pi\) 是强化学习对 policy 的惯用记号，不是新增的一份模型。

### 1.6.4 Transition：动作怎样改变下一状态

纯文本生成时：

\[
s_{t+1}=s_t+a_t.
\]

加号表示 token 序列拼接。给定当前前缀和采样 token，下一前缀是确定的。

到这里，LLM 与顺序决策对象的对应关系已经完整：

| 顺序决策对象 | LLM 生成中的含义 |
|---|---|
| state | prompt + 已生成前缀 |
| action | 下一个 token |
| policy | 语言模型 next-token 分布 |
| transition | 把 token 追加到前缀 |
| terminal condition | 生成 EOS 或任务真正结束 |

## 1.7 一整条回答的概率从哪里来

设回答共有 \(T\) 个 token：

\[
y=(y_1,\ldots,y_T).
\]

生成它需要连续做 \(T\) 次条件采样。概率的链式法则给出：

\[
p_\theta(y\mid x)
=
\prod_{t=1}^{T}
\pi_\theta(y_t\mid x,y_{<t}).
\]

\(\prod\) 表示把每一步的条件概率相乘。

假设三个合并片段的条件概率是：

\[
0.60,\qquad 0.50,\qquad 0.80.
\]

那么整条回答的概率是：

\[
0.60\times0.50\times0.80=0.24.
\]

这个乘积解释了训练入口：只要提高通向正确回答的若干 token 条件概率，正确回答的整体概率就会上升。

从 prompt 开始直到回答结束的完整记录，叫一条 **trajectory（轨迹）**。当强调它是当前模型现场采样出来的数据时，常叫 **rollout**。

在本项目里，\(\tau\) 可以直接理解为：

\[
\tau=(x,y_1,\ldots,y_T).
\]

## 1.8 Reward、return 和目标不是同一个量

判题程序在回答结束后运行：

```python
reward = 1.0 if parsed_answer == 391 else 0.0
```

### 1.8.1 Reward：某一步之后立刻发生了什么

把第 \(t\) 步后立刻得到的评分记作 \(r_t\)。

只有终局评分时：

\[
r_1=0,\qquad r_2=0,\qquad r_3=1.
\]

### 1.8.2 Return：从现在往后总共得到多少

从轨迹开始累计的回报写作：

\[
R(\tau)
=
\sum_{t=1}^{T}\gamma^{t-1}r_t.
\]

\(\gamma\in(0,1]\) 是 discount factor。短文本任务通常先取 \(\gamma=1\)，于是 return 就是所有 reward 的直接相加。

正确回答的 return 为 1，错误回答为 0。

### 1.8.3 Objective：模型反复生成时平均表现怎样

单条 rollout 可能碰巧成功。训练真正希望提高的是许多题目、许多次生成的平均回报。

设 \(\mathcal D\) 是训练 prompt 的分布。训练目标是：

\[
J(\theta)
=
\mathbb E_{
x\sim\mathcal D,\;
\tau\sim\pi_\theta(\cdot\mid x)
}
[R(\tau)].
\]

这句话读成：

> 从题库抽一道题，让当前模型生成，再对得到的回报取长期平均。

如果固定 prompt 下模型 30% 答对、70% 答错，那么：

\[
J(\theta)=0.30\times1+0.70\times0=0.30.
\]

训练希望改变 \(\theta\)，让这个平均值变大。

## 1.9 为什么这构成一个 MDP

现在所有具体对象已经出现，再给整体结构命名。

**Markov Decision Process（马尔可夫决策过程，MDP）**描述这样的过程：

1. agent 观察 state；
2. policy 选择 action；
3. environment 产生新 state 与 reward；
4. 重复直到终止；
5. 目标是最大化长期 return。

LLM 生成满足这套接口。这里的“Markov”不表示模型只看最后一个 token；state 已经包含完整前缀，因此它保留了预测后续所需的信息。

本教程不要求你先背抽象六元组。对 LLM 只需要能建立下面的具体映射：

\[
\text{完整前缀}
\rightarrow
\text{下一个 token}
\rightarrow
\text{新前缀}
\rightarrow
\text{最终评分}.
\]

## 1.10 为什么这不是普通监督学习

| 问题 | 监督微调 | 当前 RL 问题 |
|---|---|---|
| 回答从哪里来 | 固定专家数据 | 当前模型自己生成 |
| 反馈粒度 | 每个位置都有目标 token | 通常只有整条回答分数 |
| 模型更新后数据 | 原数据不变 | 下一批 rollout 分布改变 |
| 目标 | 模仿给定回答 | 提高当前策略的期望回报 |

verifier 的 1 分不是下一个 token 的标签。它没有告诉模型：

- 哪一步推理最关键；
- 错误回答应该改成哪个 token；
- 是否存在多种同样正确的回答。

因此不能直接把 `reward=1` 塞进普通交叉熵标签位置。

## 1.11 EOS、真正终止与长度截断

实现时必须区分：

- **terminated**：任务真的结束，例如模型生成 EOS，判题程序已经给出最终结果；
- **truncated**：采样器因为最大长度停止，但任务理论上还能继续；
- **padding**：为了组成 batch 补齐的无效位置。

这三者会影响后续 value bootstrap 和 mask。现在只记住：

> 最大长度到了，不等于环境真的进入了没有未来回报的终止状态。

第 3、7 章会把这个区别放进公式和代码。

## 1.12 本章贯穿例子

当前模型只会生成两类回答：

| 回答 | 概率 | reward |
|---|---:|---:|
| 正确：`\boxed{391}` | 0.30 | 1 |
| 错误：`\boxed{381}` | 0.70 | 0 |

现在可以完整描述项目：

1. state 是 prompt 与已生成前缀；
2. action 是每个 next token；
3. policy 是当前语言模型；
4. trajectory 是一条完整回答；
5. return 是判题程序的最终 0/1；
6. objective 是当前模型的平均正确率 0.30。

下一章只解决一个问题：

> token 是离散采样结果，不能对 token id 求导。怎样利用最终分数，改变产生这些 token 的概率？

## 1.13 本章自测

1. 为什么 LLM 的 state 不能只写“当前 token 位置”？
2. 一条回答的概率为什么是逐 token 条件概率的乘积？
3. reward、return 和 \(J(\theta)\) 分别是什么层级的量？
4. verifier 的 1 分为什么不是 token 标签？
5. terminated、truncated 和 padding 的差别是什么？

## 1.14 对应官方资料

- [CS285 Lecture 4 · RL Basics](https://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-4.pdf)
- [CS285 Fall 2023 · Lecture 4](https://www.youtube.com/playlist?list=PL_iWQOsE6TfVYGEGiAOMaOzzv41Jfm_Ps&index=4)
