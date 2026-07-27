# 第三章：用一条 LLM 回答算出 Value、TD 与 GAE

第二章已经得到最原始的策略更新规则：

> 一条回答最后得分高，就提高这条回答中已采样 token 的概率；得分低，就降低它们的概率。

这条规则能工作，但它把整份回答的结果原样发给所有 token，也没有区分“本来就容易的题”和“出乎意料答对的难题”。本章只解决这个问题，而且始终沿用同一条 LLM 数学回答，不凭空切换语境。

整章的因果链只有一条：

\[
\text{完成一次 rollout}
\longrightarrow
\text{得到真实 reward}
\longrightarrow
\text{学习每个前缀的正常水平}
\longrightarrow
\text{计算每一步让预期改变了多少}
\longrightarrow
\text{形成策略模型的更新权重}.
\]

后面的术语都只给这条链上的已有对象命名，不会额外引入一条平行逻辑。

## 3.1 先把一条 LLM 回答写成轨迹

用一个贯穿本章的例子：

> prompt：计算 \(17\times6\)，并写出过程。

为方便手算，下面把若干真实 token 合并成一个“生成片段”。实际训练仍然在 tokenizer 的每个 response token 上计算。

| 步 \(t\) | 生成前已经看到的内容 | 本步生成的片段 |
|---:|---|---|
| 0 | `计算 17×6，并写出过程。` | `10×6=60；` |
| 1 | `… 10×6=60；` | `7×6=42；` |
| 2 | `… 7×6=42；` | `60+42=102。<eos>` |

在强化学习记号中：

- **状态 \(s_t\)**：prompt 加上第 \(t\) 步之前已经生成的前缀；
- **动作 \(a_t\)**：第 \(t\) 步采样出的下一个 token；表中用一个片段代替若干 token；
- **即时奖励 \(r_t\)**：执行这一步后环境立即给出的分数；
- **终止**：生成 EOS，或环境明确宣布任务成功/失败。

若 verifier 只检查最终答案，中间步骤没有分数，那么这条正确轨迹可以写成：

\[
r_0=0,\qquad r_1=0,\qquad r_2=1.
\]

从第 \(t\) 步开始，后面真正拿到的折扣奖励总和叫 **return（回报）**：

\[
G_t
=
r_t+\gamma r_{t+1}+\gamma^2r_{t+2}+\cdots.
\]

\(\gamma\in[0,1]\) 是折扣系数。短 LLM 回答中常取 \(\gamma=1\)，本章的手算也先取 1。于是这条成功轨迹上：

\[
G_0=G_1=G_2=1.
\]

先把四个容易混淆的对象分开：

| 对象 | 何时得到 | 它回答什么 |
|---|---|---|
| \(r_t\) | 环境执行一步后给出 | 这一步立即拿到多少分？ |
| \(G_t\) | 一条 rollout 完成后由 reward 算出 | 这次从该前缀往后实际拿到多少分？ |
| 当前前缀的平均未来回报 | 对当前策略的理论期望 | 从该前缀继续生成，平均能拿多少分？ |

三者的差别尤其重要：reward 是一次即时观测，return 是这次样本的未来总分，第三行是许多可能后续的平均水平。到 3.3 再给第三行正式命名和记号。

## 3.2 为什么不能只把最终得分广播给所有 token

先看跨 prompt 的问题。假设当前策略在两道题上的正确率为：

| prompt | 当前策略正确率 | 本次 rollout 的最终 reward |
|---|---:|---:|
| 简单题 | 90% | 1 |
| 难题 | 5% | 1 |

两次都答对并不代表训练信息完全相同：

- 简单题得 1 分接近正常发挥；
- 难题得 1 分远高于正常发挥。

如果都用 \(G=1\) 作为权重，策略看不见这种差别。我们需要先问：

> 站在当前前缀上，按当前策略继续生成，通常能拿多少分？

课本把这种只依赖状态的参照叫 **baseline（基线）**。先把它写成
\(b(s_t)\)。本次实际回报减去正常水平：

\[
\widehat A_t=G_t-b(s_t).
\]

这里 \(A\) 来自 advantage（优势）；帽子 \(\widehat{\phantom A}\) 表示它是用样本算出的估计值，而不是真实期望。

- \(\widehat A_t>0\)：本次表现好于正常水平，提高已采样动作的概率；
- \(\widehat A_t<0\)：本次表现差于正常水平，降低已采样动作的概率；
- \(\widehat A_t\approx0\)：大致符合预期，不需要强更新。

代入上表：

\[
\widehat A_{\text{easy}}=1-0.90=0.10,
\qquad
\widehat A_{\text{hard}}=1-0.05=0.95.
\]

这也解释了 0/1 reward 下负向更新从哪里来。错误回答得到 \(G=0\)，若正常水平是
0.6，则 \(\widehat A=0-0.6=-0.6\)，策略才会明确压低这次动作。

### 为什么减去正常水平不会系统性改错方向

正常水平可以依赖状态 \(s\)，但不能偷看本次采样动作 \(a\)。固定状态后：

\[
\begin{aligned}
\mathbb E_{a\sim\pi_\theta(\cdot\mid s)}
\left[b(s)\nabla_\theta\log\pi_\theta(a\mid s)\right]
&=
b(s)\sum_a\pi_\theta(a\mid s)
\nabla_\theta\log\pi_\theta(a\mid s)\\
&=
b(s)\nabla_\theta\sum_a\pi_\theta(a\mid s)\\
&=
b(s)\nabla_\theta1=0.
\end{aligned}
\]

因此减去只依赖状态的 baseline 不改变期望梯度，只减少不同样本之间与动作优劣无关的起伏。现在真正的问题变成：这个正常水平怎么得到？

## 3.3 \(V(s)\) 不是手填的：它是一个学出来的前缀评分器

最自然的正常水平定义为：

> 固定当前前缀 \(s\)，让当前策略 \(\pi\) 从这里继续采样所有可能回答，它们未来 return 的平均值。

课本把它叫 **state value（状态价值）**，用 \(V^\pi(s)\) 表示：

\[
V^\pi(s)
=
\mathbb E_{\tau\sim\pi}\left[G_t\mid s_t=s\right].
\]

在只有终局 0/1 reward 且 \(\gamma=1\) 时，\(V^\pi(s)\) 有一个很直观的读法：

> 从这个 prompt + 已生成前缀继续采样，最终答对的概率。

例如，同一前缀若真的重复采样 100 个后续，其中 31 个答对，那么该前缀的经验 value 约为 \(0.31\)。但训练时不可能为每个前缀单独采样 100 次，所以要让神经网络在大量 rollout 之间泛化。

### Value 网络的输入和输出

用带参数的网络 \(V_\phi\) 逼近未知的 \(V^\pi\)：

\[
V_\phi(s_t)\approx V^\pi(s_t).
\]

在 LLM PPO 中，常见实现是：

1. 把 `prompt + response` 输入一个因果 Transformer；
2. 每个位置得到隐藏向量 \(h_t\)；
3. 在线性 value head 上输出一个标量：

\[
V_\phi(s_t)=w^\top h_t+b.
\]

因此一次前向就能得到整条序列各个前缀的 value，不需要真的把每个前缀分别重跑一遍。概念上，预测“生成 \(a_t\) 之前的前缀值”；代码中要通过 shift 保证 value 与对应 response token 对齐。

伪代码只表达张量含义：

```python
hidden = value_backbone(input_ids, attention_mask)    # [B, L, D]
all_values = value_head(hidden).squeeze(-1)           # [B, L]
old_value = align_values_to_response_actions(all_values)  # [B, T]
```

到这里，这个对象的作用已经清楚了，再给它课程中的常用名称：
**critic 就是这个 value 预测器**。它可以与 policy 共享部分骨干，也可以是一份独立模型；数学角色不变。示例中出现的
\(0.30\)、\(0.45\) 等数字，都是 `value_head` 的预测值，不是人工指定的训练标签。

### Value 网络拿什么当标签

rollout 结束后，真实 reward 已经知道，因此可以为每个前缀算出这次样本的 return
\(G_t\)。最直接的训练方式就是普通回归：

\[
\mathcal L_V(\phi)
=
\mathbb E_t\left[
\left(V_\phi(s_t)-G_t\right)^2
\right].
\]

仍以只有终局 0/1 reward 的任务为例：

- 一条成功 rollout 的各前缀，Monte Carlo 标签都是 1；
- 一条失败 rollout 的各前缀，标签都是 0；
- 同类前缀在大量成功与失败样本上反复出现，平方误差最优解就是平均成功率。

这就是 \(V(s)\) 的来源：**策略负责产生后续，环境或 verifier 给最终结果，critic 用这些已发生的结果学习预测尚未发生的结果。**

这里没有逻辑循环。每批数据中的训练目标来自已经结束的 rollout；目标和旧 value 在本轮优化时会 `detach`。策略更新后，“继续生成的平均结果”也会变化，下一轮再用新 rollout 继续校准 critic。

### Reward model 和 critic 不是同一个东西

二者都可能输出标量，但问题不同：

| 模型 | 典型输入 | 学习目标 | 何时使用 |
|---|---|---|---|
| reward model / verifier | 通常是完整回答 | 这份答案质量如何 | rollout 完成后产生 reward |
| critic / value model | 每个生成前缀 | 当前策略从这里继续的期望 return | rollout 中为每一步估计正常水平 |

reward model 可以说“整份答案得 0.8 分”，critic 要说“只看到这个前缀时，按当前策略继续，预计最终能得多少分”。

## 3.4 先看一条 rollout 中 value 到底长什么样

假设 critic 在本章正确轨迹的相邻前缀上给出：

| 状态 | 状态内容 | 旧 critic 预测 |
|---|---|---:|
| \(s_0\) | 只有 prompt | \(V(s_0)=0.30\) |
| \(s_1\) | 已生成 `10×6=60；` | \(V(s_1)=0.45\) |
| \(s_2\) | 又生成 `7×6=42；` | \(V(s_2)=0.80\) |
| \(s_3\) | 已输出正确答案并终止 | 后续 value 记为 0 |

这些值的含义分别是：

- 在还没开始回答时，critic 估计当前 policy 有 30% 概率答对；
- 写出第一步后，预测提高到 45%；
- 两个乘法部分都正确后，预测提高到 80%；
- 终止状态之后不会再获得未来 reward，所以 continuation value 为 0。最终的 1 分已经放在导致终止的那次转移上。

注意，value 不要求沿生成过程单调上升。critic 可能被一个貌似合理但最终错误的步骤骗得更乐观；终局 reward 会负责纠正它。

课本还定义了“固定当前动作以后，未来平均拿多少分”：

\[
Q^\pi(s,a)
=
\mathbb E[G_t\mid s_t=s,a_t=a].
\]

状态 \(s\) 下动作 \(a\) 相对正常水平的真实超额表现叫 advantage：

\[
A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s).
\]

PPO 通常不另外训练一个 \(Q\) 网络，而是用实际 rollout 和 critic 构造
\(\widehat A_t\)。下面就从相邻两个 value 推出这个估计。

## 3.5 一步之后，最终得分预期改变了多少

若 \(V\) 预测准确，那么当前状态的未来价值应该等于：

> 这一步立即得到的 reward，加上下一状态仍可获得的未来价值。

写成 Bellman 一致性：

\[
V^\pi(s_t)
=
\mathbb E\left[
r_t+\gamma V^\pi(s_{t+1})
\mid s_t
\right].
\]

在一条实际 rollout 上，不取期望，直接比较“一步后的新目标”和“一步前的旧预测”：

\[
r_t+\gamma V_\phi(s_{t+1})-V_\phi(s_t).
\]

这句话先读成：

> 看到本步 reward 和新前缀以后，对最终 return 的预期需要上调或下调多少？

课本把它叫 **temporal-difference error（TD 误差）**，用
\(\delta_t\) 表示：

\[
\delta_t
\equiv
r_t+\gamma V_\phi(s_{t+1})-V_\phi(s_t).
\]

它不是另一种 reward，只是相邻时间步之间的 value 预测残差。

### 把正确回答逐步代入

取 \(\gamma=1\)。前两步没有即时 reward，最后一步得到 1 分：

\[
\begin{aligned}
\delta_0
&=0+0.45-0.30=0.15,\\
\delta_1
&=0+0.80-0.45=0.35,\\
\delta_2
&=1+0-0.80=0.20.
\end{aligned}
\]

解释：

- 第一个正确乘法片段让成功预期从 30% 升到 45%，局部信号为 \(+0.15\)；
- 第二个片段让预期从 45% 升到 80%，局部信号为 \(+0.35\)；
- 终局真的答对，结果比 critic 的 80% 预期还好，再补 \(+0.20\)。

### 再看一条貌似合理但最终失败的回答

假设另一条 rollout 的 value 依次为
\(0.30\to0.42\to0.70\)，最终答案却错了，reward 为 0：

\[
\delta_0=0.12,\qquad
\delta_1=0.28,\qquad
\delta_2=0-0.70=-0.70.
\]

前两步的正 TD 说明 critic 当时变得更乐观，不代表这两步已经被环境认证为正确。终局的
\(-0.70\) 才揭示先前预期过高。于是产生下一个问题：终局修正怎样传回更早的 token？

## 3.6 从一步 TD 到整段回报：先看“误差相加”

若只用 \(\delta_t\) 更新第 \(t\) 个动作，信号很局部，但过度依赖 critic 的下一步预测。最直接的改进是把后面的 TD 误差也加回来。

向前累计两步：

\[
\begin{aligned}
\delta_t+\gamma\delta_{t+1}
&=
r_t+\gamma V(s_{t+1})-V(s_t)\\
&\quad+
\gamma\left[r_{t+1}+\gamma V(s_{t+2})-V(s_{t+1})\right]\\
&=
r_t+\gamma r_{t+1}+\gamma^2V(s_{t+2})-V(s_t).
\end{aligned}
\]

中间的 \(V(s_{t+1})\) 正负相消。继续加到终局时，所有中间 value 都会相消，只剩：

\[
\sum_{l=0}^{T-t}\gamma^l\delta_{t+l}
=
G_t-V(s_t).
\]

这正是“实际 return 减去正常水平”。用正确轨迹检查：

\[
0.15+0.35+0.20=0.70=1-0.30.
\]

所以 TD 并没有凭空创造新目标；它只是把同一个“比预期好多少”拆成相邻前缀之间的连续修正。

现在两端的取舍很清楚：

| 使用多少后续 | 得到的信号 | 优点 | 代价 |
|---|---|---|---|
| 只用当前 \(\delta_t\) | 1-step TD | 较局部、方差低 | 很依赖 critic，偏差可能大 |
| 一直加到终局 | \(G_t-V(s_t)\) | 不从未来 value 自举 | rollout 间波动大 |
| 使用有限后续 | \(n\)-step | 在两者间折中 | 要决定看多远 |

这里的 **bootstrapping（自举）** 是指：真实未来还没完全展开时，先拿
\(V(s_{t+n})\) 的预测接在目标末尾。它不是“模型自己证明自己正确”；终局 reward 仍然为整条链提供锚点。

## 3.7 GAE：让近处修正权重大，远处修正逐渐衰减

固定只看 1 步或 \(n\) 步都很生硬。更平滑的做法是：

1. 当前 TD 误差权重为 1；
2. 每向未来多传播一步，就再乘一个 \(\gamma\lambda\)；
3. \(\lambda\in[0,1]\) 控制愿意传播多远。

于是得到 **generalized advantage estimation（GAE，广义优势估计）**：

\[
\widehat A_t^{\text{GAE}(\gamma,\lambda)}
=
\delta_t
+\gamma\lambda\delta_{t+1}
+(\gamma\lambda)^2\delta_{t+2}
+\cdots.
\]

它也可以从后往前递推：

\[
\widehat A_t
=
\delta_t+\gamma\lambda\widehat A_{t+1}.
\]

### 把正确回答完整手算一遍

仍取 \(\gamma=1\)，再取 \(\lambda=0.8\)。从最后一步反向计算：

\[
\begin{aligned}
\widehat A_2
&=0.20,\\
\widehat A_1
&=0.35+0.8\times0.20=0.51,\\
\widehat A_0
&=0.15+0.8\times0.51=0.558.
\end{aligned}
\]

第二步的正修正不仅作用于自己，还以 0.8 的权重向前传播；更远的修正衰减得更多。

### 再把失败回答完整手算一遍

失败轨迹的 TD 为 \(0.12,0.28,-0.70\)：

\[
\begin{aligned}
\widehat A_2
&=-0.70,\\
\widehat A_1
&=0.28+0.8\times(-0.70)=-0.28,\\
\widehat A_0
&=0.12+0.8\times(-0.28)=-0.104.
\end{aligned}
\]

虽然 critic 一度被貌似合理的前缀骗得更乐观，终局失败仍通过 GAE 向前传播，使早期动作得到负的净更新权重。

\(\lambda\) 的两个极端也能直接读懂：

- \(\lambda=0\)：\(\widehat A_t=\delta_t\)，只相信一步修正；
- \(\lambda=1\)：若轨迹完整且边界处理正确，所有后续 TD 都加回来，接近
  \(G_t-V(s_t)\)。

GAE 是一种偏差—方差折中，不是真实的 token 级因果标签。critic 判断错时，GAE 也会分错信用。

## 3.8 轨迹边界：`continue_mask` 到底在切断什么

反向递推不能跨过两条轨迹的边界。对执行动作 \(a_t\) 后是否真正结束，定义：

- `terminated[t] = 0`：仍有属于同一条轨迹的真实后续；
- `terminated[t] = 1`：生成 EOS，或环境明确判定成功/失败，轨迹真正结束。

代码需要一个 0/1 开关决定是否保留后续信息：

\[
\texttt{continue\_mask}_t
=
1-\texttt{terminated}_t.
\]

| 当前动作之后 | `terminated[t]` | `continue_mask` | 保留什么 |
|---|---:|---:|---|
| 同一轨迹继续 | 0 | 1 | 下一状态 value 和后续 GAE |
| 真正终止 | 1 | 0 | 二者都切断 |

`continue_mask` 不是新的强化学习概念，只是一个乘法开关。乘 0 可以防止下一条 batch 样本的 value 串进当前轨迹。

### 真正终止不等于达到最大长度

这两个边界不能混成一个语义含糊的 `done`：

- **terminated**：任务过程真的结束，后面没有合法 continuation，应令
  `continue_mask = 0`；
- **truncated**：只是采样器达到最大长度或时间限制而停止，任务理论上可能继续。

若截断时保留了最后一个真实观测，且 value 的定义包含其后续，就应使用该状态 value
进行 bootstrap。若业务把超长明确判为失败，则应先给出相应终局 reward，再按真正终止处理。补齐 batch 的 `padding mask` 又是另一件事，它只排除填充 token。

## 3.9 把公式变成一段不会错位的代码

对一条含 \(T\) 个 response token 的 rollout，准备：

- `reward[t]`：执行第 \(t\) 个 token 后得到的 reward；
- `old_value[t]`：生成该 token 前，旧 critic 对当前前缀的预测；
- `old_value[t + 1]`：执行后新前缀的 value；截断时可用于 bootstrap；
- `terminated[t]`：执行后是否真正终止。

`old_value` 之所以带 `old`，是因为它在 rollout 阶段算好后，本批 PPO 更新期间应保持不变。

```python
# shapes:
# reward, terminated: [T]
# old_value:          [T + 1]

gae = 0.0
advantage = zeros(T)

for t in reversed(range(T)):
    continue_mask = 1.0 - terminated[t]

    delta = (
        reward[t]
        + gamma * old_value[t + 1] * continue_mask
        - old_value[t]
    )
    gae = (
        delta
        + gamma * lam * continue_mask * gae
    )
    advantage[t] = gae

value_target = advantage + old_value[:-1]

advantage = advantage.detach()
value_target = value_target.detach()
```

最后一行公式为什么是：

\[
\widehat V_t^{\text{target}}
=
\widehat A_t+V_{\text{old}}(s_t)?
\]

因为 GAE 算出的 \(\widehat A_t\) 本来就是“目标相对旧 value 高多少或低多少”。给 critic
做回归时要把旧 value 加回去，恢复成绝对的 return 目标。以正确轨迹第一步为例：

\[
\widehat V_0^{\text{target}}
=
0.558+0.30=0.858.
\]

它不是最终 reward 1，也不是旧预测 0.30，而是 \(\lambda=0.8\) 下融合了多步真实信息和旧预测的训练目标。

## 3.10 到这里再把策略与 critic 合成 Actor–Critic

现在所有对象都已经出现：

- **actor**：策略 \(\pi_\theta(a_t\mid s_t)\)，决定下一个 token 的分布；
- **critic**：前缀预测器 \(V_\phi(s_t)\)，估计当前策略的未来 return；
- **advantage**：actor 的有符号更新权重；
- **value target**：critic 的回归标签。

最基本的两个 loss 是：

\[
\mathcal L_{\text{actor}}
=
-\mathbb E_t\left[
\widehat A_t\log\pi_\theta(a_t\mid s_t)
\right],
\]

\[
\mathcal L_{\text{critic}}
=
\mathbb E_t\left[
\left(
V_\phi(s_t)-\widehat V_t^{\text{target}}
\right)^2
\right].
\]

所谓 **Actor–Critic**，不是突然多出第三套训练逻辑，只是一个闭环：

1. actor 生成回答；
2. verifier / reward model 给 rollout 打分；
3. critic 预测各前缀的正常水平；
4. reward 与旧 value 构造 TD、GAE；
5. advantage 更新 actor，value target 更新 critic；
6. actor 改变后重新 rollout，再校准 critic。

同一批数据内，`old_value`、`advantage` 和 `value_target` 都应冻结。不能在每个 PPO
minibatch 中一边更新 critic，一边重新计算这批样本的 advantage，否则 actor 的训练标签会在一次优化过程中漂移。

## 3.11 只有终局 reward 时，GAE 做到了什么、没做到什么

在数学推理任务中，常见 reward 结构是：

\[
r_t=0\quad(t<T),\qquad r_T=R.
\]

GAE 仍能产生不同的 token 权重，是因为 learned critic 会随前缀改变预测：

- 新前缀让成功预期上升，产生正 TD；
- 新前缀暴露错误，让成功预期下降，产生负 TD；
- 终局 reward 纠正整条预测链；
- GAE 再控制这种修正向前传播多远。

但必须保留边界：

1. GAE 使用的是**学到的前缀价值变化**，不是人工标注的推理步骤对错；
2. critic 不准，细粒度 credit assignment 就不准；
3. 同一序列的所有 token 使用同一个终局分数，并不等于获得过程监督；
4. 真正的过程监督需要过程 reward、步骤 verifier 或其他额外信息。

第二周会介绍另一条路线：对同一 prompt 采样一组回答，用组内平均表现代替 learned
value baseline。那是 GR-REINFORCE / GRPO 的出发点，不要与本章的 critic 混为一谈。

### 第三章检查点

读完后应能不看页面完成三件事：

1. 用一句话区分 reward、return 和 value，并说明示例中的 \(V(s)=0.30\) 从哪里来；
2. 给定一条三步 LLM rollout 的 reward 与 value，手算每步 TD 和 GAE；
3. 解释真正终止、最大长度截断和 padding 为什么需要三个不同的 mask 语义。

如果这三项还不能独立完成，先不要进入 PPO clipping；第四章会直接使用本章算出的
`advantage`。
