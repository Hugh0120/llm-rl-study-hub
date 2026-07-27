# 第三章：一份答案的分数，怎样变成每个 token 的训练信号

第二章已经完成了最关键的一步：一条回答得到回报 \(G\) 后，可以用它调整这条回答中已采样 token 的概率。

但若 verifier 只在答案结束时给一个分数，整条回答里的 token 都会拿到同一个 \(G\)。这太粗了。本章只回答一个问题：

> 怎样利用当前模型对不同前缀的判断，把整份答案的分数变成更合理的逐 token 更新权重？

全章只用一个例子。prompt 是：

> 计算 \(17\times6\)，并写出过程。

为方便阅读，下面把若干真实 token 合并成三个生成片段。实际训练时仍会在 tokenizer 的每个 response token 上计算。

## 3.1 为什么直接使用整份答案的回报太粗

先看不同 prompt。假设当前模型的表现是：

| prompt | 当前模型通常答对的概率 | 这一次的最终得分 |
|---|---:|---:|
| 简单题 | 90% | 1 |
| 难题 | 5% | 1 |

两次都得 1 分，但训练含义不同：

- 简单题答对接近正常发挥；
- 难题答对远高于正常发挥。

所以真正有用的不是“这次得了几分”，而是：

\[
\text{这次实际得到的回报}
-
\text{站在当前前缀上通常能得到的回报}.
\]

课本把后面这个“通常水平”叫 **baseline（基线）**。它必须在采样当前 token
之前就能确定，不能看完这次选了哪个 token 后再随意修改。

现在只剩一个具体问题：

> 对一个 LLM 前缀，“通常能得到多少回报”究竟怎么计算？

## 3.2 \(V(s)\) 从哪里来：先看理想算法，再看实际训练

先做一个昂贵但直观的思想实验。

**第一步：只给 prompt。**

固定前缀：

```text
计算 17×6，并写出过程。
```

让当前模型从这里独立生成 10 次。假设 3 次最终答对，那么从这个前缀继续生成的平均得分约为：

\[
\frac{3}{10}=0.30.
\]

**第二步：固定一个已经生成的前缀。**

```text
计算 17×6，并写出过程。10×6=60；
```

再从这里采样 10 个后续。假设 5 个最终答对，那么这个前缀的平均未来得分约为
\(0.50\)。

**第三步：再给一个正确片段。**

```text
计算 17×6，并写出过程。10×6=60；7×6=42；
```

若 10 个后续里有 8 个答对，这个前缀的平均未来得分约为 \(0.80\)。

这三个数不是手填的“好感度”，而是在回答同一个问题：

> 固定当前前缀，按当前策略继续生成，最终平均能得多少分？

现在再给它正式记号。状态 \(s_t\) 是“prompt + 已生成前缀”，其状态价值定义为：

\[
V^\pi(s_t)
=
\mathbb E_\pi\left[G_t\mid s_t\right].
\]

- \(V\) 来自 value；
- 右上角 \(\pi\) 表示这个平均值取决于当前生成策略；
- \(G_t\) 是从当前状态往后实际获得的未来回报。

在只有终局 0/1 得分且 \(\gamma=1\) 时，\(V^\pi(s_t)\) 可以直接理解为：

> 当前模型从这个前缀继续生成，最终答对的概率。

因此上面的思想实验给出：

\[
V(s_0)\approx0.30,\qquad
V(s_1)\approx0.50,\qquad
V(s_2)\approx0.80.
\]

**Value 网络实际怎样学？**

真实训练不会为每个前缀额外分叉采样 10 次，那样太贵。实际做法是：

1. 当前策略正常生成一批完整回答；
2. verifier 给每条回答最终打分；
3. 一次成功 rollout 为沿途每个前缀提供一次“未来回报为 1”的样本，失败 rollout
   提供一次“未来回报为 0”的样本；
4. 用很多 rollout 训练一个神经网络 \(V_\phi(s)\)，让它预测这些未来回报；
5. 神经网络在相似 prompt 和相似前缀之间泛化，近似刚才昂贵的重复采样平均值。

最简单的训练目标就是平方误差：

\[
\mathcal L_V(\phi)
=
\mathbb E_t\left[
\left(V_\phi(s_t)-G_t\right)^2
\right].
\]

在 LLM 中，通常把前缀送进因果 Transformer，再把对应位置的隐藏向量通过一个线性
value head，输出一个标量。一次完整前向可以同时得到各个 token 前缀的 value。

课程里把这个 value 预测网络叫 **critic**。它不是答案评分器：

- verifier / reward model 负责评价已经完成的答案；
- critic 负责预测尚未完成的前缀，按当前策略继续生成后通常能拿多少分。

有了这个正常水平，本次样本的相对表现可以估计为：

\[
\widehat A_t=G_t-V_\phi(s_t).
\]

这里 \(A\) 来自 advantage（优势），帽子表示这是由本次样本估计出来的量。

- \(\widehat A_t>0\)：这次比预期好，提高所采样 token 的概率；
- \(\widehat A_t<0\)：这次比预期差，降低所采样 token 的概率。

但 \(G_t-V(s_t)\) 仍然要等整条回答结束。接下来把它拆成相邻前缀之间的修正。

## 3.3 一条回答怎样逐步算出 TD

考虑一条正确回答。令 \(\gamma=1\)，中间没有 reward，最终答对得到 1 分：

| 当前前缀 | 当前 \(V\) | 本步生成 | 本步 reward | 新前缀的 \(V\) |
|---|---:|---|---:|---:|
| \(s_0\)：只有 prompt | 0.30 | `10×6=60；` | 0 | 0.50 |
| \(s_1\)：已有第一步 | 0.50 | `7×6=42；` | 0 | 0.80 |
| \(s_2\)：已有两步 | 0.80 | `60+42=102。<eos>` | 1 | 0 |

终止后的 value 写成 0，因为后面不会再得到未来 reward；最后的 1 分已经放在导致终止的那一步。

现在比较：

\[
\text{看到本步结果后的新预期}
-
\text{执行本步之前的旧预期}.
\]

三步分别是：

\[
\begin{aligned}
0+0.50-0.30&=0.20,\\
0+0.80-0.50&=0.30,\\
1+0-0.80&=0.20.
\end{aligned}
\]

到这里再给它名字。第 \(t\) 步的这个预测修正叫
**temporal-difference error（TD 误差）**，记作：

\[
\delta_t
=
r_t+\gamma V(s_{t+1})-V(s_t).
\]

\(\delta\) 只是“差”的记号。它不是一种新 reward。

在正确轨迹上：

\[
\delta_0=0.20,\qquad
\delta_1=0.30,\qquad
\delta_2=0.20.
\]

把三步相加：

\[
0.20+0.30+0.20
=
0.70
=
1-0.30.
\]

中间的 value 会前后抵消，最后正好回到：

\[
\text{实际最终回报}-\text{最初预期}.
\]

再看失败回答。假设 critic 一度也给出
\(0.30\to0.50\to0.80\)，但最后答案错误，终局 reward 为 0：

\[
\delta_0=0.20,\qquad
\delta_1=0.30,\qquad
\delta_2=0-0.80=-0.80.
\]

critic 曾被貌似合理的前缀骗得越来越乐观，因此前两步 TD 为正；终局失败产生
\(-0.80\) 的修正。问题变成：这个终局修正应该向前传播多远？

## 3.4 GAE 只做一件事：控制终局修正向前传多远

有两个极端：

- 只用当前一步的 \(\delta_t\)：信号局部、波动较小，但过度相信 critic；
- 把后面所有 \(\delta\) 全部加回来：能得到完整的
  \(G_t-V(s_t)\)，但长回答上的样本波动更大。

GAE 在两者之间取一个平滑折中。它从后向前递推：

\[
\widehat A_t
=
\delta_t+\gamma\lambda\widehat A_{t+1}.
\]

\(\lambda\in[0,1]\) 是传播系数：

- 距离当前步越远，修正会多乘几次 \(\gamma\lambda\)；
- \(\lambda=0\) 时只保留当前 TD；
- \(\lambda\) 越接近 1，越接近把后续修正全部传回来。

仍令 \(\gamma=1,\lambda=0.8\)。从最后一步反向计算：

| 轨迹 | \((\delta_0,\delta_1,\delta_2)\) | \(\widehat A_2\) | \(\widehat A_1\) | \(\widehat A_0\) |
|---|---|---:|---:|---:|
| 最终答对 | \((0.20,0.30,0.20)\) | 0.20 | \(0.30+0.8\times0.20=0.46\) | \(0.20+0.8\times0.46=0.568\) |
| 最终答错 | \((0.20,0.30,-0.80)\) | -0.80 | \(0.30-0.8\times0.80=-0.34\) | \(0.20-0.8\times0.34=-0.072\) |

现在能看到 GAE 的实际作用：

- 正确回答的终局正修正向前传播，三个片段都得到正权重；
- 错误回答的终局负修正向前传播，抵消 critic 先前的错误乐观；
- 传播越远，权重衰减越多。

实际 LLM 会对每个 response token 做同样的递推，而不是对表中的三个大文本片段递推。

GAE 不是步骤正确性的真实标签。它只根据 critic 的前缀预测和最终 reward 分配信用；
critic 判断错时，GAE 也可能分错。

## 3.5 最后再看代码：`continue_mask` 只是轨迹边界开关

对一条含 \(T\) 个 response token 的 rollout，先准备：

- `reward[t]`：执行第 \(t\) 个 token 后得到的 reward；
- `old_value[t]`：生成该 token 之前，旧 critic 对前缀的预测；
- `old_value[t + 1]`：生成之后，新前缀的 value；
- `terminated[t]`：生成后是否真正终止。

`old` 表示这些 value 在 rollout 时已经算好。本批策略更新期间固定不变。

若已经真正终止，下一状态的 value 和后续 GAE 都不能再接进来。因此定义：

\[
\texttt{continue\_mask}_t
=
1-\texttt{terminated}_t.
\]

| 情况 | `terminated[t]` | `continue_mask` | 含义 |
|---|---:|---:|---|
| 同一条回答仍在继续 | 0 | 1 | 保留下一 value 和后续 GAE |
| 已生成 EOS 或环境宣布结束 | 1 | 0 | 同时切断二者 |

它只是一个乘法开关，不是新的强化学习量。完整递推为：

```python
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

`advantage` 给策略模型使用。它表示目标相对旧 value 高多少或低多少；把旧 value
加回去，就得到 critic 要拟合的绝对目标 `value_target`。

还要区分三种边界：

- **真正终止**：生成 EOS，或任务明确成功/失败，`continue_mask = 0`；
- **达到最大长度而截断**：任务理论上可能继续；若保留最后状态的 value，通常仍应
  bootstrap，不能自动当成真正终止；
- **padding**：只是 batch 对齐产生的填充 token，应由单独的 response mask 排除。

至此，一轮训练的顺序只有：

\[
\text{策略生成回答}
\to
\text{verifier 打分}
\to
\text{critic 预测各前缀 value}
\to
\text{TD/GAE 算 advantage}
\to
\text{更新策略与 critic}.
\]

课程把生成 token 的策略称为 **actor**，把预测前缀 value 的网络称为
**critic**。**Actor–Critic** 只是上面这个闭环的名称，没有额外冒出第三套算法。

本章只需要带走五句话：

1. return 是本次 rollout 真正得到的未来总分；
2. \(V(s)\) 是当前策略从该前缀继续生成的平均未来回报；
3. value 网络用大量已完成 rollout 的回报做监督学习；
4. TD 是相邻前缀之间的预期修正；
5. GAE 控制后续修正向前传播多远。

<details>
<summary>可选：为什么减去 baseline 不会系统性改错策略梯度？</summary>

固定状态 \(s\) 后，只依赖状态的 \(b(s)\) 不依赖本次动作。它乘上的 score function
在动作分布下期望为零：

\[
\begin{aligned}
\mathbb E_{a\sim\pi_\theta}
\left[b(s)\nabla_\theta\log\pi_\theta(a\mid s)\right]
&=
b(s)\sum_a\nabla_\theta\pi_\theta(a\mid s)\\
&=
b(s)\nabla_\theta1=0.
\end{aligned}
\]

所以 baseline 改变样本权重的波动，却不改变期望更新方向。若 baseline 偷看当前动作，
这条结论通常不再自动成立。

</details>
