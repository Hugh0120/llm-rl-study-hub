# 第 2 章：最终分数怎样改变已生成 token 的概率

## 2.1 本章符号说明

| 符号 | English | 本章含义 |
|---|---|---|
| \(J(\theta)\) | objective | 当前策略的期望回报 |
| \(p_\theta(\tau)\) | trajectory probability | 当前模型生成轨迹 \(\tau\) 的概率 |
| \(R(\tau)\) | trajectory return | 整条回答的回报 |
| \(\nabla_\theta\) | gradient | 对模型参数 \(\theta\) 求梯度 |
| \(\log\pi_\theta(a_t\mid s_t)\) | sampled-token log-prob | 实际采样 token 的对数概率 |
| \(G_t\) | reward-to-go | 从第 \(t\) 步开始的累计后续 reward |
| \(N\) | batch size | 本批 rollout 数量 |

## 2.2 本章目标

读完本章，你应该能够：

1. 从期望回报逐行推出 policy gradient；
2. 解释为什么对 log-prob 求导，而不是对 token id 求导；
3. 把整条轨迹的 log-prob 拆成逐 token log-prob；
4. 从梯度上升写成框架中最小化的 loss；
5. 解释正、负、零训练权重分别怎样改变采样 token 概率；
6. 指出原始 REINFORCE 的两个结构性缺陷。

## 2.3 本章主线

第 1 章已经定义了训练目标：

\[
J(\theta)=\mathbb E_{\tau\sim p_\theta}[R(\tau)].
\]

现在的困难非常具体：

> \(\tau\) 由离散 token 组成。token id 不能对模型参数求导，但生成这些 token 的概率可以求导。

本章只完成“分数到梯度”的转换。得到梯度后，我们再观察它为什么仍然噪声很大。

## 2.4 本章新增概念

| 名词 | 中文 | 在本章中的作用 |
|---|---|---|
| score-function estimator | 得分函数估计 | 把采样分布的梯度改写成可由样本估计的期望 |
| log-derivative trick | 对数导数技巧 | 使用 \(\nabla p=p\nabla\log p\) |
| Monte Carlo estimate | 蒙特卡洛估计 | 用有限条真实 rollout 近似期望 |
| policy gradient | 策略梯度 | 期望回报关于策略参数的梯度 |
| REINFORCE | 经典采样策略梯度方法 | 直接用采样 return 加权 log-prob 梯度 |
| reward-to-go | 后续回报 | 只让动作对它发生后的 reward 负责 |

## 2.5 第一步：把期望写成所有回答的概率加权和

先固定一个 prompt，暂时省略对题库的平均。期望回报是：

\[
J(\theta)
=
\sum_\tau p_\theta(\tau)R(\tau).
\]

对参数求梯度：

\[
\nabla_\theta J(\theta)
=
\sum_\tau R(\tau)\nabla_\theta p_\theta(\tau).
\]

这里假设判题程序本身固定，\(R(\tau)\) 不直接依赖模型参数。参数影响的是“哪些回答更容易被生成”。

问题是：右边还不是一个可以直接通过采样求平均的形式，因为缺少概率权重 \(p_\theta(\tau)\)。

## 2.6 第二步：把概率梯度改写成 log 概率梯度

对任意正概率：

\[
\nabla_\theta\log p_\theta(\tau)
=
\frac{1}{p_\theta(\tau)}
\nabla_\theta p_\theta(\tau).
\]

两边乘以 \(p_\theta(\tau)\)：

\[
\nabla_\theta p_\theta(\tau)
=
p_\theta(\tau)
\nabla_\theta\log p_\theta(\tau).
\]

这条恒等变换叫 **log-derivative trick（对数导数技巧）**。它不是近似，也没有添加新假设。

代回目标梯度：

\[
\begin{aligned}
\nabla_\theta J(\theta)
&=
\sum_\tau
p_\theta(\tau)R(\tau)
\nabla_\theta\log p_\theta(\tau)\\
&=
\mathbb E_{\tau\sim p_\theta}
\left[
R(\tau)\nabla_\theta\log p_\theta(\tau)
\right].
\end{aligned}
\]

现在右边重新变成了一个期望，所以可以让当前模型生成一批回答，用样本平均估计。

## 2.7 第三步：把整条回答的 log 概率拆开

第 1 章得到：

\[
p_\theta(\tau)
=
\prod_{t=1}^{T}
\pi_\theta(a_t\mid s_t).
\]

乘积取 log 后变成求和：

\[
\log p_\theta(\tau)
=
\sum_{t=1}^{T}
\log\pi_\theta(a_t\mid s_t).
\]

因此：

\[
\nabla_\theta\log p_\theta(\tau)
=
\sum_{t=1}^{T}
\nabla_\theta
\log\pi_\theta(a_t\mid s_t).
\]

代回：

\[
\boxed{
\nabla_\theta J(\theta)
=
\mathbb E
\left[
R(\tau)
\sum_{t=1}^{T}
\nabla_\theta\log\pi_\theta(a_t\mid s_t)
\right]
}
\]

从左到右读：

> 一条回答得到多高回报，就用多大权重推动这条回答中实际采样 token 的 log-prob。

这就是 policy gradient 的最基本采样形式。

## 2.8 为什么 log-prob 梯度会提高或降低采样 token

设模型在某个位置输出 logits \(z_j\)，softmax 概率为 \(\pi(j\mid s)\)。对本次采样 token \(a\)：

\[
\frac{\partial\log\pi(a\mid s)}
{\partial z_j}
=
\mathbf 1[j=a]-\pi(j\mid s).
\]

\(\mathbf 1[j=a]\) 是指示函数：当 \(j\) 正好是采样 token 时取 1，否则取 0。

因此：

- 对采样 token，导数是 \(1-\pi(a\mid s)>0\)；
- 对其他 token，导数是 \(-\pi(j\mid s)<0\)。

再乘训练权重 \(w\)：

| 权重 \(w\) | 梯度上升的结果 |
|---:|---|
| \(w>0\) | 提高采样 token 相对其他 token 的概率 |
| \(w<0\) | 降低采样 token 的概率 |
| \(w=0\) | 这条样本没有策略梯度 |

“奖励好动作、压低坏动作”不是人为添加的规则，而是 `weight × log-prob gradient` 的直接结果。

## 2.9 Reward-to-go：每个动作只对未来负责

如果中间也有 reward，第 \(t\) 个动作不能影响已经发生的过去奖励。它只应由当前与未来 reward 加权。

从第 \(t\) 步开始的累计回报定义为：

\[
G_t
=
\sum_{k=t}^{T}
\gamma^{k-t}r_k.
\]

这个量叫 **reward-to-go**。

于是更精确的策略梯度写成：

\[
\nabla_\theta J(\theta)
=
\mathbb E
\left[
\sum_{t=1}^{T}
\gamma^{t-1}G_t
\nabla_\theta
\log\pi_\theta(a_t\mid s_t)
\right].
\]

只有终局 0/1 reward 且 \(\gamma=1\) 时，同一条回答中所有 \(G_t\) 相同。后面会解释为什么这仍然太粗。

## 2.10 从梯度上升变成可最小化的 loss

深度学习框架通常做梯度下降。为了最大化 \(J\)，最小化它对应的负目标。

对 \(N\) 条 rollout：

\[
\mathcal L_{\text{policy}}
=
-
\frac{1}{N}
\sum_{i=1}^{N}
\sum_{t=1}^{T_i}
G_t^{(i)}
\log\pi_\theta
\left(
a_t^{(i)}\mid s_t^{(i)}
\right).
\]

这套“用采样 return 加权采样动作 log-prob”的方法叫 **REINFORCE**。

最小实现：

```python
loss = -masked_mean(
    reward_to_go.detach() * sampled_token_logp,
    response_mask,
)
loss.backward()
optimizer.step()
```

`detach()` 的含义是：reward 是训练权重，不沿判题程序反向传播；梯度只通过 `sampled_token_logp` 回到语言模型。

## 2.11 贯穿例子：为什么 0/1 reward 仍然能学

当前模型只生成两类回答：

| 回答 | 当前概率 | reward |
|---|---:|---:|
| `\boxed{391}` | 0.30 | 1 |
| `\boxed{381}` | 0.70 | 0 |

期望回报：

\[
J=0.30.
\]

如果本批采到正确回答，它的 token 得到正权重，正确回答概率会上升。

如果采到错误回答，reward 为 0，权重也是 0，这条样本不更新。

反复采样后，只要正确回答偶尔出现，就会被逐渐增强。因此 REINFORCE 已经解决：

> 没有标准 token 标签，只有最终判分时，怎样让高分回答更容易再次出现。

## 2.12 原始 REINFORCE 还留下什么问题

刚才的例子同时暴露两个问题。

### 2.12.1 零分不是负反馈

错误回答得到 0 时，只是没有更新，并不会被直接压低。

如果把 reward 人为改成 `正确 +1、错误 -1`，虽然能产生负更新，但绝对分数仍没有考虑题目难度和当前模型水平。

### 2.12.2 同一分数在不同情境中含义不同

| prompt | 当前模型通常答对率 | 本次 reward |
|---|---:|---:|
| 简单题 | 90% | 1 |
| 难题 | 5% | 1 |

两次都得 1 分，但难题成功更值得强调。

真正有信息量的权重应该是：

\[
\text{本次实际结果}
-
\text{当前情境下的正常结果}.
\]

这就把我们带到下一章：

> “当前前缀下通常能拿多少分”怎样定义、估计并变成逐 token 信号？

## 2.13 工程检查点

实现原始 REINFORCE 时先检查：

1. `sampled_token_logp` 只 gather 实际采样 token；
2. prompt 与 padding 位置不参与 policy loss；
3. reward、return 和 advantage 类权重全部 detach；
4. 正 reward 样本更新后，其采样 token log-prob 应上升；
5. 负权重样本更新后，其 log-prob 应下降；
6. 训练数据必须来自当前策略，不能长期复用而不校正。

## 2.14 本章自测

1. 为什么不能直接对 token id 求导？
2. \(\nabla p=p\nabla\log p\) 在推导中解决了什么？
3. 整条回答的 log-prob 为什么可以拆成逐 token 求和？
4. reward-to-go 为什么比整条轨迹 return 更符合因果性？
5. 原始 REINFORCE 为什么方差高？
6. 0/1 reward 中，错误样本为什么可能没有梯度？

## 2.15 对应官方资料

- [CS285 Lecture 5 · Policy Gradients](https://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-5.pdf)
- [CS285 Fall 2023 · Lecture 5](https://www.youtube.com/playlist?list=PL_iWQOsE6TfVYGEGiAOMaOzzv41Jfm_Ps&index=5)
