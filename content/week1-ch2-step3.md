### 第三步：把“整条回答的 log 概率”拆成每一步

第一章已经从 LLM 的实际生成过程得到：

\[
p_\theta(\tau\mid x)
=
\prod_{t=1}^{T}
\pi_\theta(a_t\mid s_t).
\]

这里固定当前 prompt \(x\)；\(s_t\) 已经包含 prompt 和生成前缀，\(a_t\) 是实际采样的 token。乘积取对数后变成求和：

\[
\log p_\theta(\tau\mid x)
=
\sum_{t=1}^{T}
\log\pi_\theta(a_t\mid s_t).
\]

因此，对模型参数 \(\theta\) 求梯度：

\[
\nabla_\theta\log p_\theta(\tau\mid x)
=
\sum_{t=1}^{T}
\nabla_\theta\log\pi_\theta(a_t\mid s_t).
\]

这一步不再需要额外的初始状态分布或环境转移符号。prompt 来自固定训练题库，追加 token
也不包含可训练参数；模型能够改变的只有每一步 next-token 概率。

代回目标函数，得到策略梯度：

\[
\boxed{
\nabla_\theta J(\theta)
=
\mathbb E_{\tau\sim p_\theta}
\left[
R(\tau)
\sum_{t=1}^{T}
\nabla_\theta\log\pi_\theta(a_t\mid s_t)
\right]
}
\]

如果采样 \(N\) 条 rollout，期望可用样本平均近似：

\[
\nabla_\theta J(\theta)
\approx
\frac{1}{N}\sum_{i=1}^{N}
R(\tau^{(i)})
\sum_{t=1}^{T_i}
\nabla_\theta\log\pi_\theta
(a_t^{(i)}\mid s_t^{(i)}).
\]

完整逻辑链现在是：

```text
最大化模型反复生成时的平均回报
→ 把平均值展开成“每条回答的概率 × 它的回报”
→ 用 ∇log p = (1/p)∇p 改写成可以用采样估计的期望
→ 把整条回答的 log 概率拆成逐 token log 概率之和
→ 用回答回报加权每个已采样 token 的 log-prob 梯度
```

