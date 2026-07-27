# 第四章：为什么同一批回答训练几次后会失效

第三章已经为 rollout 中每个 token 算出了更新权重
\(\widehat A_t\)：正数表示这次采样比预期好，负数表示比预期差。

如果每批回答只更新一次，然后立刻重新生成，第二章的更新方式已经够用。但生成回答和调用 verifier 往往很贵，工程上自然会问：

> 能不能把同一批回答切成 minibatch，训练多个 epoch？

问题在于，第一次参数更新后，模型已经变了，回答却仍来自更新前的模型。本章只解决“怎样有限复用旧回答”。

## 4.1 第一次参数更新后，什么东西已经变了

考虑 rollout 中的一个前缀：

```text
计算 17×23。……所以答案是
```

采样时，模型给 token `391` 的概率为 0.20，并最终采到了它。训练代码会保存：

```text
token_id = "391"
sampling_logp = log(0.20)
advantage = +0.70
```

使用这批数据更新一次后，同一个模型在同一个前缀下可能已经给 `391` 概率
0.26。回答还是原来那条，但两件事已经不同：

| 对象 | `391` 的概率 | 角色 |
|---|---:|---|
| 生成这批数据时的模型快照 | 0.20 | 说明样本当初怎样产生 |
| 完成一次更新后的模型 | 0.26 | 现在正在继续优化 |

到这里再给两者命名：

- **old policy**：生成本批 rollout 时冻结下来的模型快照，记作
  \(\pi_{\text{old}}\)；
- **current policy**：当前正在更新的模型，记作 \(\pi_\theta\)。

在刚开始处理一批 rollout 时，二者参数相同。第一次 optimizer step 之后，
current policy 改变，old policy 仍保持不动。

这就是“数据变旧”的准确含义：

> token 是按 old policy 的概率采到的，却要继续用于更新 current policy。

## 4.2 怎样用旧回答估计当前模型下的结果

先用一个只有两个候选 token 的小例子，不引入抽象占位函数。

假设同一前缀下：

| 候选 token | old policy 概率 | current policy 概率 | 已算好的相对表现 |
|---|---:|---:|---:|
| `391` | 0.25 | 0.50 | \(+0.70\) |
| `381` | 0.75 | 0.50 | \(-0.30\) |

如果能直接按 current policy 重新采样，它的平均相对表现是：

\[
0.50\times0.70
+0.50\times(-0.30)
=
0.20.
\]

但现有数据是按 old policy 采的。直接按旧频率平均会得到：

\[
0.25\times0.70
+0.75\times(-0.30)
=
-0.05,
\]

这不是 current policy 下的结果，因为旧数据中 `391` 太少、`381` 太多。

修正办法从表中直接得到：

- `391` 在新模型中占 0.50，在旧数据中只占 0.25，因此它的旧样本权重乘
  \(0.50/0.25=2\)；
- `381` 在新模型中占 0.50，在旧数据中占 0.75，因此它的旧样本权重乘
  \(0.50/0.75=2/3\)。

修正后的旧数据平均为：

\[
0.25\times2\times0.70
+0.75\times\frac23\times(-0.30)
=
0.20.
\]

它与直接按 current policy 计算的结果相同。

现在才需要一个通用名字。对 rollout 中实际采到的 token \(a_t\)，把
“当前概率除以采样时概率”记作：

\[
\texttt{ratio}_t
=
\frac{
\pi_\theta(a_t\mid s_t)
}{
\pi_{\text{old}}(a_t\mid s_t)
}.
\]

它的读法很直接：

- `ratio = 1`：该 token 的概率没有变化；
- `ratio = 1.2`：当前概率是采样时的 1.2 倍；
- `ratio = 0.7`：当前概率是采样时的 0.7 倍。

训练时通常已经保存 old log-prob，因此稳定的计算方式是：

```python
ratio = exp(current_logp - old_logp)
```

用“新概率 / 旧概率”修正旧样本权重的方法叫
**importance weighting（重要性加权）**。这个名称到这里才有用；真正需要记住的是上面的两 token 算例。

## 4.3 为什么按这个概率比反复训练会失控

第三章给每个 token 的 \(\widehat A_t\) 在本批优化期间保持固定。旧样本对 current
policy 的更新贡献可以写成：

\[
\texttt{ratio}_t\,\widehat A_t.
\]

分正负看：

- 若 \(\widehat A_t>0\)，提高该 token 概率会增大 `ratio`，目标继续变好；
- 若 \(\widehat A_t<0\)，降低该 token 概率会减小 `ratio`，目标也继续变好。

仍看 `391`。采样时概率为 0.25，advantage 为 \(+0.70\)：

| current 概率 | ratio | `ratio × advantage` |
|---:|---:|---:|
| 0.25 | 1.0 | 0.70 |
| 0.50 | 2.0 | 1.40 |
| 0.75 | 3.0 | 2.10 |

若在同一批数据上不断训练，这个目标会持续鼓励模型把少数高分旧样本的概率推得更高。失败样本则会被持续压低。

问题不是“好 token 不该提高”，而是：

1. advantage 是有限 rollout 算出的估计，可能含噪声；
2. ratio 很大时，少数旧样本支配梯度；
3. current policy 变化太大后，old policy 访问到的前缀也不再代表当前模型会访问的前缀。

所以概率比只适合做局部修正，不能无限放大。

## 4.4 怎样停止从旧样本获得额外的便宜收益

选一个小正数 \(\epsilon\)。若 \(\epsilon=0.2\)，先把“本批旧数据仍可信的局部范围”设为：

\[
[1-\epsilon,1+\epsilon]=[0.8,1.2].
\]

这里不是说超出区间的概率变化在参数上绝对禁止，而是：

> 当更新已经沿 advantage 希望的方向走得足够远，就不再让这条旧样本提供额外收益。

先看正 advantage。若
\(\widehat A=+2\)、`ratio = 1.35`：

\[
\text{原贡献}=1.35\times2=2.70.
\]

把 ratio 限在上界 1.2 后：

\[
\text{限制后的贡献}=1.20\times2=2.40.
\]

优化时取较小的 2.40，因此继续把好 token 概率推高，不再从这条旧样本获得额外收益。

再看负 advantage。若
\(\widehat A=-2\)、`ratio = 0.70`：

\[
\text{原贡献}=0.70\times(-2)=-1.40,
\]

\[
\text{限制后的贡献}=0.80\times(-2)=-1.60.
\]

优化目标取较小的 \(-1.60\)，因此继续把坏 token 概率压低，也不会得到额外收益。

把一个数限制在上下界之间的操作叫 `clip`。把 batch 中 sampled token
取平均记作 \(\mathbb E_t\)。上面两种情况合并为：

\[
\mathcal L_{\text{reuse}}
=
\mathbb E_t\left[
\min\left(
\texttt{ratio}_t\widehat A_t,\;
\operatorname{clip}
(\texttt{ratio}_t,1-\epsilon,1+\epsilon)
\widehat A_t
\right)
\right].
\]

\(\mathcal L_{\text{reuse}}\) 只是给“有限复用旧样本时要最大化的量”起的名字。代码通常最小化它的负数。

到这里，才给这套方法正式命名：

> **PPO（Proximal Policy Optimization）** 用概率比修正旧样本，再用悲观的
> clipped objective 限制同一批数据能推动策略多远。

`min` 的作用不是把所有 ratio 强行裁进区间。若模型朝 advantage 不希望的方向移动，原始分支仍保留惩罚；它只截断“沿有利方向走太远后继续获益”的部分。

## 4.5 限制 sampled token 仍不等于限制整个模型

PPO clip 只查看：

- old policy 访问到的前缀；
- 这些前缀下实际采到的 token；
- 这些 sampled token 的 current/old 概率比。

但一次 Transformer 参数更新还会改变：

- 同一前缀下没有被采到的其他词表 token；
- 本批 rollout 没有访问过的其他前缀；
- 模型在通用任务上的行为。

因此还需要一个比较**完整 next-token 分布变化**的量。

对固定前缀 \(s\)，令 \(\mathcal V\) 表示整个 token 词表。把 old policy
分布与 current policy 分布的差异定义为：

\[
D_{\mathrm{KL}}
\left(
\pi_{\text{old}}(\cdot\mid s)
\;\|\;
\pi_\theta(\cdot\mid s)
\right)
=
\sum_{v\in\mathcal V}
\pi_{\text{old}}(v\mid s)
\log
\frac{
\pi_{\text{old}}(v\mid s)
}{
\pi_\theta(v\mid s)
}.
\]

这个量叫 **KL divergence（KL 散度）**：

- 两个分布完全相同时为 0；
- old policy 认为重要的 token 在 current policy 中变化越大，数值通常越大；
- 竖线两侧不能随意交换，KL 不是对称距离。

old–current KL 回答：

> 本轮 current policy 已经离生成这批数据的快照走了多远？

工程上至少同时监控：

| 指标 | 回答什么 |
|---|---|
| ratio 的均值、最大值和分位数 | sampled token 的概率变化是否失控 |
| clip fraction | 有多少 token 已经碰到局部复用边界 |
| old–current KL | 完整 token 分布离 rollout 快照多远 |
| entropy | 生成分布是否过早变得确定 |

达到 KL 阈值时可以提前停止当前批次的 epoch，丢弃旧 rollout，再用最新策略生成。

还有一个不同的长期问题：即使每轮 current 都没有离 old 太远，很多轮小更新累积后，模型仍可能远离最初的语言行为。因此 LLM 后训练常再冻结一份
SFT 起点作为 **reference policy**，记作 \(\pi_{\text{ref}}\)。

两种比较不能混淆：

| 比较对象 | 生命周期 | 目的 |
|---|---|---|
| old ↔ current | old 每轮刷新 | 判断当前 rollout 是否已经过期 |
| current ↔ reference | reference 长期冻结 | 判断模型是否远离初始行为锚点 |

完整的数据复用循环现在是：

```text
复制 current 得到 old
→ old 生成 rollout，并保存 old_logp
→ 计算 reward、value 和 advantage
→ 用 ratio + clipping 做少量 epoch
→ 监控 old-current KL
→ 丢弃 rollout，用新的 current 再采样
```

第五章会把这条循环与 critic 更新、reference 约束放进同一段可执行伪代码。

### 第四章检查点

读完后应能独立完成：

1. 用两 token 表格解释为什么旧样本需要乘 current/old 概率比；
2. 给定 old/current 概率，手算 ratio；
3. 分 advantage 正负说明哪一侧停止提供额外收益；
4. 解释 PPO clip 为什么不是参数或 KL 的硬约束；
5. 区分 old policy 与 reference policy 的刷新周期和用途。
