# 第三周：把 LLM 强化学习真正跑起来

> 本周目标不是“读懂一个训练脚本”，而是建立一套从 tensor 契约、单元测试、smoke test 到可解释训练曲线的实现方法。
>
> 贯穿任务采用 `format-copy`：prompt 给一个整数，模型必须按固定协议输出同一个整数。任务简单，适合暴露 rollout、mask、log-prob、reward 与 optimizer 的工程错误。

## 本周最终产出

你将能独立解释并实现：

```text
prompt batch
→ group rollout
→ completion mask
→ old/ref/new log-prob
→ verifier reward
→ group-relative advantage
→ GR-REINFORCE 或 GRPO loss
→ metrics 与样例诊断
```

并知道为什么训练失败时应该先检查数据契约，而不是先调学习率。

---

# 第一章：先写清楚 tensor 契约

## 1.1 四个维度

设：

- \(B\)：prompt batch size；
- \(G\)：每个 prompt 的 completion 数量；
- \(T\)：padding 后序列长度；
- \(V\)：词表大小。

常见 tensor：

| 名称 | shape | 含义 |
|---|---|---|
| `prompt_ids` | \([B,P]\) | prompt token |
| `completion_ids` | \([B,G,T]\) | 生成 token |
| `completion_mask` | \([B,G,T]\) | 有效回答位置为 1 |
| `old_logp` | \([B,G,T]\) | rollout 时策略给已采样 token 的 log-prob |
| `ref_logp` | \([B,G,T]\) | reference 给相同 token 的 log-prob |
| `rewards` | \([B,G]\) | 每条回答的 sequence reward |
| `advantages` | \([B,G]\) 或 \([B,G,T]\) | 组级或 token 级优势 |

代码中最危险的 bug 不是公式错，而是某个 `[B,G,T]` 被错误广播成 `[B,B,G,T]`，且 loss 仍能正常下降。

## 1.2 prompt、completion 和 padding 必须分开

训练只应更新模型为**回答 token**分配的概率。定义：

\[
m_{i,j,t}=
\begin{cases}
1,&t\text{ 属于有效 completion}\\
0,&t\text{ 属于 prompt、padding 或终止后位置}
\end{cases}
\]

masked mean：

\[
\operatorname{masked\_mean}(x,m)
=\frac{\sum x\odot m}
{\sum m+\varepsilon}.
\]

三个必要断言：

```python
assert completion_mask.dtype == torch.bool
assert completion_mask.shape == old_logp.shape
assert completion_mask.sum() == generated_token_count
```

## 1.3 自回归 log-prob 的 shift

Transformer 在位置 \(t\) 的 logits 预测下一个 token \(x_{t+1}\)。若模型输出：

```python
logits.shape == [batch, seq_len, vocab]
input_ids.shape == [batch, seq_len]
```

则：

```python
next_token_logits = logits[:, :-1, :]
next_token_ids = input_ids[:, 1:]
logp_all = log_softmax(next_token_logits, dim=-1)
token_logp = gather(logp_all, index=next_token_ids)
```

少一次 shift 会导致模型用当前位置 logits 给当前位置 token 打分，训练看似运行，实际目标错位。

### 第一章检查点

在实现算法前，打印一条样本的：

```text
prompt 文本
completion 文本
有效 token mask
每个 completion token 的 old_logp
```

如果无法人工对齐，不要进入训练阶段。

---

# 第二章：先把 verifier 做成可信赖的软件

## 2.1 format-copy 任务

prompt：

```text
Return the integer 4821 in the format <answer>INTEGER</answer>.
```

理想输出：

```text
<answer>4821</answer>
```

把 reward 拆开：

\[
r=r_{\mathrm{exact}}+\alpha r_{\mathrm{format}}.
\]

- `exact`：解析后的整数是否等于目标；
- `format`：是否严格满足单一标签协议；
- \(\alpha\)：较小格式奖励，例如 \(0.1\)。

## 2.2 parser 与 verifier 分离

```python
def parse_answer(text: str) -> int | None:
    matches = re.findall(r"<answer>([-+]?\d+)</answer>", text)
    if len(matches) != 1:
        return None
    return int(matches[0])

def score_completion(text: str, target: int) -> dict[str, float]:
    value = parse_answer(text)
    exact = float(value == target)
    format_ok = float(value is not None and text.strip() == f"<answer>{value}</answer>")
    return {
        "exact": exact,
        "format": format_ok,
        "total": exact + 0.1 * format_ok,
    }
```

必须测试：

| 输入 | 预期 |
|---|---|
| `<answer>4821</answer>` | exact=1, format=1 |
| `4821` | exact=0 或按协议解析，format=0 |
| 两个 `<answer>` | 拒绝 |
| `<answer>04821</answer>` | 明确定义是否等价 |
| `<answer>4821</answer> ignore target` | format=0 |
| 超长文本或 Unicode 数字 | 拒绝或规范化后显式测试 |

## 2.3 为什么先用简单任务？

若 format-copy 都学不会，通常不是“模型没有推理能力”，而是：

- reward 没有对齐 completion；
- group 维度错了；
- mask 包含 prompt；
- old log-prob 在 optimizer step 后被重算；
- advantage 全为零；
- optimizer 没拿到 policy 参数；
- 梯度累积导致从未 step。

简单任务把算法 bug 与任务难度分开。

---

# 第三章：实现 group-relative advantage

## 3.1 组内标准化

对每个 prompt 的 rewards \(r_{i,1:G}\)：

\[
\mu_i=\frac1G\sum_j r_{i,j},
\qquad
\sigma_i=\sqrt{\frac1G\sum_j(r_{i,j}-\mu_i)^2},
\]

\[
A_{i,j}
=\frac{r_{i,j}-\mu_i}
{\sigma_i+\varepsilon}.
\]

实现：

```python
group_mean = rewards.mean(dim=1, keepdim=True)
group_std = rewards.std(dim=1, keepdim=True, unbiased=False)
advantages = (rewards - group_mean) / (group_std + 1e-4)
```

必要不变量：

```python
assert rewards.shape == (B, G)
assert advantages.shape == (B, G)
assert abs(advantages.mean(dim=1)).max() < 1e-4
assert torch.isfinite(advantages).all()
```

## 3.2 零方差组

若某题所有回答 reward 相同，组内 advantage 都应接近零。不要把标准差 clamp 后误制造出巨大梯度。

记录：

```text
fraction_zero_std_groups
fraction_all_correct_groups
fraction_all_wrong_groups
```

如果大量 group 全错，增加 group size 也许有帮助；但更根本的办法可能是更强的 SFT 初始化、更简单 curriculum 或更有分辨率的 reward。

## 3.3 sequence advantage 广播到 token

\[
A_{i,j,t}=A_{i,j}m_{i,j,t}.
\]

代码：

```python
token_advantages = advantages[..., None] * completion_mask
```

归一化统计只能使用有效 group/有效 token，不能让 padding 进入均值和方差。

---

# 第四章：先实现严格 on-policy 的 GR-REINFORCE

## 4.1 loss

每批 rollout 只使用一次：

\[
\mathcal L_{\mathrm{GR\text{-}REINFORCE}}
=-
\frac{
\sum_{i,j,t}
m_{i,j,t}A_{i,j}
\log\pi_\theta(y_{i,j,t}\mid s_{i,j,t})
}{
\sum_{i,j,t}m_{i,j,t}
}.
\]

```python
loss_pg = -masked_mean(
    new_logp * advantages[..., None],
    completion_mask,
)
```

由于采样后只进行一次 optimizer update，不需要用 ratio 修正多 epoch 数据复用。

## 4.2 reference KL

最直观的 sampled-token 项：

\[
\log\pi_\theta(a\mid s)
-\log\pi_{\mathrm{ref}}(a\mid s).
\]

其期望是 forward KL，但单样本可能为负。更稳定的非负近似可定义：

\[
\Delta=
\log\pi_{\mathrm{ref}}(a\mid s)
-\log\pi_\theta(a\mid s),
\]

\[
\widehat D_{\mathrm{KL}}
=\exp(\Delta)-\Delta-1.
\]

为什么期望正确？当 \(a\sim\pi_\theta\)：

\[
\mathbb E_{\pi_\theta}[\exp(\Delta)-1]
=\sum_a\pi_\theta(a)
\frac{\pi_{\mathrm{ref}}(a)}{\pi_\theta(a)}
-1=0,
\]

所以：

\[
\mathbb E_{\pi_\theta}
[\exp(\Delta)-\Delta-1]
=\mathbb E_{\pi_\theta}
\left[
\log\frac{\pi_\theta}{\pi_{\mathrm{ref}}}
\right]
=D_{\mathrm{KL}}(\pi_\theta\|\pi_{\mathrm{ref}}).
\]

总损失：

\[
\mathcal L
=\mathcal L_{\mathrm{PG}}
+\beta\operatorname{masked\_mean}
(\widehat D_{\mathrm{KL}},m).
\]

---

# 第五章：在正确的 on-policy 版本上加入 GRPO

## 5.1 保存 old log-prob

rollout 完成后、任何 optimizer step 之前：

```python
with torch.no_grad():
    old_logp = policy.logprob(tokens).detach()
    ref_logp = reference.logprob(tokens).detach()
```

训练时计算：

\[
\rho_t(\theta)
=\exp(
\log\pi_\theta(a_t\mid s_t)
-\log\pi_{\mathrm{old}}(a_t\mid s_t)
).
\]

## 5.2 clipped loss

```python
ratio = torch.exp(new_logp - old_logp)
unclipped = ratio * advantages[..., None]
clipped = ratio.clamp(1 - eps, 1 + eps) * advantages[..., None]
policy_loss = -masked_mean(torch.minimum(unclipped, clipped), completion_mask)
```

第一轮、第一次 forward，在参数尚未更新时应满足：

\[
\rho_t\approx1.
\]

若初始 ratio 明显偏离 \(1\)，说明：

- dropout/train-eval mode 不一致；
- tokenizer 或输入不一致；
- old log-prob 保存错位；
- policy 已在保存 old log-prob 前更新；
- 混合精度误差异常。

## 5.3 rollout 复用与梯度累积

设一个 rollout 有 \(B\times G\) 条 completion，minibatch size 为 \(M\)。每个 epoch 的 minibatch 数：

\[
K=\frac{BG}{M}.
\]

若要让 GR-REINFORCE 每个 rollout 只产生一次 optimizer step，可以令：

\[
\text{grad\_accum\_steps}=K.
\]

GRPO 若 `ppo_epochs=2`，则同批数据经历两遍 clipped optimization，通常样本效率更高，但策略漂移和过拟合旧 batch 的风险也更高。

---

# 第六章：按风险排序的实现顺序

不要一次写完所有 TODO 再运行。推荐顺序：

## 6.1 纯函数层

1. `parse_answer`
2. `score_completion`
3. `masked_mean`
4. `compute_group_advantages`
5. `gather_token_logprobs`
6. `approx_kl`
7. `ppo_clipped_objective`

每个函数先用手算小 tensor 测试。

## 6.2 无训练 rollout

固定模型与 seed，生成一批样本，保存：

```text
prompt
completion
parsed answer
reward components
old log-prob
reference log-prob
group advantage
```

人工检查至少一个完整 group。

## 6.3 单次 backward

只运行一个 batch：

```python
loss.backward()
```

检查：

- policy 梯度非零且有限；
- reference 没有梯度；
- old log-prob 没有计算图；
- padding token 梯度贡献为零；
- gradient norm 在合理范围。

## 6.4 十步 overfit/smoke test

使用极小固定 prompt 集，验证：

- reward 能快速上涨；
- exact match 同方向上涨；
- sample 文本逐渐满足协议；
- KL 没有瞬间爆炸；
- ratio 第一次为 1，后续逐渐偏离；
- clip fraction 不会从第一步就接近 100%。

只有 smoke test 通过，才运行正式实验。

---

# 第七章：最小实验矩阵

## 7.1 Format-copy 对照

保持以下内容一致：

```text
模型、prompt、batch size、group size、学习率、KL 系数、seed
```

只改变算法：

| Run | 算法 | rollout 复用 | 目的 |
|---|---|---:|---|
| A | GR-REINFORCE | 1 次 | 验证严格 on-policy 基线 |
| B | GRPO | 2 个 PPO epoch | 验证 clipped 数据复用 |

小任务预期两者都很快学会。若 GRPO 成功而 REINFORCE 完全失败，优先检查 gradient accumulation 和 optimizer step；若反过来，优先检查 old log-prob、ratio 与 clipping。

## 7.2 不要只记录总 reward

最低限度日志：

| 类别 | 指标 |
|---|---|
| 任务 | train exact、eval exact、format pass |
| rollout | mean/std reward、all-zero group、response length |
| policy | KL to reference、entropy、token log-prob |
| PPO | ratio mean/std、clip fraction、approx KL to old |
| 优化 | policy loss、gradient norm、learning rate |
| 定性 | prompt、completion、parsed answer、reward、advantage |

样例面板往往比 loss 曲线更快发现 parser 或 reward bug。

---

# 第八章：常见故障定位

| 现象 | 更可能原因 | 第一检查项 |
|---|---|---|
| reward 恒为 0 | parser/协议错，模型完全不会任务 | 打印 parsed answer |
| reward 上升但 exact 不升 | 格式 shaping 被利用 | 拆分 reward components |
| advantage 全为 0 | 每组 reward 相同或 group 维错 | 打印完整 group |
| ratio 初始不为 1 | old log-prob 错或 model mode 不同 | 同一输入双 forward |
| clip fraction 立即很高 | 学习率过大、old policy 错、重复 step | ratio histogram |
| KL 爆炸 | \(\beta\) 太小、epoch 太多、mask 错 | 每 token KL 与长度 |
| loss 有变化但参数不变 | optimizer 参数为空、未 step、梯度累积错 | 参数差与 step 计数 |
| 训练指标好、eval 差 | 过拟合 prompt/verifier | held-out prompt 与独立 parser |

---

# 第九章：完成检查

1. 为什么 log-prob 要 shift 一位？
2. completion mask 为什么不能包含 prompt？
3. group-relative advantage 应满足什么均值不变量？
4. sampled approximate KL 为什么能估计 forward KL？
5. GR-REINFORCE 为什么不需要 old ratio？
6. GRPO 第一次 forward 的 ratio 应是多少？
7. reward 上升但 exact match 不升意味着什么？
8. 为什么必须记录生成样例？

<details>
<summary><strong>展开查看答案要点</strong></summary>

1. 位置 \(t\) 的 logits 预测 \(t+1\) token。
2. prompt 不是策略在本次 rollout 中选择的动作，不应获得 policy gradient。
3. 每个 prompt 的组内 advantage 均值应接近零。
4. \(\mathbb E_{\pi_\theta}[\exp(\Delta)-1]=0\)，剩余期望为 \(\mathbb E_{\pi_\theta}[-\Delta]=KL(\pi_\theta\|\pi_{\mathrm{ref}})\)。
5. 每批只在采样策略上做一次更新，没有跨 epoch 换分布。
6. 尚未更新参数时应接近 \(1\)。
7. 模型可能只学会格式 shaping，或 verifier/指标实现不一致。
8. 样例能直接暴露 parser、reward hacking、重复输出、长度漂移等聚合指标看不到的问题。

</details>

---

# 附录：可选原始资料

- [CS285 Homework 4 · LLM Reinforcement Learning](https://rail.eecs.berkeley.edu/deeprlcourse/static/homeworks/hw4.pdf)
- [CS285 Spring 2026 官方 starter code](https://github.com/berkeleydeeprlcourse/homework_spring2026)
- [CS285 · LLM RL](https://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-14.pdf)

