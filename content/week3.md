# 第三周：把 LLM 强化学习实现成一条可验证的数据流水线

> 本周不是算法名词复习，而是一份实现教材。目标是把第二周的 GR-REINFORCE 和 GRPO 落成张量、纯函数、单元测试和可诊断训练循环。

## 本周最终产出

你将实现并验证：

1. 自回归 token log-prob 对齐；
2. prompt、completion、padding 的严格 mask；
3. verifier 与 reward 分项；
4. 同 prompt 分组与 group advantage；
5. 非负的 sampled reference KL；
6. 一次更新的 GR-REINFORCE；
7. 可有限复用 rollout 的 GRPO；
8. 从单元测试到十步 smoke test 的调试阶梯。

先记住唯一总流程：

\[
\text{prompt}
\rightarrow
\text{rollout}
\rightarrow
\text{token log-prob}
\rightarrow
\text{verifier reward}
\rightarrow
\text{group advantage}
\rightarrow
\text{loss}
\rightarrow
\text{下一批 rollout}.
\]

每个箭头都必须能单独测试。

# 第一章：先写 tensor 契约，再写一行 loss

## 1.1 四个维度

设：

- \(P\)：本批不同 prompt 数；
- \(G\)：每个 prompt 采样的 completion 数；
- \(B=P\times G\)：总序列数；
- \(L\)：padding 后序列长度；
- \(V\)：词表大小。

模型输出一个三维实数张量。数学上写作
\(\mathbb R^{B\times L\times V}\)：\(\mathbb R\) 表示元素是实数，
右上角三个维度依次是 batch、序列位置和词表；这里的 \(\mathbb R\)
不是 reward。于是：

\[
\texttt{logits}\in\mathbb R^{B\times L\times V}.
\]

rollout 相关张量：

| 张量 | shape | 含义 |
|---|---|---|
| `input_ids` | \([B,L]\) | prompt + completion + padding |
| `attention_mask` | \([B,L]\) | 非 padding 位置 |
| `completion_mask` | \([B,L]\) | 只标 completion token |
| `old_logp` | \([B,L-1]\) | rollout 策略对目标 token 的 log-prob |
| `ref_logp` | \([B,L-1]\) | reference 对同一目标 token 的 log-prob |
| `reward` | \([B]\) | 每条 completion 的序列 reward |
| `advantage` | \([B]\) | 同 prompt 组内相对表现 |
| `group_id` | \([B]\) | 每条序列属于哪个 prompt |

为什么 log-prob 是 \(L-1\)？下一节从自回归预测定义推出。

## 1.2 第 \(t\) 个位置预测的是第 \(t+1\) 个 token

假设序列是：

```text
[BOS, 求, 解, :, 3, 9, 1, EOS]
```

位置关系：

| 模型读取到的前缀末尾 | 该位置 logits 的目标 |
|---|---|
| `BOS` | `求` |
| `求` | `解` |
| `解` | `:` |
| `3` | `9` |
| `9` | `1` |
| `1` | `EOS` |

因此：

```python
logits = model(
    input_ids=input_ids,
    attention_mask=attention_mask,
).logits                              # [B, L, V]

predictive_logits = logits[:, :-1, :] # [B, L-1, V]
targets = input_ids[:, 1:]             # [B, L-1]

token_logp = predictive_logits.log_softmax(-1).gather(
    dim=-1,
    index=targets.unsqueeze(-1),
).squeeze(-1)                          # [B, L-1]
```

如果拿 `logits[:, 1:]` 对 `input_ids[:, 1:]`，就是让模型看到了目标 token 本身，语义错位。

## 1.3 completion mask 也必须跟着目标右移

设原始 `completion_mask[:, j]=1` 表示 `input_ids[:, j]` 属于 completion。`token_logp[:, j-1]` 才对应这个 token，所以：

```python
token_mask = completion_mask[:, 1:] * attention_mask[:, 1:]
```

这张 mask 必须同时排除：

- prompt token：不应该被 RL loss 更新；
- padding：不是模型动作；
- padding 后虚构的转移；
- 必要时，是否包含 EOS 要由任务契约明确决定。

### 最小人工测试

```text
input_ids:       [P0, P1, C0, C1, EOS, PAD]
completion_mask: [ 0,  0,  1,  1,   1,   0]
token_mask:          [0,  1,  1,   1,   0]
targets:             [P1, C0, C1, EOS, PAD]
```

`token_mask` 中第一个 1 对应目标 `C0`。如果它对应 `P1` 或 `C1`，shift 就错了。

## 1.4 先固定 masked mean 的定义

```python
def masked_mean(x, mask, dim=None, eps=1e-8):
    mask = mask.to(x.dtype)
    numerator = (x * mask).sum(dim=dim)
    denominator = mask.sum(dim=dim).clamp_min(eps)
    return numerator / denominator
```

把第 \(i\) 条序列在所有有效 completion token 上的平均 log-prob
记作 \(\bar\ell_i\)：横线表示平均，\(\ell\) 是 log-likelihood
的惯用字母，下标 \(i\) 表示第 \(i\) 条序列。定义为：

\[
\bar\ell_i
=
\frac{\sum_t m_{i,t}\log\pi(a_{i,t}\mid s_{i,t})}
{\sum_t m_{i,t}}.
\]

```python
sequence_logp = masked_mean(token_logp, token_mask, dim=-1) # [B]
```

这一步明确了长度语义。若改用 token sum，长 completion 天然有更大绝对 loss；这不是无害重构，而是目标改变。

### 第一章验收

- [ ] 所有函数入口都有 shape 注释；
- [ ] shift 用一个 6-token 玩具序列测试过；
- [ ] prompt 和 padding 对 loss 的贡献严格为 0；
- [ ] 空 completion 会被数据层拒绝，而不是靠除数钳位静默吞掉。

# 第二章：rollout 是一份有生命周期的数据快照

## 2.1 为什么必须有 old policy 快照

生成数据时的策略记为 \(\pi_{\text{old}}\)。GRPO 优化阶段会改变 current policy，因此必须缓存：

\[
\log\pi_{\text{old}}(a_t\mid s_t).
\]

它不是训练参数，而是这份 rollout 的“出生证明”。

一个最小 rollout batch：

```python
@dataclass
class RolloutBatch:
    input_ids: Tensor          # [B, L]
    attention_mask: Tensor     # [B, L]
    completion_mask: Tensor    # [B, L]
    old_logp: Tensor           # [B, L-1], detached
    ref_logp: Tensor           # [B, L-1], detached
    rewards: Tensor            # [B], detached
    advantages: Tensor         # [B], detached
    group_ids: Tensor          # [B]
```

## 2.2 一批数据的生命周期

```text
冻结 current 为 rollout snapshot
        ↓
每个 prompt 采样 G 个 completion
        ↓
缓存 old_logp 与 ref_logp
        ↓
verifier 得到 reward
        ↓
按 prompt 算 advantage
        ↓
做 1 次或 K 个 epoch 更新
        ↓
整批作废，重新 rollout
```

进入优化器之前：

```python
old_logp = old_logp.detach()
ref_logp = ref_logp.detach()
rewards = rewards.detach()
advantages = advantages.detach()
```

若 `old_logp.requires_grad=True`，ratio 的分母也在移动，已经不是 PPO/GRPO 的旧策略比值。

## 2.3 分组顺序不能靠“我记得是连续的”

推荐显式保留 `group_ids`，或者在构造 batch 时建立严格断言：

```python
assert batch_size == num_prompts * group_size
assert torch.equal(
    group_ids,
    torch.arange(num_prompts).repeat_interleave(group_size),
)
```

若 dataloader shuffle 后仍直接：

```python
rewards.view(num_prompts, group_size)
```

很可能把不同 prompt 的回答放进同一组，训练仍能运行，但 advantage 已无意义。

## 2.4 reference policy 与 old policy再次分清

| 缓存 | 来源 | 生命周期 | 参与哪个量 |
|---|---|---|---|
| `old_logp` | rollout 时 current snapshot | 一批 rollout | current/old ratio |
| `ref_logp` | 固定 reference | 多轮训练 | current/reference KL |

reference 可以在 rollout 时一次算好，因为它冻结；current log-prob 必须在每个优化 step 重算。

# 第三章：先把 verifier 做成可信软件

## 3.1 从 format-copy smoke task 开始

复杂数学任务同时包含生成、解析、reward、分组和优化，出错后很难定位。先用简单任务：

```text
prompt: 把字符串 "blue-17" 放进 <answer>...</answer>。
target: <answer>blue-17</answer>
```

可按 CS285 Homework 4 的思路拆分 reward：

| 条件 | 分项 reward |
|---|---:|
| 完全精确匹配 | \(+1.0\) |
| 包含 `<answer>` 标签 | \(+0.2\) |
| 严格满足 XML 结构 | \(+0.1\) |

关键不是具体系数，而是每一项单独记录：

```python
RewardBreakdown(
    exact_match=...,
    answer_tag=...,
    strict_format=...,
)
```

只记录总 reward，会让你分不清模型学会了任务，还是只学会了格式。

## 3.2 parser、verifier、reward 聚合必须分层

```python
def parse_answer(text: str) -> ParseResult:
    ...

def verify_answer(parsed: ParseResult, target: str) -> VerifyResult:
    ...

def compute_reward(parsed, verified) -> RewardBreakdown:
    ...
```

三层职责：

- parser：文本里提取到了什么；
- verifier：提取结果是否正确；
- reward：训练要如何加权这些事件。

同一个 parser 应同时服务训练和评估，或明确版本化。否则训练认为正确、评估认为错误的样本会制造假信号。

## 3.3 verifier 的纯函数测试

至少覆盖：

```text
1. 完全正确
2. 答案正确但格式松散
3. 格式正确但答案错误
4. 没有标签
5. 两个冲突答案
6. 空输出
7. 极长输出
8. 特殊字符与非法数值
```

断言的不只是总分：

```python
assert result.parse_ok is True
assert result.format_ok is False
assert result.answer_ok is True
assert result.total == expected
```

## 3.4 从序列 reward 到 group advantage

得到 `rewards: [B]` 后，按 prompt 组成：

\[
R\in\mathbb R^{P\times G}.
\]

```python
grouped = rewards.reshape(num_prompts, group_size)
mean = grouped.mean(dim=1, keepdim=True)
std = grouped.std(dim=1, keepdim=True, unbiased=False)
group_adv = (grouped - mean) / (std + 1e-8)
advantages = group_adv.reshape(batch_size)
```

必须验证：

```python
assert torch.isfinite(advantages).all()
assert advantages.shape == rewards.shape
```

对于非零方差组：

```python
assert torch.allclose(
    group_adv.mean(dim=1),
    torch.zeros(num_prompts),
    atol=1e-5,
)
```

零方差组的 advantage 全为 0 是正确行为，不是数值 bug。

# 第四章：把 token 概率与 reference KL 算对

## 4.1 只 gather 实际采样 token

不需要保存完整：

\[
[B,L,V]
\]

的 log-softmax 到 rollout buffer。只 gather 实际 token 后得到：

\[
[B,L-1].
\]

```python
def token_logprobs(model, input_ids, attention_mask):
    logits = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    ).logits[:, :-1, :]
    targets = input_ids[:, 1:]
    return logits.log_softmax(-1).gather(
        -1, targets.unsqueeze(-1)
    ).squeeze(-1)
```

单元测试应使用手工 logits。比如某位置目标 token 的 softmax 概率恰为 0.25，输出就应为：

\[
\log 0.25\approx-1.3863.
\]

## 4.2 sampled KL 为什么用这个形式

在当前策略采到的 token 上，先计算 reference log-prob 减去 current
log-prob。把这个差记作 \(\Delta_t\)：大写希腊字母 delta 常用来表示差，
下标 \(t\) 表示 token 位置：

\[
\Delta_t
=
\log\pi_{\text{ref}}(a_t\mid s_t)
-\log\pi_\theta(a_t\mid s_t).
\]

再把这个差转换成 sampled KL contribution。记作
\(\widehat k_t\)：\(k\) 表示 KL contribution，帽子表示采样估计：

\[
\widehat k_t=e^{\Delta_t}-\Delta_t-1.
\]

代码：

```python
def sampled_kl(current_logp, ref_logp):
    delta = ref_logp - current_logp
    return torch.exp(delta) - delta - 1.0
```

应验证三件事：

1. current 与 reference 相同，\(\widehat k=0\)；
2. 任意有限输入，\(\widehat k\ge0\)；
3. 梯度只能流向 `current_logp`。

```python
kl = sampled_kl(current_logp, ref_logp.detach())
kl_loss = masked_mean(kl, token_mask)
```

## 4.3 sampled KL 与 full-vocabulary KL 的区别

full KL 在每个状态对整个词表求和：

\[
\sum_a
\pi_\theta(a\mid s)
\log\frac{\pi_\theta(a\mid s)}
{\pi_{\text{ref}}(a\mid s)}.
\]

sampled KL 只用实际采到的 token，是 Monte Carlo 估计，计算便宜但有采样噪声。日志里应命名为 `sampled_ref_kl`，避免让人误以为做了全词表精确求和。

# 第五章：先实现一次更新的 GR-REINFORCE

## 5.1 为什么先做它

GR-REINFORCE 不需要 old/current ratio，也不需要复用多个 epoch。它最适合检验：

- generation 正常吗；
- verifier 正常吗；
- group advantage 正常吗；
- log-prob、mask、梯度符号正常吗。

如果它都不能在 format-copy 上学习，直接加 GRPO 只会增加变量。

## 5.2 loss 从序列定义到张量

序列平均 log-prob：

\[
\bar\ell_i
=
\frac{\sum_t m_{i,t}\log\pi_\theta(a_{i,t}\mid s_{i,t})}
{\sum_t m_{i,t}}.
\]

策略 loss 记作 \(\mathcal L_{\text{pg}}\)：\(\mathcal L\) 表示 loss，
下标 `pg` 表示 policy gradient：

\[
\mathcal L_{\text{pg}}
=
-\frac1B\sum_i A_i\bar\ell_i.
\]

```python
current_logp = token_logprobs(policy, input_ids, attention_mask)
sequence_logp = masked_mean(current_logp, token_mask, dim=-1)
pg_loss = -(advantages * sequence_logp).mean()

kl_token = sampled_kl(current_logp, ref_logp)
kl_loss = masked_mean(kl_token, token_mask)

loss = pg_loss + beta * kl_loss
```

## 5.3 用符号检查梯度方向

单条样本：

\[
\mathcal L_i=-A_i\log\pi_\theta(y_i\mid x_i).
\]

- 若 \(A_i>0\)，最小化 loss 会提高该序列 log-prob；
- 若 \(A_i<0\)，会降低它；
- 若 \(A_i=0\)，策略项无梯度，只剩 KL。

写一个两样本 toy test：固定 \(A=[1,-1]\)，做一次很小的 optimizer step，断言正样本 log-prob 上升、负样本下降。

## 5.4 一批只能做一次 policy update

```python
rollout = collect_rollout(policy)
loss = gr_reinforce_loss(policy, reference, rollout)
optimizer.zero_grad()
loss.backward()
optimizer.step()
# rollout 到此作废
```

若对同一批数据循环多次，却没有保存 `old_logp` 和 ratio 修正，就已离开推导中的 on-policy 条件。

# 第六章：确认基线正确后，再加入 GRPO

## 6.1 复用 rollout 需要什么新数据

只新增一份冻结快照：

\[
\texttt{old\_logp}
=
\log\pi_{\text{old}}(a_t\mid s_t).
\]

每个优化 step 重算：

\[
\texttt{current\_logp}
=
\log\pi_\theta(a_t\mid s_t).
\]

比值在 log 空间计算：

```python
log_ratio = current_logp - old_logp
ratio = torch.exp(log_ratio)
```

`old_logp` 若在每个 epoch 重新计算，ratio 会反复回到 1，clipping 失去意义。

## 6.2 clipped token objective

序列级 advantage 广播到 token：

```python
token_adv = advantages[:, None]        # [B, 1] -> broadcast
unclipped = ratio * token_adv
clipped = ratio.clamp(1-eps, 1+eps) * token_adv
token_objective = torch.minimum(unclipped, clipped)
pg_loss = -masked_mean(token_objective, token_mask)
```

再加 reference KL：

```python
kl_loss = masked_mean(
    sampled_kl(current_logp, ref_logp),
    token_mask,
)
loss = pg_loss + beta * kl_loss
```

注意是两组不同的 log-prob 差：

```text
current - old  → ratio / clipping / rollout 复用
ref - current  → sampled reference KL / 长期锚定
```

## 6.3 一个完整训练轮次

```python
for train_step in range(num_train_steps):
    # 1. 采样阶段：policy 此时就是 old policy
    with torch.no_grad():
        sequences = generate_grouped(
            policy,
            prompts,
            group_size=G,
        )
        old_logp = token_logprobs(
            policy, sequences.ids, sequences.attention_mask
        )
        ref_logp = token_logprobs(
            reference, sequences.ids, sequences.attention_mask
        )
        rewards, reward_parts = verifier_batch(sequences)
        advantages = group_advantages(
            rewards, sequences.group_ids
        )

    rollout = RolloutBatch(
        input_ids=sequences.ids,
        attention_mask=sequences.attention_mask,
        completion_mask=sequences.completion_mask,
        old_logp=old_logp.detach(),
        ref_logp=ref_logp.detach(),
        rewards=rewards.detach(),
        advantages=advantages.detach(),
        group_ids=sequences.group_ids,
    )

    # 2. 优化阶段：只在这批数据上有限复用
    for epoch in range(ppo_epochs):
        for mb in shuffled_minibatches(rollout):
            current_logp = token_logprobs(
                policy, mb.input_ids, mb.attention_mask
            )
            mask = mb.completion_mask[:, 1:]

            ratio = torch.exp(current_logp - mb.old_logp)
            adv = mb.advantages[:, None]
            surr1 = ratio * adv
            surr2 = ratio.clamp(1-eps, 1+eps) * adv
            pg_loss = -masked_mean(torch.minimum(surr1, surr2), mask)

            ref_kl = sampled_kl(current_logp, mb.ref_logp)
            kl_loss = masked_mean(ref_kl, mask)
            loss = pg_loss + beta * kl_loss

            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(policy.parameters(), max_grad_norm)
            optimizer.step()

    # 3. 本批作废；下一轮重新生成
```

## 6.4 minibatch 不能破坏序列和分组语义

group advantage 在完整 rollout 上先算好，之后 minibatch 可以按序列 shuffle。不要在每个 minibatch 内重新算组均值，因为同一 prompt 的 \(G\) 个样本可能被拆散。

正确顺序：

\[
\text{完整分组 reward}
\rightarrow
\text{冻结 advantage}
\rightarrow
\text{序列级 minibatch}.
\]

## 6.5 监控 ratio 是否真的受限

至少记录：

```python
approx_old_kl = masked_mean(old_logp - current_logp, mask)
clip_fraction = masked_mean(
    (torch.abs(ratio - 1.0) > eps).float(),
    mask,
)
```

- `clip_fraction` 很快接近 1：学习率或复用 epoch 太大；
- old-current KL 单调暴涨：rollout 已严重过期；
- ratio 始终精确为 1：current 没更新，或 old_logp 被错误重算。

# 第七章：按风险排序实现，而不是一次跑完整训练

## 7.1 第一层：纯函数

按 CS285 Homework 4 的实现重点，先完成：

1. `token_logprobs`
2. `completion_mask`
3. `masked_mean`
4. `sampled_kl`
5. `group_advantages`
6. `shuffled_minibatches`

每个函数都用极小手工张量测试，不加载大模型。

## 7.2 第二层：无训练 rollout

只生成 1 个 prompt、\(G=4\) 条回答，打印：

```text
decoded completion
completion length
parse result
reward breakdown
old sequence log-prob
reference KL
group advantage
```

人工核对一遍。任何一项不合理都不要启动 backward。

## 7.3 第三层：单次 backward

断言：

```python
assert torch.isfinite(loss)
assert policy_grad_norm > 0
assert reference_grad_norm == 0
assert old_logp.requires_grad is False
assert advantages.requires_grad is False
```

然后检查一次很小更新后的正负样本方向。

## 7.4 第四层：十步 overfit / smoke test

固定少量 format-copy prompt，跑十到几十步。预期：

- exact-match reward 明显上升；
- parse rate 与 format rate 可单独解释；
- sampled reference KL 有限；
- completion 长度没有无故爆炸；
- 输出样例肉眼看起来与曲线一致。

这一步通过后，才上 math-hard。

## 7.5 官方代码的推荐阅读顺序

先从数据流入口读，而不是先扎进算法文件：

```text
train.py
  → logprobs / mask
  → rollout buffer
  → reinforce.py
  → grpo.py
  → tasks / verifier
  → sampler
```

每读一个模块，都回答：

1. 输入 shape 是什么？
2. 哪些 tensor 来自 rollout，必须冻结？
3. 哪些 token 被 mask？
4. 哪个维度在做平均？

# 第八章：用不变量定位故障

## 8.1 七个训练不变量

1. padding 位置 loss 恒为 0；
2. prompt token 的 policy loss 恒为 0；
3. reference 参数永远无梯度；
4. old log-prob 在一个 rollout 内不变；
5. 非零方差组的 advantage 均值近似 0；
6. sampled KL 每个有效 token 非负；
7. 每个 rollout 只复用配置的 epoch 数。

## 8.2 症状表

| 症状 | 首查 | 常见根因 |
|---|---|---|
| loss 有数值但 reward 不动 | verifier 分项、样例 | reward 写错或全零 |
| answer 正确却 reward=0 | parser | 格式/归一化不一致 |
| KL 为负 | KL 公式 | 直接用了有符号 log-ratio |
| group advantage 巨大/NaN | std、分组 | 零方差处理或混组 |
| prompt 复读越来越严重 | token mask | prompt 被纳入 policy loss |
| completion 越来越长 | sum/mean、EOS mask | 长度偏差 |
| clip fraction 一步到 100% | old_logp、学习率 | old 快照错误或更新过大 |
| ratio 永远为 1 | old/current 计算时机 | 每 epoch 重算 old |
| reward 上升但样例变怪 | reward breakdown | 学会钻 verifier 漏洞 |

## 8.3 最小日志契约

每个 train step 至少记录：

```text
reward/total_mean
reward/exact_match
reward/format
reward/parse_rate
rollout/completion_length_mean
rollout/nonzero_reward_group_rate
policy/pg_loss
policy/entropy
policy/clip_fraction
policy/old_current_kl
policy/reference_sampled_kl
optimizer/grad_norm
```

再固定保存若干 `(prompt, completion, parsed, reward_parts)`。曲线与样例缺一不可。

## 8.4 本周完成标准

- [ ] 能在纸上画出 logits 与 target 的 shift；
- [ ] completion mask 同时排除 prompt 与 padding；
- [ ] 能解释 \(B=P\times G\) 和分组顺序；
- [ ] old/ref log-prob 均缓存且不参与梯度；
- [ ] verifier 有 parser、判定、reward 三层测试；
- [ ] group advantage 在完整组上计算；
- [ ] sampled KL 非负且只向 current 传梯度；
- [ ] GR-REINFORCE 每批只更新一次；
- [ ] GRPO ratio 在多个 epoch 中使用同一 old_logp；
- [ ] format-copy smoke test 通过后才运行 math-hard。

# 附录：官方实现资料

| 用途 | 官方资料 |
|---|---|
| 作业定义、公式、实现顺序 | [CS285 Homework 4 · LLM RL](https://rail.eecs.berkeley.edu/deeprlcourse/static/homeworks/hw4.pdf) |
| starter code | [Berkeley Deep RL Homework Spring 2026](https://github.com/berkeleydeeprlcourse/homework_spring2026) |
| LLM policy gradient 背景 | [CS285 L14 · LLM RL](https://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-14.pdf) |
| 偏好优化录像 | [CS224R Spring 2025 · Lecture 9](https://www.youtube.com/playlist?list=PLoROMvodv4rPwxE0ONYRa_itZFdaKCylL&index=9) |
| 推理 RL 录像 | [CS224R Spring 2025 · Lecture 10](https://www.youtube.com/playlist?list=PLoROMvodv4rPwxE0ONYRa_itZFdaKCylL&index=10) |

下一周不再改 loss，而是学习如何证明一次训练真的有效：怎样设计基线、消融、统计区间、失败样例审计和 Agent RL 迁移。
