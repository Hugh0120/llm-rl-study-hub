# 第 7 章：把公式变成不会悄悄错位的训练代码

## 7.1 本章符号说明

| 符号 | English | 本章含义 |
|---|---|---|
| \(B\) | prompt batch size | 一批不同 prompt 的数量 |
| \(G\) | group size | 每个 prompt 的 rollout 数 |
| \(N=B\times G\) | rollout count | 展平后的回答总数 |
| \(L\) | padded sequence length | prompt 与 completion 拼接后的最大长度 |
| \(T_i\) | response length | 第 \(i\) 条回答的真实 completion token 数 |
| \(m_{i,t}\) | completion mask | 第 \(i\) 条回答第 \(t\) 个位置是否参与训练 |
| `old_logp` | behavior log-prob | 生成 rollout 时采样 token 的对数概率 |
| `ref_logp` | reference log-prob | 固定 reference 对相同 token 的对数概率 |
| `current_logp` | current log-prob | 当前 actor 在更新时重新计算的对数概率 |

## 7.2 本章目标

读完本章，你应该能够：

1. 为 rollout batch 写出清晰的张量 shape 与语义契约；
2. 正确对齐 logits、labels 与 sampled-token log-prob；
3. 只在 completion 的有效 token 上计算 policy loss；
4. 区分生成时冻结的数据与每次更新重新计算的数据；
5. 将 parser、verifier 和 reward shaping 分成可独立测试的层；
6. 写出 GR-REINFORCE 或 GRPO 的端到端训练骨架；
7. 用不变量和最小 smoke test 定位常见静默错误。

## 7.3 本章主线

前六章已经回答：

- 模型为什么可以看作策略；
- 最终 reward 怎样变成 log-prob 梯度；
- value、GAE 怎样降低方差；
- PPO 怎样受控复用 rollout；
- 偏好和 verifier 怎样提供反馈；
- 同题多采样怎样得到 group-relative advantage。

现在不再引入新的优化算法，而是固定一份**数据契约**。大模型 RL 中最危险的 bug 往往不会报错：

- loss 正常下降，但 log-prob 与 token 错一位；
- prompt token 也被 reward 更新；
- padding 被当成回答；
- old policy 在每个 epoch 被重算，ratio 永远接近 \(1\)；
- 解析失败和答案错误混成同一指标；
- 长回答仅因 token 更多而获得更大梯度。

本章从“一条 rollout 到底要保存什么”开始，逐层落实。

## 7.4 本章新增概念

| 名词 | 中文直觉 | 本章中解决的问题 |
|---|---|---|
| tensor contract | 张量的 shape、对齐与语义约定 | 防止代码能运行但含义错位 |
| rollout record | 一条生成经验的冻结记录 | 明确哪些量来自采样时刻 |
| completion mask | 只选择真实回答 token | 排除 prompt 与 padding |
| sampled-token log-prob | 只取实际生成 token 的 log-prob | 从词表分布得到 policy loss 所需标量 |
| lifecycle invariant | 跨 rollout/update 阶段必须保持的事实 | 防止 old、current、reference 混用 |
| smoke test | 最小可解释测试 | 在大规模训练前验证方向和边界 |

## 7.5 先固定一条 rollout record

对每条回答，建议至少保存：

```text
prompt_id
prompt_token_ids
response_token_ids
input_ids              # prompt + response
attention_mask
completion_mask
terminated
truncated
old_logp               # 每个有效 response token
ref_logp               # 每个有效 response token
raw_model_text
parsed_answer
parse_ok
task_reward
format_reward
total_reward
group_id
advantage
```

若使用 critic，还要保存或计算：

```text
old_value
advantage_per_token
value_target
```

一条重要原则：

> rollout record 是“当时发生了什么”的历史记录，不应随着 optimizer step 被改写。

尤其是 `old_logp`。它必须记录生成当前 token 的行为策略概率；若每次更新都用 current actor 重算，ratio 会失去数据来源的含义。

## 7.6 推荐的 batch shape

采样阶段保留 prompt 与 group 两个维度便于组内统计：

| 张量 | shape | 含义 |
|---|---|---|
| `input_ids` | \([B,G,L]\) | prompt 与回答拼接并 padding |
| `attention_mask` | \([B,G,L]\) | Transformer 可见的非 padding 位置 |
| `completion_mask` | \([B,G,L-1]\) | 每个 next-token label 是否属于真实回答 |
| `old_logp` | \([B,G,L-1]\) | old 对每个 label token 的 log-prob |
| `ref_logp` | \([B,G,L-1]\) | reference 对相同 label 的 log-prob |
| `reward` | \([B,G]\) | 完整回答分数 |
| `advantage` | \([B,G]\) 或 \([B,G,L-1]\) | 序列级组优势或 token 级 GAE |

组内标准化完成后，可将前两维展平为 \(N=B\times G\)，进入 mini-batch 更新。无论怎样 reshape，都必须保证 `group_id` 与回答顺序一致。

## 7.7 logits、labels 与 token 为什么要错开一位

给定：

```text
input_ids = [BOS, 题, 目, :, 2, 3, ×, ..., EOS]
```

语言模型在位置 \(j\) 的 logits 预测位置 \(j+1\) 的 token。因此：

```python
logits = model(input_ids).logits       # [N, L, V]

next_logits = logits[:, :-1, :]        # [N, L-1, V]
next_token_ids = input_ids[:, 1:]      # [N, L-1]
```

实际采样 token 的 log-prob：

```python
all_logp = log_softmax(next_logits, dim=-1)

sampled_logp = gather(
    all_logp,
    dim=-1,
    index=next_token_ids.unsqueeze(-1),
).squeeze(-1)                          # [N, L-1]
```

最小人工检查：

```text
next_logits 的第 j 个位置
必须对应 next_token_ids 的第 j 个 token。
```

如果直接用 `logits` 与 `input_ids` 同位置 gather，模型实际上会用“读完 token 后的分布”给这个 token 打分，整体错一位。shape 完全合法，loss 也可能下降，所以必须用单元测试抓住。

## 7.8 completion mask 怎样构造

假设一条序列：

```text
[prompt token × P] [response token × T] [padding × K]
```

policy loss 只应覆盖 `response token × T`。在 next-token 对齐后的 \([L-1]\) 轴上：

```python
completion_mask[j] = (
    label_at_j_belongs_to_response
    and not_padding
)
```

需要排除：

- prompt 内 token：它们来自数据集，不是本次 policy 采取的动作；
- padding：只是 batch 对齐；
- 某些实现中的固定模板 token：若并非由 actor 采样，也不应当作动作；
- 终止后的任何位置。

推荐用每条样本的 `prompt_length` 与 `response_length` 明确构造，而不是通过 token 值猜边界。pad token 可能与 eos token 共用 id，但语义不同。

## 7.9 masked mean 要先确定归一化对象

policy loss 常写成：

\[
\operatorname{masked\_mean}(z,m)
=
\frac{\sum z\cdot m}{\sum m}.
\]

但有两种不同的平均策略：

### 7.9.1 token-level mean

所有有效 token 一起平均。长回答在 batch 中贡献更多 token。

### 7.9.2 sequence-level mean

先对每条回答内部 token 取平均，再对回答取平均：

\[
\frac1N
\sum_{i=1}^{N}
\frac{
\sum_t m_{i,t}z_{i,t}
}{
\sum_t m_{i,t}
}.
\]

它让每条回答权重相同，通常更适合序列级 reward。两者没有普遍唯一答案，但必须写进实验配置；否则改变最大生成长度会悄悄改变优化目标。

## 7.10 old、current、reference 的数据生命周期

| 量 | rollout 时 | update epoch 1 | update epoch 2 | 下一批 rollout |
|---|---|---|---|---|
| `old_logp` | 由行为策略计算并保存 | 冻结 | 仍冻结 | 重新生成 |
| `ref_logp` | 固定 reference 计算 | 冻结 | 仍冻结 | 对新 token 重算 |
| `current_logp` | 可与 old 相同 | 重新前向 | 再次重新前向 | 变成新 old 的来源 |
| reward | verifier/RM 计算 | 冻结 | 冻结 | 对新回答重算 |
| advantage | rollout 后计算 | 冻结 | 冻结 | 对新组重算 |

`advantage` 也不应在每个 PPO epoch 随 current model 重算。否则目标分布、value 和数据来源同时移动，ratio 的局部校正解释不再成立。

如果使用 Actor–Critic，可在下一批 rollout 前用最新 critic 重新估值；对当前批的 value target 通常保持冻结。

## 7.11 把解析、核验和 shaping 分成三层

不要写一个返回单个 float 的巨型 `reward_fn(text)`。建议拆成：

### 7.11.1 parser

输入原始文本，输出结构化结果：

```python
ParseResult(
    ok=True,
    answer="391",
    format="boxed",
    error=None,
)
```

### 7.11.2 verifier

输入 prompt 元数据与结构化答案，只判断任务结果：

```python
VerifierResult(
    correct=True,
    expected="391",
    observed="391",
)
```

### 7.11.3 reward composer

显式组合：

```python
task_reward = float(verifier.correct)
format_reward = 0.1 * float(parse.format == "boxed")
length_penalty = ...

total_reward = (
    task_reward
    + format_reward
    - length_penalty
)
```

训练日志必须同时记录每个分量。否则总 reward 上升时，你无法判断模型是真的更正确，还是只学会了格式或钻 parser 漏洞。

## 7.12 group advantage 的稳健实现

```python
def group_advantage(reward, group_mask, eps=1e-6):
    # reward, group_mask: [B, G]
    count = group_mask.sum(dim=1, keepdim=True)
    mean = (reward * group_mask).sum(dim=1, keepdim=True) / count

    centered = (reward - mean) * group_mask
    var = (centered.square().sum(dim=1, keepdim=True) / count)
    std = sqrt(var)

    advantage = centered / (std + eps)

    # 零方差组没有组内排序信息
    non_degenerate = (std > eps).float()
    return advantage * non_degenerate
```

还要记录：

- `zero_variance_group_fraction`；
- 每组有效样本数；
- 每组 parse success 数；
- group reward mean 的分布。

如果数据并行把同一 prompt 的回答拆到不同 device，组统计必须先跨设备聚合，否则每张卡得到的 baseline 不同。

## 7.13 GR-REINFORCE 的最小训练骨架

```python
rollout = actor.generate_group(prompts, group_size=G)

with torch.no_grad():
    reward_parts = score_pipeline(rollout)
    advantage = group_advantage(
        reward_parts.total,
        rollout.group_mask,
    )
    ref_logp = reference.sampled_logp(rollout)

current_logp = actor.sampled_logp(rollout)
seq_logp = masked_mean_per_sequence(
    current_logp,
    rollout.completion_mask,
)

policy_loss = -(advantage * seq_logp).mean()

delta = ref_logp - current_logp
sampled_kl = exp(delta) - delta - 1
kl_loss = masked_mean(sampled_kl, rollout.completion_mask)

loss = policy_loss + beta * kl_loss
loss.backward()
clip_grad_norm_(actor.parameters(), max_grad_norm)
optimizer.step()
```

这份骨架只做一次更新，所以 `current_logp` 仍对应采样策略，不需要 importance ratio。

## 7.14 GRPO 多轮复用骨架

```python
with torch.no_grad():
    rollout.old_logp = old_policy.sampled_logp(rollout)
    rollout.ref_logp = reference.sampled_logp(rollout)
    rollout.advantage = group_advantage(...).detach()

for epoch in range(update_epochs):
    for mb in minibatches(rollout):
        current_logp = actor.sampled_logp(mb)
        ratio = exp(current_logp - mb.old_logp)

        advantage = mb.advantage[..., None]
        unclipped = ratio * advantage
        clipped = clip(ratio, 1-eps, 1+eps) * advantage

        pg_objective = minimum(unclipped, clipped)

        delta = mb.ref_logp - current_logp
        sampled_kl = exp(delta) - delta - 1

        objective = pg_objective - beta * sampled_kl
        loss = -masked_mean(objective, mb.completion_mask)

        optimize(loss)

        metrics = {
            "approx_kl_old": masked_mean(
                mb.old_logp - current_logp,
                mb.completion_mask,
            ),
            "clip_fraction": masked_mean(
                (abs(ratio - 1) > eps).float(),
                mb.completion_mask,
            ),
            "ratio_p99": masked_quantile(ratio, 0.99),
        }

        if metrics["approx_kl_old"] > target_kl:
            stop_reusing_this_rollout()
```

## 7.15 八条必须自动检查的不变量

1. 每条样本 `completion_mask.sum() == response_length`；
2. mask 为 \(0\) 的位置改变 log-prob，不会改变 loss；
3. rollout 刚生成时，current 与 old 参数相同，ratio 的有效位置应接近 \(1\)；
4. reference 与 actor 初始 checkpoint 相同时，训练开始的 sampled KL 应接近 \(0\)；
5. 每组 advantage 在有效回答上的均值接近 \(0\)；
6. 零方差组的 policy contribution 为 \(0\)；
7. `old_logp.requires_grad == False`，`advantage.requires_grad == False`；
8. 终止后的 token、padding 和 prompt token 的 actor gradient 为 \(0\)。

这些断言比“跑起来了”更有价值。

## 7.16 从小到大的 smoke test

### Test 1：单 token 人造策略

词表只有 `A/B`，reward(`A`)=1，reward(`B`)=0。训练几十步后 \(P(A)\) 必须上升。

### Test 2：正负 advantage 梯度方向

固定一条样本：

- advantage \(>0\) 做一步后 sampled token log-prob 应上升；
- advantage \(<0\) 做一步后应下降。

### Test 3：mask

改变 prompt token 和 padding 位置的 advantage/log-prob，loss 应完全不变。

### Test 4：PPO clipping

手工构造 ratio \(1.5\)、advantage \(+1\)、\(\epsilon=0.2\)，objective 应等于 \(1.2\)；构造 ratio \(0.5\)、advantage \(-1\)，应选择 \(-0.8\)。

### Test 5：真实小任务过拟合

只训练 8–32 道可验证算术题。模型应能明显提高训练题 reward；若连小数据都无法过拟合，先查数据对齐和 reward，不要直接扩集群。

### Test 6：held-out 与格式拆分

训练 reward 上升后，分别检查：

- parse rate；
- parsed 样本条件正确率；
- 端到端正确率；
- 平均长度；
- reference KL。

## 7.17 常见症状到根因

| 症状 | 优先检查 |
|---|---|
| reward 不变，loss 在下降 | verifier、group 全零方差、token 对齐 |
| ratio 永远为 1 | old_logp 是否被每步重算；optimizer 是否真的更新 |
| ratio 第一轮就很大 | old/current checkpoint 是否一致；logp 是否同一 token 对齐 |
| KL 为负且波动巨大 | 是否把单样本 log-ratio误当 KL；符号方向 |
| 输出越来越长 | sequence/token 归一化、长度 reward、截断处理 |
| 格式正确率升高但任务正确率不升 | shaping 奖励压过 task reward |
| 多卡结果与单卡不同 | 同 prompt group 是否跨卡聚合 |
| 训练偶尔 NaN | 空 mask、全零组、极端 log-ratio、未裁剪梯度 |

## 7.18 本章自测

1. 为什么 logits 要取 `[:, :-1]`，labels 要取 `input_ids[:, 1:]`？
2. prompt token 为什么不应进入 policy loss？
3. `old_logp` 和 `current_logp` 分别何时计算？为什么前者不能重算？
4. token-level mean 与 sequence-level mean 对长回答的权重有何不同？
5. 为什么 parser、verifier、reward composer 应拆开测试？
6. 若训练题无法过拟合，最先查哪三类问题？

## 7.19 本章之后还缺什么

实现通过 smoke test，只能证明训练管线“可能按预期工作”。reward curve 上升仍不能证明模型真的更好：

- 它可能只优化格式分；
- 可能记住训练题；
- 可能在 `pass@k` 上靠更多随机尝试；
- 可能利用 verifier 漏洞；
- 可能牺牲原有通用能力。

第 8 章会把“模型变好了”拆成一套可证伪的评估协议。

## 7.20 对应资料

- [Hugging Face TRL Documentation](https://huggingface.co/docs/trl/)
- [PyTorch `gather`](https://pytorch.org/docs/stable/generated/torch.gather.html)
- [OpenAI Spinning Up: PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html)
- [DeepSeekMath](https://arxiv.org/abs/2402.03300)
