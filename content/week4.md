# 第四周：怎样证明训练真的有效，再迁移到会调用工具的模型

> 前三周已经把 RL 地基、反馈来源和训练实现依次讲完。本周不再增加新的训练算法，只回答更难的问题：你凭什么相信训练有效，改善来自哪一项机制，以及相同的决策结构怎样迁移到会调用工具的模型。

## 本周最终产出

你会得到一套可直接复用的实验协议：

1. 明确研究问题与可证伪假设；
2. 建立从未训练模型到不同训练版本的可比较基线；
3. 同时观察任务指标、机制指标、样例和成本；
4. 一次只改变一个训练因素，解释改善来自哪里；
5. 系统审计模型是否钻评分规则的漏洞；
6. 把单轮文本生成扩展成多步工具交互；
7. 写出一页能支持 go/no-go 决策的报告。

贯穿全文的原则：

\[
\text{训练 reward}
\neq
\text{真实任务能力}
\neq
\text{部署价值}.
\]

# 第一章：先定义什么证据能支持“训练有效”

## 1.1 一条 reward 曲线为什么不够

训练 reward 上升至少有四种解释：

1. 模型真正更会解题；
2. 模型更会满足格式分；
3. 模型找到 parser 或 verifier 漏洞；
4. 评估样本与训练样本重合。

因此一次可信实验至少需要四层证据：

| 层 | 回答的问题 | 例子 |
|---|---|---|
| 任务指标 | 能力真的提高了吗？ | held-out exact accuracy |
| 机制指标 | 训练按预期工作吗？ | KL、entropy、clip fraction |
| 行为证据 | 模型具体改变了什么？ | 固定 prompt 前后样例 |
| 成本指标 | 改善值不值得？ | GPU-hour、每题 token、wall time |

只有四层方向一致，才能把“曲线变好”解释为“系统变好”。

## 1.2 把研究问题写成可证伪假设

坏问题：

> GRPO 好不好？

可实验的问题：

> 在相同模型、prompt 集、采样温度和生成预算下，GRPO 相比一次更新的 GR-REINFORCE，是否以更少 rollout 达到更高的 held-out exact accuracy，同时把 reference KL 控制在预设范围内？

对应假设：

- H1：rollout 复用提高样本效率；
- H2：若 epoch 过多，old-current KL 和 clip fraction 上升，最终质量可能下降；
- H3：计算效率未必提高，因为每批 rollout 需要更多 backward。

每个假设都给出了可能被数据否定的条件。

## 1.3 一次实验只改变一个主要因素

比较两个方法时固定：

```text
base model
tokenizer
prompt split
generation temperature
max completion length
group size
reward function
total rollout count
evaluation protocol
randomness control
```

若同时改模型、reward、数据量和算法，结果只能说明“整套配置不同”，不能解释是哪项机制有效。

# 第二章：在 RL 之前建立可信基线

## 2.1 三条最低基线

对 math-hard 类任务，至少记录：

1. **Base/SFT checkpoint**：RL 前的能力；
2. **GR-REINFORCE**：不复用 rollout 的在线基线；
3. **GRPO**：同样 rollout 预算下的复用版本。

如果有固定偏好数据，再加 DPO；但不要为了表格整齐强行比较数据条件完全不同的方法。

## 2.2 数据划分必须先于调参

推荐：

```text
train prompts      → rollout 与梯度更新
validation prompts → 选超参数、early stop
test prompts       → 最终只评一次或极少次
```

划分要确定性保存：

```python
split_id = stable_hash(normalized_prompt) % 100
```

同时检查：

- 重复题与同模板变体；
- 标准答案泄漏到 prompt；
- train/test 只改了数字但结构完全相同；
- verifier 是否读取了不该使用的 metadata。

## 2.3 parser 指标与能力指标分开

对每条样本保存：

```text
parse_ok
format_ok
answer_ok
reward_total
completion_length
```

至少报告：

\[
\text{parse rate}
=
\frac{\#\text{可解析输出}}{N},
\]

\[
\text{conditional accuracy}
=
\frac{\#\text{正确且可解析}}{\#\text{可解析}},
\]

\[
\text{end-to-end accuracy}
=
\frac{\#\text{最终判对}}{N}.
\]

如果 parse rate 从 40% 升到 95%，而 conditional accuracy 不变，模型主要学会了输出格式，不应写成“推理能力大幅提升”。

## 2.4 一次回答成功与多次尝试成功是两个问题

同一个模型可以用两种方式部署：

- 只生成一次，直接把这条回答交给用户；
- 允许生成 \(k\) 次，再从中挑出至少一条成功答案。

这两个场景需要分开报告。课程把“一次生成就成功的概率”记作
**pass@1**，把“允许生成 \(k\) 次且至少一次成功的概率”记作
**pass@\(k\)**。

若对每题采样 \(n\) 次，其中 \(c\) 次正确，就可以估计 pass@\(k\)。
把估计值写成
\(\widehat{\text{pass@}k}\)：帽子表示“由有限样本算出的估计”，不是新的指标。
常用无偏估计为：

\[
\widehat{\text{pass@}k}
=
1-\frac{\binom{n-c}{k}}{\binom{n}{k}},
\qquad n-c\ge k.
\]

\(\binom{n}{k}\) 表示“从 \(n\) 条回答中选出 \(k\) 条共有多少种选法”；
\(\binom{n-c}{k}\) 表示选出的 \(k\) 条恰好全来自错误回答的选法。两者相除得到
“\(k\) 次全失败”的概率，再用 1 减去它，就得到“至少一次成功”的概率。

pass@k 上升但 pass@1 不升，可能只是分布尾部多了一些好样本，并不代表默认单次回答更可靠。

## 2.5 不确定性不是可选装饰

若 \(N\) 道独立题的正确率估计为 \(\hat p\)，可以估计它因有限题量产生的
波动。标准误写作 \(\operatorname{SE}\)，即 standard error：

\[
\operatorname{SE}(\hat p)
\approx
\sqrt{\frac{\hat p(1-\hat p)}{N}}.
\]

例如 \(N=100,\hat p=0.60\)：

\[
\operatorname{SE}\approx0.049.
\]

2 个百分点的变化很可能只是噪声。正式报告应使用 Wilson 区间或按 prompt bootstrap，尤其是题目难度不均时。

### 配对比较优先

两个 checkpoint 应在同一组 prompt、尽量相同的采样配置上评估。记录每题从错变对、从对变错的配对差异，比比较两个独立均值更有统计效率。

## 2.6 为什么最终结果必须重复不同的随机运行

训练和采样都包含随机步骤。把这些随机数发生器的初始值固定后，一次运行才更容易
复现；这个初始值叫 **random seed（随机种子）**，通常简写为 `seed`。

它会影响：

- prompt 抽样；
- generation 随机性；
- minibatch 顺序；
- dropout 与初始化。

至少：

1. 开发阶段用固定 seed 复现；
2. 最终候选用多个 seed；
3. 同时报告均值、离散程度和最差 seed；
4. 不因某个 seed 漂亮就只展示它。

# 第三章：用受控消融解释训练机制

## 3.1 先写主实验表

示例：

| run | 方法 | rollout 数 | update epochs | group size | \(\beta\) | clip \(\epsilon\) |
|---|---|---:|---:|---:|---:|---:|
| A | SFT checkpoint | 0 | 0 | — | — | — |
| B | GR-REINFORCE | 20k | 1 | 8 | 0.02 | — |
| C | GRPO | 20k | 4 | 8 | 0.02 | 0.2 |

主比较固定 rollout 数，回答样本效率；另建一张固定总 FLOPs/GPU-hour 的表，回答计算效率。不要把两者混成一句“更高效”。

## 3.2 消融一：group size

比较：

\[
G\in\{2,4,8,16\}.
\]

机制：

- \(G\) 太小：组均值和标准差噪声大；
- \(G\) 增大：同题相对排序更稳定；
- \(G\) 过大：单次 rollout 成本高，prompt 多样性下降。

必须一起看：

```text
held-out accuracy
nonzero-reward group rate
zero-variance group rate
advantage std
rollout tokens
wall time
```

一个重要诊断是“有效组比例”：

对 prompt \(x\)，用 \(\sigma_x\) 表示这一组回答 reward 的标准差。
\(\sigma_x>0\) 意味着组内既有较好回答也有较差回答，因而能产生相对训练信号。

\[
\text{informative-group rate}
=
\frac{\#\{x:\sigma_x>0\}}{\#\text{groups}}.
\]

若几乎所有组都全错，增大 update epoch 没用，应该先改善探索、课程难度或 SFT 起点。

## 3.3 消融二：reference KL 系数 \(\beta\)

比较：

\[
\beta\in\{0,\beta_1,\beta_2,\beta_3\}.
\]

预期：

- \(\beta=0\)：可能训练 reward 上升最快，也最容易漂移；
- \(\beta\) 适中：在任务增益与行为锚定间折中；
- \(\beta\) 太大：policy 几乎不动。

同时报告：

```text
task accuracy
sampled reference KL
entropy
format rate
general-language regression set
```

“KL 越小越好”也是错误的。KL 几乎为零可能只是没有学习。

## 3.4 消融三：rollout 复用强度

同时改变 PPO epoch 和 clip \(\epsilon\) 会难以解释，先分别消融：
用 \(K\) 表示同一批 rollout 被优化的 epoch 数；\(\epsilon\) 是第一周定义的
clipping 半宽。

\[
K\in\{1,2,4,8\},
\]

固定 \(\epsilon\)；再固定 \(K\) 比较：

\[
\epsilon\in\{0.1,0.2,0.3\}.
\]

观察：

- old-current KL；
- clip fraction；
- 每批最后一个 epoch 的增益；
- held-out accuracy；
- 相同 rollout 数与相同 GPU-hour 下的结果。

若后几个 epoch 大量样本被 clip，且验证集不再改善，继续复用只是在消耗计算。

## 3.5 消融四：最终得分是否混入了额外的过程分

为了让稀疏任务更早出现非零信号，训练者常把答案正确、格式正确和部分过程正确组合成
一个总分。这个做法叫 **reward shaping（奖励塑形）**。它可能帮助探索，也可能让模型
只刷容易得到的附加分，因此必须单独做消融。

下面用：

- \(R_{\text{answer}}\)：最终答案正确得到的分数；
- \(R_{\text{format}}\)：输出格式满足规则得到的分数；
- \(R_{\text{partial}}\)：中间过程满足检查点得到的部分分；
- \(\alpha,\eta\)：格式分和部分分在总 reward 中的权重。

设：

\[
R
=
R_{\text{answer}}
+\alpha R_{\text{format}}
+\eta R_{\text{partial}}.
\]

至少比较：

1. 只有答案正确；
2. 答案 + 格式；
3. 答案 + 格式 + 部分分。

需要回答：

- shaping 是否提高了早期非零信号？
- 最终 answer accuracy 是否真的提高？
- 模型是否只刷容易的格式分？
- partial reward 是否与真实成功一致？

## 3.6 长度是隐藏超参数

策略 loss 用 token sum、token mean，或 reward 中加入长度项，会产生不同偏好。每次实验都记录：

\[
\text{mean/median/p95 completion length}.
\]

并画 reward 与长度的散点图。若 reward 与长度强相关，先检查 verifier 和 loss 聚合，再讨论“长推理更强”。

# 第四章：训练分数变高，模型却可能学错目标

## 4.1 先看“分数变高但能力没变”是怎样发生的

假设评分程序只检查最后一个 `\boxed{}` 中的内容。模型发现，输出多个互相冲突的
候选答案，偶尔能让解析器取到正确值。训练分数因此上升，但模型并没有更会计算。

这类现象的共同结构是：

> 策略提高了可见 reward，却没有提高设计者真正关心的目标。

现在才给它命名：**reward hacking（奖励投机）**，也就是评分规格与真实目标之间的
错位被优化过程利用了。

这不是模型“有恶意”，而是优化器忠实利用了目标中的漏洞。

## 4.2 漏洞通常来自哪四层

### 解析器漏洞

模型构造让 parser 误判的文本：

```text
\boxed{wrong} ... \boxed{correct}
```

如果 parser 总取最后一个框，模型可能学会堆多个候选。

### 代理目标漏洞

format reward 太大，模型只输出漂亮标签，不解决任务。

### 学到的 reward model 漏洞

reward model 偏爱长度、礼貌或固定句式，policy 放大这些表面特征。

### 环境与工具漏洞

会调用工具的模型通过重复调用、利用缓存、读取隐藏测试或修改环境状态获得高分，而非真正完成任务。

## 4.3 怎样在训练前主动攻击判题程序

安全工程里把这种主动寻找漏洞的过程叫 **red teaming（红队测试）**。对 verifier，
可以在训练前建立下面的对抗集合：

```text
空答案
多个冲突答案
格式嵌套
Unicode/科学计数法
超长前缀
复制 prompt 中的目标
异常终止
伪造工具返回
重复调用
越权文件或网络请求
```

每修复一次 verifier，都版本化：

```text
verifier_version
parser_version
reward_config_hash
```

否则前后实验 reward 不再可比。

## 4.4 双通道评估

训练通道和审计通道应尽量独立：

| 通道 | 用途 |
|---|---|
| train verifier | 高频、便宜地提供 reward |
| audit evaluator | 更严格规则、人工抽检或独立测试 |

若 train reward 上升而 audit 指标不升，优先按 reward hacking 处理，不要继续增加优化步数。

## 4.5 固定样例面板

每个 checkpoint 对同一批固定 prompt 展示：

```text
prompt
completion
parsed answer
reward breakdown
audit result
length
reference log-ratio
```

建议包括：

- 训练前一直能做对的题；
- 训练前偶尔做对的题；
- 训练前从不做对的题；
- parser 边界案例；
- 开放式回归样例。

# 第五章：从曲线反推问题发生在哪一层

## 5.1 reward 上升，held-out accuracy 不升

依次检查：

1. format/partial 分是否上升；
2. train 与 eval parser 是否一致；
3. train prompt 是否被记忆；
4. completion 长度是否改变；
5. 固定样例是否出现漏洞模式。

不要先调学习率，因为这更像目标或评估问题。

## 5.2 reward 与 accuracy 都不升

看有效探索：

- 非零 reward 样本率；
- informative-group rate；
- entropy；
- 每题成功样本数。

若全组为 0：

\[
A_i=0,
\]

策略没有方向。可能的解决顺序：

1. 更容易的 curriculum；
2. 更好的 SFT 起点；
3. 合理提高采样温度或组大小；
4. 可信的部分 reward；
5. 更长生成预算。

## 5.3 训练开始后迅速崩掉

看：

```text
old-current KL
reference KL
clip fraction
grad norm
entropy
NaN/Inf
```

对应动作：

- old-current KL 高：减小学习率、epoch 或增大 batch；
- reference KL 高：提高 \(\beta\)，并审计 reward；
- grad norm 爆炸：检查 advantage scale、mask、ratio；
- entropy 突降：策略过早变确定，探索消失。

## 5.4 何时停止

设置预先定义的 guardrail：

```text
validation accuracy 连续 M 次不升
sampled reference KL 超阈值
clip fraction 超阈值
audit success 下降
成本超预算
安全回归失败
```

停止不是承认失败，而是避免在已失效的数据和目标上继续优化。

# 第六章：从单轮回答扩展到会调用工具的模型

## 6.1 先把单轮生成中的四个对象逐项替换

下面把能够读取环境、选择工具、观察执行结果并继续决策的模型系统叫
**Agent（智能体）**。用强化学习训练这种多步系统，通常简称 **Agent RL**。
名称没有改变数学结构；我们仍然只需要找出状态、动作、状态变化和 reward。

单轮文本生成：

| RL 元素 | 含义 |
|---|---|
| state | prompt + 已生成 token |
| action | 下一个 token |
| transition | 拼接 token |
| reward | 回答得分 |

工具 Agent：

| RL 元素 | 含义 |
|---|---|
| state \(s_t\) | 用户目标、对话、工具返回、环境状态、记忆 |
| action \(a_t\) | 工具调用、参数、消息或停止 |
| transition | 工具/环境执行后的新状态 |
| reward | 任务成功、成本、安全与规则组合 |

把一次从开始到结束的 Agent 交互记作轨迹 \(\tau\)。希腊字母
\(\tau\) 只是 trajectory 的惯用标签；\(o_t\) 表示第 \(t\) 次工具或环境
返回的 observation。一条轨迹写成：

\[
\tau=(s_0,a_0,o_1,s_1,a_1,o_2,\ldots,s_T),
\]

其中 \(o_t\) 是环境 observation。Agent RL 不是突然换了数学，只是动作和转移不再是简单拼 token。

## 6.2 用一个客服工具模型走完整条链

目标：为符合政策的订单退款。

可能轨迹：

```text
读取订单
→ 检查退款资格
→ 请求用户确认
→ 调用 refund(order_id, amount)
→ 返回凭证
```

最终 reward 需要同时覆盖成功、成本和安全。先定义四个部分：

- \(R_{\text{task}}\)：任务是否真正成功；
- \(C_{\text{tool}}\)：无效或重复工具调用的成本；
- \(C_{\text{token}}\)：延迟与 token 成本；
- \(C_{\text{safety}}\)：越权、缺少确认、泄露信息等安全成本；
- \(\lambda_c,\lambda_t,\lambda_s\)：三类成本各自的权重。

于是总 reward 可以写成：

\[
R
=
R_{\text{task}}
-\lambda_c C_{\text{tool}}
-\lambda_t C_{\text{token}}
-\lambda_s C_{\text{safety}}.
\]

只奖励“最终状态已退款”，模型可能跳过确认或退款错误金额。reward 必须覆盖部署中真正不可妥协的约束。

## 6.3 判题程序应验证状态变化，而非模型自我声明

坏 verifier：

```python
return "refund completed" in final_message
```

可信 verifier：

```python
return (
    env.refund_created(order_id)
    and env.refund_amount(order_id) == expected_amount
    and audit.confirmation_obtained
    and not audit.unauthorized_access
)
```

Agent 说“完成了”不是完成证据；环境状态和审计日志才是。

## 6.4 最终成功怎样归因到前面的工具动作

只有终局成功 reward 时，所有工具动作共享一个稀疏结果。这就是第一周已经系统讲过的
**credit assignment（信用分配）**问题，只是动作从 token 变成了工具调用。可选路线：

1. outcome-only：最可信，但方差大；
2. learned value：预测中间状态成功率；
3. process checks：对必要里程碑给分；
4. trajectory comparison：同任务多条轨迹做组内相对评价；
5. hindsight diagnosis：失败后标记造成不可恢复错误的动作。

过程 reward 仍需防止模型刷里程碑而不完成最终任务。最终 success 必须保留为独立指标。

## 6.5 从离线到在线的安全顺序

推荐阶梯：

\[
\text{历史日志分析}
\rightarrow
\text{监督模仿}
\rightarrow
\text{离线回放评估}
\rightarrow
\text{沙箱在线 rollout}
\rightarrow
\text{影子模式}
\rightarrow
\text{小流量、可回滚部署}.
\]

每一层都要有限权工具：

- allowlist；
- 参数 schema；
- 金额/次数限制；
- 幂等键；
- 超时和最大步数；
- 人工确认；
- 完整审计日志；
- 可恢复环境。

在线 RL 的授权范围不能大于评估和回滚能力。

## 6.6 多步工具模型需要哪些实验指标

不要只报告 episode reward：

```text
task success
policy-compliant success
unsafe action rate
tool error rate
duplicate call rate
steps per success
tokens per success
wall time per success
human intervention rate
```

部署价值更接近：

\[
\frac{\text{合规成功任务数}}
{\text{总成本}}
\]

而不是单一 reward 均值。

# 第七章：一页实验报告模板

## 7.1 问题与假设

```text
问题：
主要假设：
可能推翻假设的结果：
```

## 7.2 设置

```text
base checkpoint:
dataset/split hash:
train/val/test size:
verifier/parser version:
method:
group size:
temperature:
max tokens:
beta:
clip epsilon:
update epochs:
rollout budget:
compute budget:
seeds:
```

## 7.3 结果

| 指标 | baseline | candidate | 差值 | 不确定性 |
|---|---:|---:|---:|---:|
| held-out accuracy |  |  |  |  |
| parse rate |  |  |  |  |
| pass@k |  |  |  |  |
| reference KL |  |  |  |  |
| tokens / success |  |  |  |  |
| GPU-hour |  |  |  |  |

## 7.4 机制与失败

```text
informative-group rate:
clip fraction:
entropy:
old-current KL:
reward/length correlation:
代表性改善样例:
代表性退化样例:
reward hacking 审计:
```

## 7.5 结论

```text
假设是否得到支持：
收益来自能力、格式还是探索：
已知限制：
安全/成本 guardrail：
go / no-go：
下一项最小实验：
```

# 第八章：整套课程的最终依赖链

\[
\text{MDP 建模}
\rightarrow
\text{REINFORCE}
\rightarrow
\text{baseline/value}
\rightarrow
\text{TD/GAE}
\rightarrow
\text{PPO}
\]

\[
\rightarrow
\text{偏好或 verifier}
\rightarrow
\text{DPO / RLHF / GRPO}
\rightarrow
\text{正确实现}
\rightarrow
\text{可信实验}
\rightarrow
\text{Agent RL}.
\]

每一步都由前一步暴露的问题自然引出：

| 问题 | 解法 |
|---|---|
| 离散采样不可直接求导 | log-derivative trick |
| 完整回报噪声大 | baseline/value |
| 长期信用分配困难 | TD、n-step、GAE |
| rollout 贵且更新会过期 | ratio、PPO |
| reward 不知道从哪来 | 偏好学习或 verifier |
| 不想训练 critic | 同题 group baseline |
| 代码能跑但可能算错 | tensor 契约与不变量 |
| reward 涨但未必真好 | 独立评估、消融、审计 |
| 单轮回答不能完成业务 | 多步 Agent 环境与工具动作 |

## 最终完成标准

- [ ] 能设计一项只改变一个因素的实验；
- [ ] 能区分 rollout 效率与计算效率；
- [ ] 能给 accuracy 报告不确定性；
- [ ] 能解释 parse rate、conditional accuracy 与 end-to-end accuracy；
- [ ] 能用 informative-group rate 判断探索是否有效；
- [ ] 能按四类漏洞主动攻击并检查 verifier；
- [ ] 能从曲线与样例共同定位故障；
- [ ] 能把工具 Agent 写成 state/action/transition/reward；
- [ ] 能用环境状态而非模型自述验证成功；
- [ ] 能给 Agent 设计成本、安全和回滚 guardrail。

# 附录：官方课程与继续学习

| 主题 | 官方资料 | 对应视频 |
|---|---|---|
| LLM RL 与 Homework 4 实验 | [CS285 Homework 4](https://rail.eecs.berkeley.edu/deeprlcourse/static/homeworks/hw4.pdf) | [CS285 课程页](https://rail.eecs.berkeley.edu/deeprlcourse/) |
| 推理 RL 与 step credit | [CS224R L10 · RL for Reasoning](https://cs224r.stanford.edu/spring_2025/slides/10_cs224r-rl_for_reasoning_lecture.pdf) | [CS224R Spring 2025 · Lecture 10](https://www.youtube.com/playlist?list=PLoROMvodv4rPwxE0ONYRa_itZFdaKCylL&index=10) |
| Agent 推理、规划、记忆与安全 | [Berkeley · Advanced LLM Agents](https://rdi.berkeley.edu/adv-llm-agents/sp25) | [Lecture 2 · Learning to Reason](https://www.youtube.com/live/_MNlLhU33H0) |
| inference-time reasoning | [Berkeley Agent 课程主页](https://rdi.berkeley.edu/adv-llm-agents/sp25) | [Lecture 1 · Inference-time Reasoning](https://www.youtube.com/live/g0Dwtf3BH-0) |

至此四周形成一套独立教材：Slides 和视频用于核对与扩展，不是理解正文的前置条件。
