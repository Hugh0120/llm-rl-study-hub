# 第 9 章：从单轮回答走向会调用工具的 Agent

## 9.1 本章符号说明

| 符号 | English | 本章含义 |
|---|---|---|
| \(o_t\) | observation | 第 \(t\) 步模型能看到的用户信息、工具结果和环境状态摘要 |
| \(h_t\) | history | 到第 \(t\) 步为止的完整交互历史 |
| \(e_t\) | environment state | 工具背后真实但未必完全展示给模型的环境状态 |
| \(a_t\) | agent action | 文本回复、工具调用或终止决定 |
| \(z_t\) | tool result | 环境执行工具后返回的结构化结果 |
| \(c_t\) | action cost | token、延迟、API 费用或风险成本 |
| \(P\) | environment transition | 动作执行后真实环境状态怎样变化 |
| \(O\) | observation interface | 环境状态怎样被转换成模型可见结果 |
| \(R_{\text{task}}\) | terminal task reward | 最终环境状态是否满足用户目标 |
| \(R_{\text{safety}}\) | safety reward / constraint | 是否遵守权限、隐私与不可逆操作规则 |

## 9.2 本章目标

读完本章，你应该能够：

1. 将第 1 章的单轮 MDP 映射到多步工具交互；
2. 区分语言 token、工具调用和环境状态变化；
3. 设计以真实环境结果为准而不是以模型自述为准的 verifier；
4. 将任务成功、工具成本、格式与安全信号分层；
5. 分析长轨迹中的信用分配与部分可观测问题；
6. 设计 offline、sandbox、shadow、online 的分阶段训练与评估；
7. 明确哪些安全边界不应被一个可权衡的 reward 代替。

## 9.3 本章主线

前八章的默认环境是：

```text
给定 prompt
→ 模型连续生成 token
→ 提交完整回答
→ verifier / RM 给分
```

现在考虑一个退款助手：

```text
用户要求退订单
→ 模型查询订单
→ 检查退款条件
→ 必要时请求确认
→ 调用退款工具
→ 检查最终状态
→ 回复用户
```

区别是：中途动作会改变外部世界，工具结果又成为下一步输入。我们只需要逐项替换第 1 章的对象：

| 单轮回答 | 工具 Agent |
|---|---|
| 状态：prompt + 已生成 token | 状态：对话历史 + 工具结果 + 可见环境状态 |
| 动作：下一个 token | 动作：文本片段、工具名及参数、终止 |
| 转移：拼接 token | 转移：工具执行、环境变化、返回新观察 |
| reward：最终答案分 | reward：任务结果、成本、安全与用户反馈 |

当一个模型根据观察反复选择动作并影响环境时，我们把这套系统称为 **Agent**。这个名字只是上述交互结构的简称。

## 9.4 本章新增概念

| 名词 | 中文直觉 | 本章中解决的问题 |
|---|---|---|
| tool action | 对外部系统的结构化调用 | 动作不再只是自然语言 token |
| environment state | 工具背后的真实世界状态 | 成功与否不能只看模型说了什么 |
| partial observability | 模型只能看到状态的一部分 | history 需要承担记忆与推断 |
| action cost | 每次调用的延迟、费用和风险 | 防止靠无限尝试提高成功率 |
| trajectory verifier | 检查整条交互与最终状态 | 评价是否真正完成任务且过程合规 |
| hard constraint | 不允许被 reward 抵消的边界 | 权限、隐私、不可逆操作需要门控 |
| sandbox / shadow | 隔离执行 / 只观察不执行 | 在真实上线前降低探索风险 |

## 9.5 状态：模型真正知道什么

理想 MDP 状态应包含预测未来所需的一切。但工具系统常是部分可观测的：

- 订单数据库有字段未展示给模型；
- API 返回分页结果；
- 用户真实意图不完全明确；
- 其他服务可能同时修改状态；
- 早期工具输出可能因上下文裁剪而消失。

模型实际接收 observation \(o_t\)。常用完整可见历史近似状态：

\[
s_t
\approx
h_t
=
(x,o_0,a_0,z_0,\ldots,o_t).
\]

工程上应显式规定 history 中包含：

- 系统权限与政策；
- 用户请求；
- 已确认的关键参数；
- 工具调用和结构化结果；
- 环境版本或时间戳；
- 尚未解决的错误。

不能把关键状态只藏在自然语言摘要中，否则模型可能遗漏“退款已经执行过”并重复调用。

## 9.6 动作：token 与工具调用的两层结构

一种常见 Agent action：

```json
{
  "tool": "refund_order",
  "arguments": {
    "order_id": "A123",
    "amount": 49.90
  }
}
```

从语言模型角度，它仍由 token 生成；从环境角度，它是一个原子工具动作。训练时必须分清两层：

1. **token-level policy**：怎样生成工具名和参数；
2. **environment-level action**：解析成功并通过权限检查后，实际执行什么。

若 JSON 无法解析，环境并没有执行退款。若模型输出文字“退款成功”，也不等于真实状态变化。

因此一轮转移应写成：

\[
h_t
\xrightarrow{\text{model generates }a_t}
\operatorname{parse/validate}(a_t)
\xrightarrow{\text{tool executes}}
z_t
\rightarrow
h_{t+1}.
\]

parser、权限层、工具执行器与模型本身都可能失败，日志必须分别记录。

## 9.7 转移：环境决定下一状态

在纯文本生成中：

\[
s_{t+1}=s_t\oplus a_t.
\]

工具场景中：

\[
e_{t+1}
\sim
P(\cdot\mid e_t,a_t),
\]

\[
o_{t+1}
\sim
O(\cdot\mid e_{t+1}),
\]

其中 \(e_t\) 是模型未必完全可见的真实环境状态，\(o_{t+1}\) 是工具返回的观察。

这带来新的训练事实：

- 相同调用可能因网络、权限或并发状态得到不同结果；
- 工具失败不是模型一定“推理错了”，但模型需要正确恢复；
- 旧 rollout 依赖当时环境快照，不能随意在变化后的生产环境重放；
- verifier 应记录环境版本，保证结果可追溯。

## 9.8 reward 必须以真实结果为锚

退款任务的最终成功不应由模型回答中的一句“已为您退款”判断，而应由环境检查：

```text
订单 A123 的退款状态 == completed
退款金额 == 49.90
收款账户 == 原支付方式
未产生重复退款
用户要求的其他订单未被修改
```

可以分层构造：

\[
R_{\text{total}}
=
R_{\text{task}}
-
\alpha C_{\text{tool}}
-
\eta C_{\text{token}}
+
R_{\text{quality}},
\]

但安全边界要单独处理。

| 分量 | 示例 | 适合做可权衡 reward 吗 |
|---|---|---:|
| 任务成功 | 正确完成退款 | 是 |
| 工具成本 | API 次数、延迟 | 是 |
| 回答质量 | 解释清晰、简洁 | 可以 |
| 格式 | 工具参数可解析 | 可以 |
| 未授权转账 | 越权修改资金 | 否，应硬阻断 |
| 泄露隐私 | 输出其他用户数据 | 否，应硬阻断 |
| 不可逆操作未确认 | 删除、支付、发布 | 否，应权限门控 |

如果把严重安全违规只设为 \(-10\)，而任务成功为 \(+100\)，优化器可能学会“违规但总分更高”。硬约束应由执行层拒绝，不让 policy 通过 trade-off 绕过。

## 9.9 一条退款轨迹怎样评分

### 成功轨迹

```text
1. 查询订单 A123                         cost -0.01
2. 发现订单可退，金额 49.90
3. 请求用户确认                          quality +0.1
4. 用户确认
5. 调用 refund_order(A123, 49.90)       cost -0.05
6. 查询状态，确认 completed              task +1.0
7. 向用户报告真实结果
```

### 看似成功但实际失败

```text
1. 未查询订单
2. 直接回复“退款已完成”
3. 环境状态仍为 paid                     task 0
```

### 完成任务但不安全

```text
1. 从历史猜测 order_id
2. 未经确认调用不可逆退款
3. 环境显示退款完成                      hard constraint violation
```

第三条不能因为 `task +1` 就进入正常的正 advantage 组。它应在执行前被阻止，训练数据中记录拒绝原因，并用安全策略/监督信号处理。

## 9.10 长轨迹的信用分配

若只在最后给 task reward，一条十步成功轨迹中的每个 token 都可能收到相同序列 advantage。第 3 章的问题重新出现，而且更严重：

- 哪一次查询真正有用？
- 重试是必要恢复还是浪费？
- 参数错误发生在哪一步？
- 最终成功是模型决策还是环境偶然？

可以按可靠程度逐步增加信号：

### 9.10.1 结果级 reward

只看最终环境状态。最可靠，但最稀疏。

### 9.10.2 明确的工具事件 reward

例如：

- 成功解析参数；
- 调用返回成功；
- 重复无效调用产生成本；
- 达成已知中间状态。

只有当事件与目标真实相关时才加入，避免模型刷中间分。

### 9.10.3 value / critic

预测当前交互历史最终成功的期望，使用 TD/GAE 将新工具结果造成的价值变化传回前一步。

例如查询后发现订单不可退，value 从 \(0.8\) 降到 \(0.1\)，对应的 TD 修正提示“此前对任务可完成性的判断过于乐观”。若模型随后找到合规替代方案，value 又上升。

### 9.10.4 process verifier

对某些可形式化步骤检查参数、前置条件或状态转换。它比主观地让另一个模型给每段 reasoning 打分更可靠，但覆盖范围有限。

### 9.10.5 分层策略

高层决定“查询—确认—执行—核验”，低层生成工具参数与自然语言。可减少超长 token 轨迹的信用分配距离，但系统复杂度更高。

## 9.11 group-relative 方法怎样用于 Agent

可以从同一初始环境快照采样 \(G\) 条独立轨迹，按最终任务分数和成本排序：

\[
\widehat A_i
=
\frac{
R_i-\bar R
}{
s_R+\varepsilon
}.
\]

但必须满足：

- 每条轨迹从可重置、相同的环境快照开始；
- 工具副作用隔离，不能让第一条轨迹改变第二条的初始状态；
- 随机外部失败要记录，避免误当 policy 质量；
- 严重安全违规轨迹不能仅以较低 reward 混入正常探索；
- 不同长度轨迹的 loss 归一化方式固定。

这通常要求模拟器或 sandbox。直接在生产数据库上为同一用户采样八种退款路径既不安全，也不可重复。

## 9.12 训练数据生命周期

Agent rollout record 除第 7 章字段外，还应保存：

```text
environment_snapshot_id
tool_schema_version
policy_checkpoint_id
reference_checkpoint_id
ordered_tool_calls
tool_arguments
tool_results
permission_decisions
state_diffs
terminal_reason
task_verifier_result
cost_breakdown
safety_events
```

环境或 tool schema 变化后，旧 rollout 的意义可能变化。例如 API 参数从 `amount` 改为 `amount_cents`，旧轨迹不能直接按新 parser 重放并当作同分布数据。

## 9.13 从离线到真实环境的四个阶段

### 阶段 1：offline

使用人工示范、历史安全日志和失败轨迹做 SFT/DPO。目标是先学会工具格式与基本政策，避免从随机探索开始。

### 阶段 2：sandbox

在可重置模拟环境中在线 RL：

- 环境状态可复制；
- 工具调用无真实副作用；
- verifier 可检查最终状态；
- 可进行 group sampling。

### 阶段 3：shadow

模型观察真实请求并提出动作，但不执行；与现有系统/人工操作比较：

- action agreement；
- 参数正确率；
- 预计成功率；
- 安全门控触发率。

### 阶段 4：limited online

只开放低风险、可逆、权限明确的动作：

- 小流量；
- 强制审批；
- 速率和金额限制；
- 实时监控与快速回滚；
- 不以在线用户作为无约束探索环境。

训练收益不能自动授权扩大执行权限。

## 9.14 Agent 评估协议

除第 8 章通用指标外，至少报告：

| 维度 | 指标 |
|---|---|
| 任务 | end-state success、部分完成率 |
| 工具 | 调用成功率、参数正确率、无效重试 |
| 成本 | 调用数、token、延迟、费用 |
| 恢复 | 工具超时/失败后的成功恢复率 |
| 安全 | 越权尝试、敏感信息泄露、未确认不可逆动作 |
| 真实性 | 模型声称成功但环境未成功的比例 |
| 泛化 | 新工具组合、新参数范围、新错误类型 |

还要进行扰动测试：

- 工具返回空值、超时、重复结果；
- API schema 小变化；
- 用户中途改变要求；
- 恶意 tool output 注入；
- 缺少权限；
- 并发状态改变；
- 上下文裁剪。

## 9.15 整门课程的闭环

现在可以把前九章放回同一条问题链：

1. **第 1 章**：把生成看成状态—动作—转移—回报；
2. **第 2 章**：把最终分数变成 sampled-token log-prob 梯度；
3. **第 3 章**：用 value、TD、GAE 构造更稳定的相对信号；
4. **第 4 章**：用 old/current ratio 与 clipping 安全复用 rollout，用 reference KL 控制漂移；
5. **第 5 章**：按示范、偏好、可验证结果选择反馈路径；
6. **第 6 章**：同题多采样时，用组内 baseline 代替 critic，并得到 GRPO；
7. **第 7 章**：用张量契约、生命周期和 smoke test 把数学落地；
8. **第 8 章**：用 held-out、配对比较和失败分析证明真实改进；
9. **第 9 章**：把 token-only 环境替换为会变化、有成本、有权限边界的工具环境。

没有哪一章在发明一套孤立术语。它们都在回答同一个工程目标的下一个障碍：

> 如何让语言模型通过与环境交互获得反馈，在不破坏原有能力和安全边界的前提下，更可靠地完成任务。

## 9.16 本章容易混淆的结论

| 容易误解成 | 正确理解 |
|---|---|
| Agent 是会输出 function call 的 LLM | Agent 还包括环境转移、状态、反馈、权限与终止规则 |
| 模型说“成功”就是任务成功 | 必须检查真实环境最终状态 |
| 工具调用仍只是普通 token | 对模型是 token，对环境是可能有副作用的原子动作 |
| 所有安全问题都可做负 reward | 严重不可逆风险需要执行层硬约束 |
| 生产环境可直接做 group exploration | 需要可重置、隔离的 sandbox |
| Agent RL 需要全新数学 | 核心仍是前面相同的 trajectory、reward、advantage 与受控更新 |

## 9.17 本章自测

1. 工具 Agent 中 state、action、transition、reward 分别是什么？
2. 为什么“模型输出了正确 JSON”和“工具动作成功执行”是两件事？
3. 退款任务的 verifier 为什么必须查询真实订单状态？
4. 哪些成本适合放进 reward，哪些安全规则应做硬门控？
5. 同一初始状态做 group sampling 需要环境满足哪些条件？
6. offline、sandbox、shadow、limited online 各自降低了什么风险？

## 9.18 下一步学习建议

完成本课程后，可以按目标选择深入方向：

- 想研究训练算法：阅读 PPO、GAE、DPO、DeepSeekMath/GRPO 原论文并复现实验；
- 想做推理 RL：研究 verifier、搜索、process supervision 与 test-time compute；
- 想做工程系统：实现第 7 章的 rollout schema、分布式 group sampling 与完整监控；
- 想做 Agent：先构建可重置 sandbox 和 end-state verifier，再考虑在线 RL；
- 想做评估：建立数据污染检查、配对基准、reward hacking 红队与回退套件。

最好的下一个项目不是堆更多算法名，而是在一个小而可靠的可验证任务上，从 SFT baseline、rollout、reward、更新到 held-out 评估完整跑通一次。

## 9.19 对应资料

- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)
- [Toolformer](https://arxiv.org/abs/2302.04761)
- [WebArena](https://arxiv.org/abs/2307.13854)
- [Sutton & Barto, Chapter 17: Frontiers](http://incompleteideas.net/book/RLbook2020.pdf)
