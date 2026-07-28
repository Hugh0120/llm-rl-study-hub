# 大模型强化学习：从回答得分到可靠训练系统

> 面向已经熟悉 Transformer、语言模型训练和基本概率论，但没有系统学习强化学习的算法工程师。
>
> 这不是四周讲义的拼接，也不要求先看 slides。全书只追踪一个项目：**让一个偶尔能答对数学题的语言模型，学会更稳定地生成正确答案。**

## 0.1 这套教程最后要完成什么

我们从一个已经做过监督微调的语言模型开始。它面对：

```text
计算 23 × 17，并把最终答案放在 \boxed{} 中。
```

有时生成：

```text
23×10=230，23×7=161，所以 \boxed{391}
```

也会生成：

```text
23×10=230，23×7=151，所以 \boxed{381}
```

一个程序可以检查最终答案是否等于 391，却不能逐 token 告诉模型应该怎样修改。整套教程要把这个看似简单的接口，发展成一套可以实现、调试和评估的训练系统：

```text
当前模型生成回答
→ 外部系统评分
→ 分数变成策略更新信号
→ 控制同一批回答的复用强度
→ 重新生成
→ 用独立证据判断能力是否真的提高
```

学完后，你应能独立回答：

1. 为什么一个最终分数可以改变前面采样 token 的概率？
2. value、critic、TD 和 GAE 分别在解决哪一步的问题？
3. PPO 为什么同时出现 old policy、current policy 和 reference policy？
4. DPO、RLHF、RLVR、GR-REINFORCE 和 GRPO 的数据接口究竟有什么不同？
5. 一份 GRPO 实现里，哪些张量必须冻结、对齐和加 mask？
6. reward 上升时，怎样证明模型不是只学会利用判题漏洞？
7. 把文本生成扩展成工具 Agent 时，哪些数学不变，哪些环境接口改变？

## 0.2 全书统一符号

后续每章仍会重复本章用到的符号。这里先给一张全局索引，目的是方便回查，不要求现在记住。

| 符号 | English | 在本教程中的含义 |
|---|---|---|
| \(x\) | prompt | 输入题目 |
| \(y=(y_1,\ldots,y_T)\) | completion | 模型生成的完整回答 |
| \(s_t\) | state | prompt 加上已经生成的前缀 |
| \(a_t\) | action | 第 \(t\) 步采样的下一个 token |
| \(\pi_\theta(a_t\mid s_t)\) | policy | 参数为 \(\theta\) 的语言模型给出的 next-token 概率 |
| \(\tau\) | trajectory / rollout | 从 prompt 到回答结束的一次完整生成 |
| \(r_t\) | reward | 第 \(t\) 步之后立即得到的评分 |
| \(G_t\) | return | 从第 \(t\) 步开始累计的后续 reward |
| \(V^\pi(s_t)\) | state value | 从前缀 \(s_t\) 继续生成时的平均未来回报 |
| \(\widehat A_t\) | advantage estimate | 本次选择相对当前正常水平好多少 |
| \(\delta_t\) | TD error | 生成一步后，对未来回报预期的修正 |
| \(\pi_{\text{old}}\) | old policy | 产生当前这批 rollout 的策略快照 |
| \(\pi_{\text{ref}}\) | reference policy | 长期冻结、用于限制行为漂移的参考模型 |

记号的使用遵循三条规则：

1. 每个公式前先用自然语言说清它要计算什么；
2. 新符号第一次出现时解释字母、下标和条件；
3. 公式只是压缩已经讲清的逻辑，不承担第一次解释概念的任务。

## 0.3 全课程只有一条问题链

这套课不按算法名分类，而按前一步留下的问题组织：

```text
语言模型怎样被写成顺序决策问题？
↓
最终分数怎样对离散采样产生梯度？
↓
整份答案共用一个分数，信号太粗怎么办？
↓
rollout 很贵，怎样安全复用？
↓
真实任务中的 reward 从哪里来？
↓
不训练 critic，怎样用同题多次采样构造参照？
↓
公式怎样落到 tensor、mask 和训练循环？
↓
怎样证明 reward 上升代表真实能力上升？
↓
工具调用把状态与动作改成什么？
```

每一章开头都会明确写出：

- 上一章已经解决了什么；
- 还剩下什么具体问题；
- 本章只解决其中哪一个。

## 0.4 章节目录

| 章节 | 本章只回答的核心问题 | 最终得到什么 |
|---|---|---|
| 第 1 章 | LLM 生成为什么是顺序决策？ | 状态、动作、策略、轨迹、回报和目标 |
| 第 2 章 | 最终得分怎样改变 token 概率？ | Policy Gradient 与 REINFORCE |
| 第 3 章 | 怎样把整份答案的分数变成更合理的逐步权重？ | baseline、value、critic、TD、GAE |
| 第 4 章 | 为什么旧 rollout 会失效，怎样有限复用？ | importance ratio、PPO、两类 KL |
| 第 5 章 | 没有天然 reward 时，训练信号从哪里来？ | SFT、reward model、RLHF、DPO、RLVR |
| 第 6 章 | 同一道题采样多次，怎样省掉 critic？ | group baseline、GR-REINFORCE、GRPO |
| 第 7 章 | 这些公式怎样变成不会悄悄算错的代码？ | tensor contract、mask、数据生命周期与调试 |
| 第 8 章 | reward 涨了，凭什么说模型更强？ | 基线、统计证据、消融与 reward hacking 审计 |
| 第 9 章 | 文本回答怎样扩展成多步工具 Agent？ | Agent MDP、环境验证、信用分配与安全上线 |

## 0.5 每章固定的阅读结构

为了避免读到一半才知道本章在讲什么，每章都使用同一模板：

1. **本章符号说明**：只列本章马上要使用的量；
2. **本章目标**：读完能够完成哪些动作；
3. **本章主线**：用一段话说明上一章如何推出本章；
4. **本章新增概念**：英文、中文、解决的问题和与当前公式的关系；
5. **正文推导**：每一步都回答“为什么现在需要它”；
6. **贯穿例子**：回到同一条数学回答做数值计算；
7. **工程落点**：对应到 LLM 的 log-prob、mask、rollout 或训练代码；
8. **自测问题**：不用翻正文也能回答，才算完成。

## 0.6 怎样读公式

碰到一个公式，按下面顺序读：

```text
左边是什么问题的答案？
→ 右边每一项来自哪条数据？
→ 哪些量由当前模型计算？
→ 哪些量是冻结的训练标签？
→ 这个估计的偏差和方差从哪里来？
→ 它失败时日志会怎样变化？
```

例如后面看到：

\[
\widehat A_t=G_t-V(s_t)
\]

不要先背名字。先读成：

> 这次实际得到的后续回报，减去模型从当前前缀出发通常能得到的回报。

当这句话已经清楚，advantage 只是在给它命名。

## 0.7 建议学习顺序

如果你没有 RL 基础，严格按第 1–7 章顺序阅读。第 8、9 章可以在第一次实现后再读。

每章建议完成三件事：

1. 手算本章的贯穿例子；
2. 把本章主线用自己的话复述一遍；
3. 在代码或伪代码中指出每个数学量来自哪里。

如果时间有限，优先顺序是：

```text
1 → 2 → 3 → 4 → 6 → 7 → 8
```

第 5 章帮助你选择训练接口，第 9 章负责迁移到工具环境。

## 0.8 官方资料怎样使用

正文独立可读，官方课件用于核对推导和继续深入：

| 主题 | 官方课件 | 视频 |
|---|---|---|
| RL 基本设定 | [CS285 Lecture 4](https://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-4.pdf) | [CS285 Fall 2023 播放列表](https://www.youtube.com/playlist?list=PL_iWQOsE6TfVYGEGiAOMaOzzv41Jfm_Ps&index=4) |
| Policy Gradient | [CS285 Lecture 5](https://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-5.pdf) | [Lecture 5](https://www.youtube.com/playlist?list=PL_iWQOsE6TfVYGEGiAOMaOzzv41Jfm_Ps&index=5) |
| Actor–Critic | [CS285 Lecture 6](https://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-6.pdf) | [Lecture 6](https://www.youtube.com/playlist?list=PL_iWQOsE6TfVYGEGiAOMaOzzv41Jfm_Ps&index=6) |
| PPO / advanced PG | [CS285 Lecture 10](https://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-10.pdf) | [课程主页](https://rail.eecs.berkeley.edu/deeprlcourse/) |
| LLM preference optimization | [CS224R Lecture 9](https://cs224r.stanford.edu/spring_2025/slides/09_cs224r-2025-rlhf.pdf) | [CS224R Spring 2025](https://www.youtube.com/playlist?list=PLoROMvodv4rPwxE0ONYRa_itZFdaKCylL&index=9) |
| LLM reasoning RL | [CS224R Lecture 10](https://cs224r.stanford.edu/spring_2025/slides/10_cs224r-rl_for_reasoning_lecture.pdf) | [Lecture 10](https://www.youtube.com/playlist?list=PLoROMvodv4rPwxE0ONYRa_itZFdaKCylL&index=10) |
| 实现作业 | [CS285 Homework 4](https://rail.eecs.berkeley.edu/deeprlcourse/static/homeworks/hw4.pdf) | [官方代码](https://github.com/berkeleydeeprlcourse/homework_spring2026) |

下一章从项目的第一行代码开始：让模型自己生成一条回答。
