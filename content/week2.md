# 第二周：从偏好数据到可验证奖励

> 适用对象：已完成第一周，理解策略梯度、advantage、PPO ratio 与 KL。
>
> 这是一篇可独立阅读的教材。正文会从 pairwise preference 推导 reward model 与 DPO，再把 verifier、RLVR 和 GRPO 串成一条完整链；课件只在附录作为可选出处。

## 本周最终产出

读完后，你应能根据业务数据回答三件事：

1. 我的反馈是示范答案、偏好对，还是可自动验证的 reward？
2. 应该使用 SFT、DPO、RLHF + PPO，还是 RLVR + GRPO？
3. reward 上升时，怎样判断模型是真的变强，而不是学会投机？

整周主线：

```text
反馈数据
→ 把“什么算好”变成训练信号
→ 选择离线或在线优化方式
→ 用 KL / clip 控制策略漂移
→ 用独立评估发现 reward hacking
```

---

## 0. 阅读地图与贯穿例子

本周继续使用数学问答：

```text
prompt x：计算 17 × 23，并给出简短推理。
```

模型给出两个回答：

```text
y_w：17 × 23 = 391。因为 17 × 20 = 340，17 × 3 = 51，相加为 391。
y_l：17 × 23 = 381。
```

同一组数据可以产生不同训练信号：

| 反馈形式 | 数据长什么样 | 可采用的方法 |
|---|---|---|
| 专家示范 | \((x,y_w)\) | SFT |
| 人类偏好 | \((x,y_w,y_l)\)，只知道前者更好 | reward model、DPO |
| 自动验证 | `parse(y)==391` | RLVR、GRPO、REINFORCE、PPO |
| 标量评分 | \(r(x,y)\in\mathbb R\) | RLHF + PPO，或作为离线筛选信号 |

本周的关键不是背算法名，而是理解：**反馈接口决定你能构造什么目标函数。**

---

# 第一章：后训练方法其实在回答两个问题

**问题一：谁来定义“好回答”？问题二：优化时能不能让当前模型重新采样？**

## 1.1 两条正交轴

第一条轴是 reward 来源：

- **示范**：直接给出目标 token；
- **偏好**：只说两个回答谁更好；
- **reward model**：用模型预测人类偏好；
- **verifier**：用规则、执行器或环境直接检查结果。

第二条轴是数据方式：

- **离线**：训练期间只使用固定数据；
- **在线**：当前策略持续生成新回答，再获得反馈。

把常见方法放到同一张表：

| 方法 | reward/监督来源 | 在线采样 | 核心目标 | 主要代价 |
|---|---|---:|---|---|
| SFT | 专家答案 | 否 | 最大化示范 token likelihood | 只能模仿数据覆盖到的行为 |
| DPO | chosen/rejected 偏好对 | 否 | 提高 chosen 相对 reference 的优势 | 受固定偏好数据覆盖限制 |
| RLHF + PPO | 学到的 reward model | 是 | 最大化预测 reward，同时约束 KL | rollout、RM 推理和 PPO 都较复杂 |
| RLVR + GRPO | 自动 verifier | 是 | 从当前策略探索中提高可验证成功率 | verifier 设计与采样成本 |

## 1.2 为什么 SFT 不是 RL 的廉价替代品？

SFT 的损失：

\[
\mathcal L_{\mathrm{SFT}}
=-\sum_t\log\pi_\theta(y_t^*\mid x,y_{<t}^*).
\]

它要求一个明确的参考回答 \(y^*\)。即使另一条回答也正确，只要 token 不同，SFT 仍把它当作非目标序列。

RL 的目标：

\[
J(\theta)=
\mathbb E_{y\sim\pi_\theta(\cdot\mid x)}[r(x,y)].
\]

它允许很多不同回答获得相同 reward。模型可以在当前能力附近探索新的正确路径，而不是只复现示范。

因此：

```text
有唯一或高质量示范、需要冷启动 → 先 SFT
只有相对偏好、没有可靠标量 reward → DPO 或 reward modeling
能自动检查结果、希望在线探索 → RLVR
```

### 第一章小结

算法选择首先是数据接口选择。下一章先处理最常见的人类反馈：人们往往难以给出校准良好的绝对分数，但更容易比较两个答案谁更好。

---

# 第二章：从偏好对训练 reward model

## 2.1 为什么使用成对比较？

让标注者分别给两个回答打 `7.3` 和 `6.8` 分，存在尺度和校准问题；问“哪个更好”通常更稳定。

偏好数据记为：

\[
\mathcal D=
\{(x,y_w,y_l)\},
\]

其中 \(y_w\) 是 chosen/winner，\(y_l\) 是 rejected/loser。

reward model \(r_\phi(x,y)\) 输出一个标量。我们不要求它的绝对值有意义，只要求 chosen 的分数高于 rejected。

## 2.2 Bradley-Terry 偏好模型

假设人类选择 \(y_w\) 的概率由 reward 差决定：

\[
P_\phi(y_w\succ y_l\mid x)
=
\frac{\exp(r_\phi(x,y_w))}
{\exp(r_\phi(x,y_w))+\exp(r_\phi(x,y_l))}.
\]

分子分母同时除以 \(\exp(r_\phi(x,y_w))\)：

\[
\begin{aligned}
P_\phi(y_w\succ y_l\mid x)
&=\frac{1}
{1+\exp(r_\phi(x,y_l)-r_\phi(x,y_w))}\\
&=\sigma\left(
r_\phi(x,y_w)-r_\phi(x,y_l)
\right).
\end{aligned}
\]

所以 reward model 的负对数似然为：

\[
\boxed{
\mathcal L_{\mathrm{RM}}(\phi)
=-\mathbb E_{(x,y_w,y_l)\sim\mathcal D}
\log\sigma\left(
r_\phi(x,y_w)-r_\phi(x,y_l)
\right)
}
\]

这就是一个二分类损失：reward 差越大，模型越确信 chosen 更好。

## 2.3 reward 分数为什么不能当作真实效用？

损失只使用差值：

\[
r_\phi(x,y_w)-r_\phi(x,y_l).
\]

给所有回答同时加常数 \(c(x)\)，偏好概率完全不变：

\[
(r_w+c)-(r_l+c)=r_w-r_l.
\]

因此 reward 的零点不可识别；其尺度也受温度、数据和正则化影响。工程上不要比较两个不同 reward model 的原始均值，也不要把 `reward=3.7` 解释为客观质量。

真正应监控：

- held-out preference accuracy；
- 不同 prompt 域上的校准；
- chosen/rejected margin 分布；
- 长度与 reward 的相关性；
- 对抗样例和分布外回答。

## 2.4 用 reward model 做 RLHF

训练好 \(r_\phi\) 后，复制 SFT 模型得到可训练策略 \(\pi_\theta\)，并保留冻结 reference \(\pi_{\mathrm{ref}}\)：

\[
\max_\theta
\mathbb E_{x\sim\mathcal D,\,
y\sim\pi_\theta(\cdot\mid x)}
\left[
r_\phi(x,y)
-\beta
\log\frac{\pi_\theta(y\mid x)}
{\pi_{\mathrm{ref}}(y\mid x)}
\right].
\]

对 \(y\sim\pi_\theta\) 取期望后，第二项就是：

\[
-\beta D_{\mathrm{KL}}
(\pi_\theta(\cdot\mid x)\|
\pi_{\mathrm{ref}}(\cdot\mid x)).
\]

它表达一个明确交易：

```text
提高 reward model 分数
但每偏离 reference 一点，都要支付 β 控制的代价
```

若 \(\beta\) 太小，策略容易钻 reward model 漏洞；若太大，策略几乎无法离开 SFT 模型。

### 第二章小结

reward model 把离散偏好变成可在线优化的标量，但引入了一个可被策略利用的近似模型。下一章会看到：DPO 可以把“训练 RM + RL 优化”合并成一个直接的偏好分类目标。

---

# 第三章：DPO 为什么能绕过显式 reward model？

## 3.1 从 KL 正则化最优策略开始

对固定 prompt \(x\)，考虑：

\[
\max_\pi
\sum_y\pi(y\mid x)r(x,y)
-\beta
D_{\mathrm{KL}}(\pi(\cdot\mid x)\|
\pi_{\mathrm{ref}}(\cdot\mid x)).
\]

加入概率和为 \(1\) 的约束并求最优解，可得到：

\[
\pi^*(y\mid x)
=\frac{1}{Z(x)}
\pi_{\mathrm{ref}}(y\mid x)
\exp\left(\frac{r(x,y)}{\beta}\right),
\]

其中 \(Z(x)\) 是只依赖 prompt 的归一化常数。

两边取对数并整理：

\[
r(x,y)
=\beta\log
\frac{\pi^*(y\mid x)}
{\pi_{\mathrm{ref}}(y\mid x)}
+\beta\log Z(x).
\]

对 chosen 与 rejected 做差，\(\log Z(x)\) 抵消：

\[
\begin{aligned}
r(x,y_w)-r(x,y_l)
=\beta\Bigg[
&\log\frac{\pi^*(y_w\mid x)}
{\pi_{\mathrm{ref}}(y_w\mid x)}\\
-&\log\frac{\pi^*(y_l\mid x)}
{\pi_{\mathrm{ref}}(y_l\mid x)}
\Bigg].
\end{aligned}
\]

## 3.2 代回 Bradley-Terry 得到 DPO

用当前策略 \(\pi_\theta\) 近似最优策略，把上面的 reward 差代入偏好概率：

\[
\boxed{
\mathcal L_{\mathrm{DPO}}(\theta)
=-\mathbb E_{\mathcal D}
\log\sigma\left(
\beta[
\log\pi_\theta(y_w\mid x)-\log\pi_{\mathrm{ref}}(y_w\mid x)
-\log\pi_\theta(y_l\mid x)+\log\pi_{\mathrm{ref}}(y_l\mid x)
]
\right)
}
\]

定义相对 reference 的隐式 reward：

\[
\hat r_\theta(x,y)
=\beta\log
\frac{\pi_\theta(y\mid x)}
{\pi_{\mathrm{ref}}(y\mid x)}.
\]

DPO 就是在提高：

\[
\hat r_\theta(x,y_w)-\hat r_\theta(x,y_l).
\]

它不是简单地“提高 chosen、降低 rejected”；reference log-ratio 决定了变化的基准。

## 3.3 \(\beta\) 的作用

- 较大的 \(\beta\)：DPO logit 对 log-ratio 差更敏感，通常更强调接近 reference 的正则化解释；
- 较小的 \(\beta\)：允许更激进的相对偏移，但也可能造成训练不稳定或过拟合偏好对。

不同实现对 \(\beta\) 的直觉表述可能略有差异，最终应查看：

- chosen/rejected reward margin；
- policy 与 reference 的 KL；
- held-out win rate；
- 长度和格式偏移。

## 3.4 DPO 与 RLHF 的真正差别

| 维度 | DPO | RLHF + PPO |
|---|---|---|
| 数据 | 固定偏好对 | 当前策略在线 rollout |
| reward | 隐式存在于 log-ratio | 显式 reward model |
| 优化 | 类似二分类/SFT 的离线训练 | on-policy policy optimization |
| 探索 | 无法主动生成新分布数据 | 能探索当前策略附近的新回答 |
| 复杂度 | 较低 | 较高 |
| 风险 | 数据覆盖、偏好噪声 | RM exploitation、训练不稳、成本高 |

若业务有高质量固定偏好对、在线采样昂贵，DPO 是自然起点；若模型必须通过尝试发现新推理路径，在线 RL 更合适。

---

# 第四章：从 reward model 转向 verifier

## 4.1 verifier 与 reward model 的差别

reward model 在模仿人类判断；verifier 直接检查任务约束。

| 类型 | 示例 | 优点 | 风险 |
|---|---|---|---|
| 规则 verifier | JSON schema、正则、格式检查 | 快、可解释 | 容易只学格式 |
| 执行 verifier | 代码单测、SQL 执行、工具返回值 | 与真实任务接近 | 沙箱、安全、环境不稳定 |
| 答案 verifier | 数学答案 exact match | 便宜、客观 | 解析错误、忽略推理质量 |
| 模型 judge | 另一个 LLM 评分 | 覆盖开放任务 | 偏差、串通、提示注入 |

可自动验证结果的在线 RL 常称为 **RLVR**：reinforcement learning with verifiable rewards。

## 4.2 outcome reward 与 process reward

结果奖励：

\[
r(x,y)=\mathbf 1[\operatorname{answer}(y)=y^*].
\]

优点是定义清楚；缺点是稀疏，无法区分“差一点正确”和“完全胡说”。

过程奖励为中间步骤 \(z_t\) 打分：

\[
R(y)=\sum_t r_{\mathrm{process}}(z_t)
+r_{\mathrm{outcome}}(y).
\]

它能改善 credit assignment，但前提是过程检查器可靠。若过程 reward 只是另一个容易欺骗的模型，稠密信号也可能放大错误。

## 4.3 一个合格 verifier 的设计顺序

以数学回答为例：

1. **规范输出协议**：例如最终答案放在 `\boxed{}`；
2. **解析与验证分离**：parser 只提取答案，verifier 只比较语义；
3. **拒绝解析失败**：不要默认缺失字段等于零或空字符串；
4. **规范化**：处理空格、整数符号、等价分数，但不要过度宽松；
5. **对抗测试**：多答案、重复 box、提示注入、超长文本、Unicode 混淆；
6. **独立评估**：训练 verifier 与最终评估器尽量不同实现。

伪代码：

```python
def verify(prompt, completion, expected):
    parsed = parse_boxed_answer(completion)
    if parsed is None:
        return {"correct": 0.0, "format": 0.0}

    correct = float(canonicalize(parsed) == canonicalize(expected))
    format_ok = float(has_exactly_one_boxed_answer(completion))
    return {
        "correct": correct,
        "format": format_ok,
        "total": correct + 0.1 * format_ok,
    }
```

必须分别记录 `correct` 与 `format`。如果只看 `total`，模型可能靠格式奖励上涨掩盖准确率不变。

---

# 第五章：GRPO 如何把同题样本变成 baseline？

## 5.1 同一个 prompt 采样一组回答

对每个 prompt \(x_i\)，采样 \(G\) 个回答：

\[
y_{i,1},\ldots,y_{i,G}
\sim\pi_{\theta_{\mathrm{old}}}(\cdot\mid x_i),
\]

得到 rewards：

\[
r_{i,1},\ldots,r_{i,G}.
\]

组均值和标准差：

\[
\mu_i=\frac1G\sum_{j=1}^G r_{i,j},
\qquad
\sigma_i=
\sqrt{\frac1G\sum_{j=1}^G(r_{i,j}-\mu_i)^2}.
\]

最常见的组相对 advantage：

\[
\hat A_{i,j}
=\frac{r_{i,j}-\mu_i}
{\sigma_i+\varepsilon_{\mathrm{num}}}.
\]

它在问：

> 同一道题的这些回答中，这条回答比组内平均水平好多少？

这与第一周的 \(A=Q-V\) 同构，只是 baseline 不再来自单独训练的 value model，而来自同 prompt 样本组。

## 5.2 一个数值例子

同一道题采样四条回答，reward 为：

\[
[1,\;1,\;0,\;0].
\]

均值 \(\mu=0.5\)，标准差 \(\sigma=0.5\)，所以 advantage：

\[
[+1,\;+1,\;-1,\;-1].
\]

正确回答获得正更新，错误回答获得负更新。

若四条全部错误：

\[
[0,0,0,0],
\]

组内方差为零，所有相对 advantage 接近零。这一组不会提供方向，因为当前采样没有证据说明哪条更好。解决方式不是强行除以极小数，而是提高探索、group size、模型初始化或 reward 分辨率。

## 5.3 GRPO 的策略目标

GRPO 通常仍复用 PPO 的 token ratio：

\[
\rho_{i,j,t}(\theta)=
\frac{
\pi_\theta(y_{i,j,t}\mid x_i,y_{i,j,<t})
}{
\pi_{\theta_{\mathrm{old}}}
(y_{i,j,t}\mid x_i,y_{i,j,<t})
}.
\]

clipped objective：

\[
\mathcal L_{\mathrm{GRPO}}
=-\mathbb E_{i,j,t}
\min\left(
\rho_{i,j,t}\hat A_{i,j},
\operatorname{clip}
(\rho_{i,j,t},1-\epsilon,1+\epsilon)\hat A_{i,j}
\right)
+\beta\mathcal L_{\mathrm{KL}}.
\]

同一个 sequence-level advantage 会广播到该回答的有效 token。若有过程 reward，也可以构造 token-level advantage。

## 5.4 GRPO、GR-REINFORCE 与 PPO

| 方法 | baseline | 是否复用 rollout | 是否 clip | 是否训练 critic |
|---|---|---:|---:|---:|
| PPO | value model / GAE | 少量 epoch | 是 | 是 |
| GRPO | 同 prompt group | 少量 epoch | 是 | 否 |
| GR-REINFORCE | 同 prompt group | 单次 on-policy | 否或不需要 | 否 |

GRPO 更省去 critic 显存，但要为每个 prompt 生成多条回答；它省的是 value 模型，不一定省 rollout 算力。

---

# 第六章：怎样选择 SFT、DPO、PPO 或 GRPO？

按下面顺序决策：

```text
1. 有可靠专家答案吗？
   有 → 先 SFT 冷启动

2. 有固定 chosen/rejected 偏好对，但无法自动评分吗？
   有 → 先 DPO

3. 能对当前模型新生成的回答自动评分吗？
   有 → 在线 RL / RLVR

4. reward 稀疏但同题比较有意义，critic 又昂贵吗？
   是 → GRPO 或 GR-REINFORCE

5. 需要 token-level value、长时序过程 reward 或更成熟的控制吗？
   是 → PPO + critic / GAE
```

实际系统常组合使用：

```text
预训练
→ SFT 建立基本指令能力
→ DPO 对齐固定偏好
→ RLVR 在可验证任务上在线探索
→ 继续用 reference KL 保持通用能力
```

---

# 第七章：训练与评估的完整检查表

## 7.1 数据层

- preference 数据是否随机交换过左右顺序，避免位置偏差？
- chosen 是否真的比 rejected 好，而不只是更长？
- prompt 是否泄漏答案或评估集？
- verifier parser 是否有单测和对抗样例？

## 7.2 优化层

- DPO 是否同时加载冻结 reference？
- GRPO 每个 prompt 是否真的形成完整 group？
- advantage 是否只在组内归一化？
- 全相同 reward 的组是否安全地产生零信号？
- old log-prob 是否来自 rollout 时的策略？
- KL、clip fraction、entropy 和梯度范数是否记录？

## 7.3 评估层

- task reward 与 held-out 指标是否同时上涨？
- 长度、格式通过率和答案正确率是否拆开？
- 是否人工查看高 reward 失败样例？
- 是否比较相同采样预算和相同 prompt 集？
- 是否至少运行多个 seed 或给出置信区间？

---

# 第八章：完成检查

请合上正文回答：

1. Bradley-Terry 为什么只依赖 reward 差？
2. reward model 的绝对分数为什么不可直接解释？
3. 从 KL 正则化最优策略怎样得到 DPO 的 log-ratio？
4. DPO 与 RLHF 的在线/离线差别是什么？
5. verifier 与 reward model 的失败方式有什么不同？
6. 为什么 GRPO 不需要 value model？
7. 同组 reward 全相同时，GRPO 为什么没有学习信号？
8. old policy 与 reference policy 分别做什么？

<details>
<summary><strong>展开查看答案要点</strong></summary>

1. Bradley-Terry 概率可写成 \(\sigma(r_w-r_l)\)，共同平移会抵消。
2. pairwise loss 不能识别共同加性常数，尺度也受训练设置影响。
3. KL 正则化最优策略满足 \(\pi^*\propto\pi_{\mathrm{ref}}\exp(r/\beta)\)，取 log 并做 chosen/rejected 差即可消去归一化常数。
4. DPO 使用固定偏好对；RLHF + PPO 用当前策略持续生成回答并由 RM 打分。
5. RM 会被策略利用统计偏差；verifier 会被 parser、规则漏洞、执行环境或目标定义漏洞利用。
6. 同 prompt 的 group mean/std 直接提供相对 baseline。
7. 没有样本间相对差异，减均值后 advantage 全为零。
8. old policy 为本批 rollout 与 ratio 提供分母；reference policy 是长期 KL 锚点。

</details>

---

# 附录：可选原始资料

- [CS224R · Reward Learning](https://cs224r.stanford.edu/spring_2025/slides/08_cs224r_reward_learning_2025.pdf)
- [CS224R · RLHF / Preference Optimization](https://cs224r.stanford.edu/spring_2025/slides/09_cs224r-2025-rlhf.pdf)
- [CS224R · RL for Reasoning](https://cs224r.stanford.edu/spring_2025/slides/10_cs224r-rl_for_reasoning_lecture.pdf)
- [CS285 · LLM RL](https://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-14.pdf)

