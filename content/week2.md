# 第二周：没有唯一标准答案时，训练信号从哪里来

> 第一周假设外部系统已经能给回答打分。本周从更早的问题开始：人或程序究竟能提供什么反馈？反馈接口不同，能构造出的训练目标也不同。

第一周已经集中讲完本课程反复使用的 RL 地基：模型怎样逐步生成、最终得分怎样
改变采样概率、怎样构造相对更新权重，以及旧回答为什么只能受限复用。本周不再补一套
新的 RL 前置知识，只改变一个问题：**外部反馈究竟从哪里来？**

## 本周最终产出

完成后，你应能先根据数据回答三个问题：有没有目标答案、能不能比较两个回答、能不能自动判定结果；然后再选择对应训练方法，而不是从算法缩写反推问题。

贯穿全文的任务是数学推理：

```text
prompt: 求解 23 × 17，并把最终答案写在 \boxed{} 中。
```

模型可能输出：

- \(y_1\)：推理正确，`\boxed{391}`；
- \(y_2\)：格式正确，答案错误；
- \(y_3\)：答案正确，但没有按格式输出；
- \(y_4\)：语言流畅，却没有完成计算。

我们会分别问：如果手中是标准答案、偏好对或自动判题器，能学到什么？

## 0. 先只看反馈，不看算法名

| 手中反馈 | 能直接知道什么 | 可以直接构造什么 | 仍缺什么 |
|---|---|---|---|
| 专家示范 \(y^\star\) | “应该输出什么” | 逐 token 模仿目标 | 不知道模型自己的错误分布 |
| 偏好对 \(y_w\succ y_l\) | “两者谁更好” | 两个回答的相对目标 | 没有绝对分数 |
| 自动判题程序 | “结果是否满足规则” | 对模型新回答持续打分 | 信号常稀疏，需要探索 |

本周依赖链：

\[
\text{反馈接口}
\rightarrow
\text{可训练目标}
\rightarrow
\text{是否需要在线采样}
\rightarrow
\text{baseline 与更新约束}.
\]

# 第一章：先分清三种反馈，不要先背算法

## 1.1 示范：把目标回答当监督标签

先用自然语言说清一条数据：第 \(i\) 条记录包含 prompt \(x_i\)
和专家回答 \(y_i^\star\)，星号表示“这里把它当作目标答案”。把整批示范数据记作
\(\mathcal D_{\text{demo}}\)：\(\mathcal D\) 表示 dataset，下标 `demo`
表示 demonstration。于是：

\[
\mathcal D_{\text{demo}}=\{(x_i,y_i^\star)\},
\]

这种用专家 token 做监督目标的训练叫
**supervised fine-tuning（监督微调，SFT）**。最自然的目标是最大似然。把训练时最小化的 loss 记作
\(\mathcal L_{\text{SFT}}\)：\(\mathcal L\) 表示 loss，下标说明训练方法：

\[
\mathcal L_{\text{SFT}}
=
-\mathbb E_{(x,y^\star)\sim\mathcal D}
\sum_t\log\pi_\theta(y_t^\star\mid x,y^\star_{<t}).
\]

它告诉模型沿着专家轨迹每一步应该预测什么 token。

SFT 很强，因为监督密集、训练稳定。但它优化的是“复现数据中的答案”，不是“从当前模型会犯的错误中继续探索”。若专家数据覆盖不足，模型不会自动看到自己最新策略下的新失败模式。

## 1.2 偏好：只问两个回答谁更好

开放式任务往往没有唯一标准答案。让标注者分别给 1–10 分，容易受个人尺度影响；比较两个回答谁更好通常更稳定。

这时第 \(i\) 条数据包含同一个 prompt \(x_i\)、胜出回答
\(y_{i,w}\) 和落败回答 \(y_{i,l}\)。把整批偏好数据记作
\(\mathcal D_{\text{pref}}\)，下标 `pref` 表示 preference：

\[
\mathcal D_{\text{pref}}
=
\{(x_i,y_{i,w},y_{i,l})\},
\]

其中 \(w\) 是 preferred/chosen，\(l\) 是 rejected。

偏好只提供顺序：

\[
y_w\succ y_l,
\]

没有告诉我们 \(y_w\) 是 8.3 分，也没有说两个回答差多少。第二章会先从这一事实推出一个能学习相对质量的模型。

## 1.3 自动验证：程序可以判定结果

数学、代码、结构化抽取等任务，常有程序化检查：

```python
def verifier(prompt, completion):
    parsed = parse_boxed_answer(completion)
    return float(parsed == ground_truth[prompt])
```

这类反馈不需要人工逐条比较，可以对模型新生成的回答持续打分。由可验证奖励驱动的在线强化学习常称为
**reinforcement learning with verifiable rewards（RLVR）**。

它的优势是便宜、可扩展；局限也直接来自接口：

- 只有最终 0/1，信号稀疏；
- parser 或测试用例可能有漏洞；
- 判对结果不等于判对推理过程；
- 如果当前策略从未采样成功，reward 无法告诉它成功方向。

## 1.4 两个判断轴

看见一个后训练任务，先问：

1. **反馈来自哪里？** 示范、偏好还是 verifier？
2. **训练数据由谁产生？** 固定离线数据，还是当前策略在线采样？

| 训练起点 | 反馈 | 数据分布 |
|---|---|---|
| 模仿专家回答 | 示范 | 固定离线数据 |
| 直接学习回答间的相对关系 | 偏好对 | 通常是固定离线数据 |
| 学一个评分器后在线优化 | 学到的标量分数 | 当前模型的新回答 |
| 用程序评分后在线优化 | 自动判题结果 | 当前模型的新回答 |

这是整周最重要的分类法。

# 第二章：偏好如何变成一个可学习的标量

## 2.1 从“谁更好”到选择概率

设一个未知函数 \(r_\phi(x,y)\) 表示回答的潜在质量，其中 \(\phi\)
是这个打分模型的参数。

我们要预测“胜出回答被选中”的概率。把这个概率记作
\(P_\phi(y_w\succ y_l\mid x)\)：大写 \(P\) 表示 probability，下标
\(\phi\) 表示概率由参数为 \(\phi\) 的打分模型决定。用
\(\sigma\) 表示把任意实数压到 0–1 的 sigmoid 函数，于是：

\[
P_\phi(y_w\succ y_l\mid x)
=
\sigma\left(
r_\phi(x,y_w)-r_\phi(x,y_l)
\right),
\]

sigmoid 的具体定义是：

\[
\sigma(z)=\frac{1}{1+e^{-z}}.
\]

这个 Bradley–Terry 形式只依赖分数差：

- 差为 0，预测偏好概率为 0.5；
- \(y_w\) 高很多，概率趋近 1；
- 顺序反过来，概率趋近 0。

对已标注的胜者最大化似然，就能训练这个标量打分模型。
这个根据偏好对学习回答标量分数的模型叫 **reward model（奖励模型）**。
它的 loss 记作 \(\mathcal L_{\text{RM}}\)：\(\mathcal L\) 表示 loss，
下标 RM 表示 reward model：

\[
\mathcal L_{\text{RM}}(\phi)
=
-\mathbb E_{(x,y_w,y_l)\sim\mathcal D_{\text{pref}}}
\log\sigma\left(
r_\phi(x,y_w)-r_\phi(x,y_l)
\right).
\]

## 2.2 手算一次偏好概率

若：

\[
r_\phi(x,y_w)=2.0,\qquad r_\phi(x,y_l)=0.5,
\]

则分数差为 1.5：

\[
P(y_w\succ y_l)=\sigma(1.5)\approx0.818.
\]

训练若希望该概率更接近 1，就会继续拉大胜者与败者的分差。

## 2.3 reward model 的数字没有天然绝对单位

因为 loss 只看差值，对同一个 prompt 同时加上常数 \(c(x)\)：

\[
r'_\phi(x,y)=r_\phi(x,y)+c(x),
\]

偏好概率完全不变。

因此：

- reward model 的 3.2 不是“真实效用 3.2”；
- 不同版本 reward model 的均值不能直接横比；
- 模型只在偏好数据覆盖区域内相对可信；
- 策略可能找到 reward model 的漏洞，把预测分推高却没有真正变好。

## 2.4 把学到的评分器放回在线训练需要哪些角色

得到 reward model 后，可以让当前策略在线生成：

\[
y\sim\pi_\theta(\cdot\mid x),
\]

并最大化：

\[
\mathbb E[r_\phi(x,y)].
\]

但只追逐一个不完美预测器容易偏离自然语言能力，因此加入 reference 约束。
\(\pi_{\text{ref}}\) 是冻结的参考策略；\(\beta>0\) 控制“追逐 reward”
与“留在 reference 附近”的折中。下面的 \(\max_\theta\) 表示要寻找一组
模型参数 \(\theta\)，使整个括号的平均值最大：

\[
\max_\theta\;
\mathbb E_{y\sim\pi_\theta}
\left[
r_\phi(x,y)
-\beta\log
\frac{\pi_\theta(y\mid x)}
{\pi_{\text{ref}}(y\mid x)}
\right].
\]

这种“从人类偏好训练 reward model，再在线优化策略”的路线叫
**reinforcement learning from human feedback（RLHF）**。一套 PPO 式 RLHF 常包含：

| 组件 | 是否更新 | 作用 |
|---|---:|---|
| policy | 是 | 生成并改进回答 |
| reward model | 通常否 | 近似人类偏好 |
| value/critic | 是 | 预测 policy 的未来回报，构造 advantage |
| reference policy | 否 | 约束长期漂移 |

顺序是：

\[
\text{SFT}
\rightarrow
\text{收集偏好对}
\rightarrow
\text{训练 reward model}
\rightarrow
\text{在线 rollout + PPO}.
\]

这一管线表达力强，但在线采样、value 拟合和 PPO 调参都增加了复杂度。下一章问：如果手里已经有固定偏好对，能否直接训练 policy？

# 第三章：固定偏好对能否直接训练语言模型

## 3.1 先解一个“如果 reward 已知”的最优策略

固定 prompt \(x\)，考虑所有回答 \(y\)。用 \(\pi(y\mid x)\) 表示
我们暂时要寻找的回答概率分布；这里先优化整个分布 \(\pi\)，还不是直接优化神经网络参数。
我们想同时：

1. 提高期望 reward；
2. 不要离 reference policy 太远。

公式前先固定三个量：

- \(r(x,y)\)：回答 \(y\) 在 prompt \(x\) 下的 reward；
- \(\pi_{\text{ref}}(y\mid x)\)：第一周已经介绍的长期参考模型，本节保持不变；
- \(\beta>0\)：偏离参考模型的惩罚强度；越大越不允许新分布远离参考模型。

\[
\max_\pi
\sum_y\pi(y\mid x)r(x,y)
-\beta\sum_y\pi(y\mid x)
\log\frac{\pi(y\mid x)}{\pi_{\text{ref}}(y\mid x)}.
\]

还必须满足：

\[
\sum_y\pi(y\mid x)=1.
\]

对这个带归一化约束的问题求最优解，可得最优分布。
\(\pi^\star\) 上的星号表示“最优解”；\(Z(x)\) 是 normalization
constant，只负责让所有回答概率加起来等于 1：

\[
\pi^\star(y\mid x)
=
\frac{1}{Z(x)}
\pi_{\text{ref}}(y\mid x)
\exp\left(\frac{r(x,y)}{\beta}\right),
\]

这句话比公式更重要：

> 最优策略等于 reference policy，再按 reward 的指数权重重新分配概率。

## 3.2 反解 reward

对上式取对数并整理：

\[
r(x,y)
=
\beta\log
\frac{\pi^\star(y\mid x)}
{\pi_{\text{ref}}(y\mid x)}
+\beta\log Z(x).
\]

偏好 loss 只需要两个回答的 reward 差，因此同一个 prompt 的 \(\beta\log Z(x)\) 会消掉：

\[
\begin{aligned}
r(x,y_w)-r(x,y_l)
=\beta\Bigg[
&\log\frac{\pi^\star(y_w\mid x)}
{\pi_{\text{ref}}(y_w\mid x)}\\
-&\log\frac{\pi^\star(y_l\mid x)}
{\pi_{\text{ref}}(y_l\mid x)}
\Bigg].
\end{aligned}
\]

现在用要训练的 \(\pi_\theta\) 代替未知最优策略，再代回
Bradley–Terry loss。到这里才给整套目标命名：
**direct preference optimization（直接偏好优化，DPO）**。

把它的 loss 记作 \(\mathcal L_{\text{DPO}}\)：\(\mathcal L\) 表示 loss，
下标 DPO 说明它直接训练 policy：

\[
\mathcal L_{\text{DPO}}(\theta)
=
-\mathbb E_{\mathcal D_{\text{pref}}}
\log\sigma\left(
\beta
\left[
\log\frac{\pi_\theta(y_w\mid x)}
{\pi_{\text{ref}}(y_w\mid x)}
-
\log\frac{\pi_\theta(y_l\mid x)}
{\pi_{\text{ref}}(y_l\mid x)}
\right]
\right).
\]

这不是“凭空发明的对比 loss”。它来自：

\[
\text{KL 正则化最优策略}
\rightarrow
\text{隐式 reward}
\rightarrow
\text{偏好似然}.
\]

## 3.3 DPO 真正在比较什么

定义相对 reference 的序列 log-prob 增益：

\[
m_\theta(x,y)
=
\log\pi_\theta(y\mid x)
-\log\pi_{\text{ref}}(y\mid x).
\]

DPO 希望：

\[
m_\theta(x,y_w)>m_\theta(x,y_l).
\]

也就是说，不只是让 chosen 的绝对概率高，而是让 chosen **相对 reference 的提升幅度**大于 rejected。

## 3.4 \(\beta\) 不只是普通 loss 权重

\(\beta\) 来自原始 KL 正则化强度：

- 较大 \(\beta\)：更强调贴近 reference，对偏好差的响应更保守；
- 较小 \(\beta\)：允许更激进地重排 chosen/rejected 概率。

实际还会与序列 log-prob 聚合、长度分布、学习率相互作用，因此不能脱离实现只凭一句直觉调参。

## 3.5 DPO 与在线 RL 的分界

| 问题 | DPO | reward-model PPO / RLVR |
|---|---|---|
| 数据 | 固定偏好对 | 当前 policy 新 rollout |
| 显式 reward model | 不需要 | 前者需要，RLVR 可用 verifier |
| learned critic | 不需要 | PPO 通常需要 |
| 探索新解 | 受离线覆盖限制 | 可以 |
| 实现复杂度 | 较低 | 较高 |
| 分布漂移适应 | 弱 | 持续获得新样本 |

DPO 使用固定偏好数据，不应仅根据“有没有 critic”与后面将出现的在线分组方法混为一谈；二者的数据来源和目标不同。

# 第四章：有了自动判题以后，回答应该由谁产生

## 4.1 verifier 与 reward model 的根本区别

| | reward model | verifier |
|---|---|---|
| 来源 | 从人类偏好拟合 | 程序规则、测试、执行结果 |
| 输出 | 预测偏好分数 | 可检查的成功/失败或分项得分 |
| 主要风险 | 分布外预测、过优化 | parser/测试漏洞、规格不完整 |
| 适合 | 开放式质量 | 数学、代码、结构化任务 |

verifier 不一定是神谕。它只验证写进规则里的东西。

## 4.2 结果验证与过程验证

若只检查最终答案，就把 verifier 对 prompt \(x\) 和回答 \(y\)
给出的序列 reward 记作 \(R(x,y)\)。符号
\(\mathbf 1[\text{条件}]\) 是指示函数：条件成立取 1，否则取 0。因此：

\[
R(x,y)=\mathbf 1[\text{final}(y)=y^\star],
\]

这是 outcome reward。它可靠但稀疏，不说明中间哪一步正确。

若对中间步骤逐项判断：

\[
r_t=\text{step\_verifier}(x,y_{\le t}),
\]

这是 process reward。信号更密，但高质量逐步标签或过程验证器更难获得，也可能把某种解题风格写死。

工程上常先用可信 outcome verifier 建立基线，再决定是否值得引入过程监督。

## 4.3 verifier 的设计顺序

以 `\boxed{391}` 为例：

1. **解析**：从文本提取 boxed 内容；
2. **规范化**：处理空格、等价表达、数值格式；
3. **判定**：与标准答案比较；
4. **分项记录**：`parse_ok`、`format_ok`、`answer_ok` 分开；
5. **对抗测试**：空输出、多答案、提示注入、超长文本、非法代码。

不要把 parser 失败和答案错误压成一个不可诊断的布尔值。

## 4.4 稀疏 reward 的第一个难题：同题中谁高于正常水平

假设同一 prompt 采样 \(G\) 个回答：

\[
y_1,\ldots,y_G\sim\pi_{\text{old}}(\cdot\mid x),
\]

verifier 给出：

\[
R_1,\ldots,R_G.
\]

我们希望不训练额外 critic，也能构造“相对表现”。最自然的参照就是同题样本的平均分。这就引出下一章的 group-relative 方法。

# 第五章：同一道题采样多次后，怎样构造相对分数

## 5.1 组内标准化不是魔法 value

对同一 prompt 的 \(G\) 个回答，用 \(R_j\) 表示第 \(j\) 个回答的
reward。先算组均值 \(\mu_x\)；再算组标准差 \(\sigma_x\)；最后把第
\(j\) 个回答的相对分数记作 \(A_j\)。\(\varepsilon\) 是防止除零的很小正数。
依次为：

\[
\mu_x=\frac1G\sum_{j=1}^G R_j,
\]

\[
\sigma_x=
\sqrt{
\frac1G\sum_{j=1}^G(R_j-\mu_x)^2
},
\]

\[
A_j=
\frac{R_j-\mu_x}{\sigma_x+\varepsilon}.
\]

\(A_j\) 读作：

> 第 \(j\) 个回答比当前策略在这道题上的同组样本平均水平好多少个标准差。

它是 sample-group baseline，不是 \(V(s_t)\) 的 learned prediction。

### 数值例子：一组四个回答只有一个正确

\[
R=[1,0,0,0].
\]

使用总体标准差：

\[
\mu=0.25,\qquad
\sigma=\sqrt{0.1875}\approx0.433.
\]

于是：

\[
A_{\text{correct}}\approx1.73,
\qquad
A_{\text{wrong}}\approx-0.58.
\]

正确回答被强化，三个错误回答被压低；同一 prompt 内 advantage 和为近似 0。

### 零方差组

若：

\[
R=[0,0,0,0]
\quad\text{或}\quad
R=[1,1,1,1],
\]

则所有 \(R_j-\mu=0\)，组内没有排序信息：

\[
A_j=0.
\]

\(\varepsilon\) 只能防止除零，不能制造学习信号。前者说明题太难或探索不足，后者说明题太简单。

## 5.2 先让每批新回答只更新一次

对 completion \(y_i=(y_{i,1},\ldots,y_{i,T_i})\)，先把它的逐 token
log-prob 取平均。这个平均值记作 \(\bar\ell_i(\theta)\)：横线表示平均，
\(\ell\) 是 log-likelihood 的惯用字母，下标 \(i\) 表示第 \(i\) 条回答：

\[
\bar\ell_i(\theta)
=
\frac1{T_i}
\sum_{t=1}^{T_i}
\log\pi_\theta(y_{i,t}\mid x_i,y_{i,<t}).
\]

这套“同题采样一组回答、计算组内相对分数、每批只更新一次”的方法，本课程称为
**group-relative REINFORCE（GR-REINFORCE）**。把相应的 policy loss 命名为
\(\mathcal L_{\text{GR-REINFORCE}}\)：每个回答的平均 log-prob
由组内相对分数 \(A_i\) 加权，然后取负号以便最小化：

\[
\mathcal L_{\text{GR-REINFORCE}}
=
-\frac1N\sum_{i=1}^{N}A_i\bar\ell_i(\theta).
\]

训练顺序：

1. 当前策略采样一组回答；
2. verifier 打分；
3. 按 prompt 算组内 advantage；
4. 对这批数据做一次 on-policy 更新；
5. 丢弃 rollout，重新采样。

这里同一序列的 \(A_i\) 会广播给 completion 中所有 token。它解决了“相对哪条回答更好”，但没有自动解决“回答内部哪一步推理更好”。

## 5.3 reference KL 的逐 token 采样估计

先定义 sampled token 上 current 与 reference 的 log-prob 差。
课本用大写希腊字母 \(\Delta\)（delta）表示差；这里特意采用
`reference - current` 的方向：

\[
\Delta_t
=
\log\pi_{\text{ref}}(a_t\mid s_t)
-\log\pi_\theta(a_t\mid s_t).
\]

CS285 Homework 4 再把这个差变成逐 token 的非负 KL 估计。
把估计值记作 \(\widehat k_t\)：小写 \(k\) 表示 KL contribution，
帽子表示它是采样估计：

\[
\widehat k_t
=
e^{\Delta_t}-\Delta_t-1.
\]

令 \(x=e^{\Delta_t}>0\)，因为：

\[
x-\log x-1\ge0,
\]

所以每个样本上的 \(\widehat k_t\) 都非负。并且在
\(a_t\sim\pi_\theta\) 下，它的期望等于：

\[
D_{\mathrm{KL}}
\left(
\pi_\theta(\cdot\mid s_t)
\;\|\;
\pi_{\text{ref}}(\cdot\mid s_t)
\right).
\]

相比直接用 \(-\Delta_t\)，这个估计量不会让单个样本出现“负 KL 惩罚”。

加入正则后，把 policy loss 与平均 sampled KL 相加。下面没有新的算法：
\(\mathcal L\) 是总 loss，\(\beta\) 是 KL 项的权重：

\[
\mathcal L
=
\mathcal L_{\text{GR-REINFORCE}}
+\beta\,
\operatorname{masked\_mean}(\widehat k_t).
\]

## 5.4 同一批分组回答怎样有限复用

GR-REINFORCE 更新一次就重采样，稳定但样本利用率低。若要复用同一批 rollout，保存采样时：

\[
\log\pi_{\text{old}}(a_t\mid s_t).
\]

优化时重算 current log-prob，并计算 current/old 概率比。
沿用 PPO 记号把它写成 \(r_{i,t}(\theta)\)，但这里的 \(r\) 表示
ratio，**不是 reward**；实现里应命名为 `ratio`：

\[
r_{i,t}(\theta)
=
\exp\left(
\log\pi_\theta(a_{i,t}\mid s_{i,t})
-\log\pi_{\text{old}}(a_{i,t}\mid s_{i,t})
\right).
\]

把组内相对分数与第一周推导的概率比、clipping 结合，就得到
**group relative policy optimization（GRPO）**。把这一部分 policy loss
命名为 \(\mathcal L_{\text{GRPO,pg}}\)，其中 `pg` 表示 policy-gradient
部分：

\[
\mathcal L_{\text{GRPO,pg}}
=
-
\mathbb E_{i,t}
\left[
\min\left(
r_{i,t}A_i,\;
\operatorname{clip}(r_{i,t},1-\epsilon,1+\epsilon)A_i
\right)
\right].
\]

再加 reference KL，就得到总的
\(\mathcal L_{\text{GRPO}}\)：

\[
\mathcal L_{\text{GRPO}}
=
\mathcal L_{\text{GRPO,pg}}
+\beta\mathcal L_{\text{KL}}.
\]

GRPO 在这套课程里的准确定位是：

> 用同 prompt 的一组回答构造 baseline，省掉 learned critic；再用 PPO 式 ratio 与 clipping 有限复用 rollout。

## 5.5 三个常见混淆

| 混淆 | 正确区分 |
|---|---|
| group advantage = token credit | 它通常是序列级标量，广播到 token |
| GRPO = DPO | GRPO 在线采样并使用 reward；DPO 通常用固定偏好对 |
| 无 critic = 无 baseline | 组内均值本身就是 sample baseline |

# 第六章：按数据和目标选择方法

## 6.1 决策表

| 你的实际条件 | 推荐起点 | 原因 |
|---|---|---|
| 有高质量标准答案，想先建立能力 | SFT | 监督密集、稳定 |
| 有固定 chosen/rejected，没有可靠在线打分 | DPO | 直接利用偏好对 |
| 开放式质量且有大量人类偏好 | reward model + PPO | 可在线优化学到的偏好 |
| 有可靠自动 verifier，当前模型偶尔能成功 | GR-REINFORCE | 最小在线基线，易调试 |
| rollout 贵，希望有限复用 | GRPO | group baseline + PPO 式复用 |
| 当前策略从不成功 | 先 SFT/curriculum | 全零 reward 无学习方向 |

## 6.2 推荐训练阶梯

\[
\text{预训练模型}
\rightarrow
\text{SFT 建立基本格式与能力}
\rightarrow
\text{可信 verifier / 偏好数据}
\rightarrow
\text{最简单可诊断目标}
\rightarrow
\text{必要时加入 rollout 复用}.
\]

对可验证推理任务：

1. 用 SFT 让模型至少偶尔答对；
2. 单独验证 parser 和 reward；
3. 跑 GR-REINFORCE 作为 on-policy 基线；
4. 确认 reward、KL、长度和样例都合理；
5. 再加入 GRPO 的 old log-prob、minibatch 和多个 epoch。

## 6.3 本周完成标准

- [ ] 能从反馈接口判断是示范、偏好还是 verifier；
- [ ] 能从 Bradley–Terry 写出 reward-model loss；
- [ ] 能解释 reward model 分数为何没有绝对零点；
- [ ] 能写出 RLHF 中四个模型的职责；
- [ ] 能从 KL 正则化最优策略推到 DPO loss；
- [ ] 能解释 DPO 为什么依赖离线偏好覆盖；
- [ ] 能手算一组 0/1 reward 的 group advantage；
- [ ] 能解释零方差组为何没有学习信号；
- [ ] 能区分 GR-REINFORCE 与 GRPO；
- [ ] 能说明 group advantage 不等于过程监督。

# 附录：本周官方课件与对应视频

| 正文主题 | 官方课件 | 对应视频 |
|---|---|---|
| 偏好与 reward learning | [CS224R L8 · Reward Learning](https://cs224r.stanford.edu/spring_2025/slides/08_cs224r_reward_learning_2025.pdf) | [CS224R Spring 2025 · Lecture 8](https://www.youtube.com/playlist?list=PLoROMvodv4rPwxE0ONYRa_itZFdaKCylL&index=8) |
| RLHF 与 DPO | [CS224R L9 · Preference Optimization](https://cs224r.stanford.edu/spring_2025/slides/09_cs224r-2025-rlhf.pdf) | [CS224R Spring 2025 · Lecture 9](https://www.youtube.com/playlist?list=PLoROMvodv4rPwxE0ONYRa_itZFdaKCylL&index=9) |
| RLVR 与推理 | [CS224R L10 · RL for Reasoning](https://cs224r.stanford.edu/spring_2025/slides/10_cs224r-rl_for_reasoning_lecture.pdf) | [CS224R Spring 2025 · Lecture 10](https://www.youtube.com/playlist?list=PLoROMvodv4rPwxE0ONYRa_itZFdaKCylL&index=10) |
| LLM policy gradient 总览 | [CS285 L14 · LLM RL](https://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-14.pdf) | [CS285 课程页](https://rail.eecs.berkeley.edu/deeprlcourse/) |
| GR-REINFORCE、GRPO、KL 估计 | [CS285 Homework 4](https://rail.eecs.berkeley.edu/deeprlcourse/static/homeworks/hw4.pdf) | [官方 starter code](https://github.com/berkeleydeeprlcourse/homework_spring2026) |

下一周只做工程实现：张量在哪里对齐、mask 如何定义、old log-prob 何时缓存、reward 与 advantage 如何进入 loss。
