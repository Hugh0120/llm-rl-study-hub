# 第三章：从“整份答案得几分”到“这一步比预期好多少”

第二章已经得到一个能训练的动作规则：

> 某一步之后得到的回报高，就提高当时采样 token 的概率；回报低，就降低它的概率。

本章开头暂时不再给这条规则起新的公式名。真正需要解决的问题是：第二章使用的 \(G_t\)（从第 \(t\) 步开始实际得到的后续回报）同时混入了两类信息：

1. 这道题本来有多难；
2. 当前这个 token 选得是否比通常更好。

策略更新真正需要的是第二类信息。本章只解决这一件事。路线是：

> 先构造“正常水平”作为参照 → 再学习这个参照 → 再用相邻前缀之间的预测变化分配信用 → 最后才给已经理解的做法加上课本术语。

## 3.1 为什么绝对回报不是理想的更新权重

假设同一批数据里有两道题：

| prompt | 正确率 | 某次回答 reward |
|---|---:|---:|
| \(x_{\text{easy}}\) | 90% | 1 |
| \(x_{\text{hard}}\) | 5% | 1 |

两次回答都得 1 分，但含义不同：

- 简单题答对接近正常发挥；
- 难题答对远高于正常发挥。

如果都用 \(G=1\) 更新，策略看不见这种差别。因此先做一件不需要新术语的事：用实际回报减去当前前缀的正常水平。

\[
\underbrace{G_t}_{\text{这次实际后续回报}}
-
\underbrace{b(s_t)}_{\text{当前前缀的正常水平}}.
\]

这里的 \(b\) 来自 baseline（参照）；\(b(s_t)\) 表示参照值只由当前状态 \(s_t\) 决定。

先把上面的差值读成一句中文：

> 当前动作之后得到的回报，减去站在当前前缀上原本预计能得到的回报。

这个含义清楚后，再给差值一个方便后文引用的名字：**优势估计**。记作
\(\widehat A_t\)：字母 \(A\) 来自 advantage；帽子 \(\widehat{\phantom A}\)
表示它是用样本算出的估计值；下标 \(t\) 表示第 \(t\) 个生成位置。因此：

\[
\widehat A_t
=
G_t-b(s_t).
\]

于是：

- \(\widehat A_t>0\)：这一步之后的结果好于预期，应提高该 token 的概率；
- \(\widehat A_t<0\)：结果差于预期，应降低该 token 的概率；
- \(\widehat A_t\approx0\)：结果符合预期，更新应很小。

这也解释了为什么 0/1 reward 不能直接读成“错误 token 一定被惩罚”。如果完全不用参照，错误回答的 \(G=0\) 只会让梯度近似为零；有了正的预期分数后，\(0-b(s_t)<0\)，错误回答才会形成明确的负向信号。

## 3.2 减去参照为什么不会把正确梯度改错

我们要求 \(b(s_t)\) 可以依赖状态，但不能偷看当前采样动作。固定一个状态 \(s\)，参照项对梯度的期望为：

\[
\begin{aligned}
\mathbb E_{a\sim\pi_\theta(\cdot\mid s)}
\left[b(s)\nabla_\theta\log\pi_\theta(a\mid s)\right]
&=
b(s)\sum_a \pi_\theta(a\mid s)
\nabla_\theta\log\pi_\theta(a\mid s)\\
&=
b(s)\sum_a\nabla_\theta\pi_\theta(a\mid s)\\
&=
b(s)\nabla_\theta\sum_a\pi_\theta(a\mid s)\\
&=
b(s)\nabla_\theta 1=0.
\end{aligned}
\]

因此：

\[
\mathbb E[(G_t-b(s_t))\nabla\log\pi]
=
\mathbb E[G_t\nabla\log\pi].
\]

参照没有改变期望梯度，只减少了样本之间无关的起伏。这就是它能“降方差而不系统性改方向”的原因。

> 边界：如果参照直接依赖当前动作 \(a_t\)，上面的最后一步通常不成立。那已不再是普通 baseline，需要重新证明。

## 3.3 最自然的参照：站在当前前缀上，平均能得多少分

理想参照不是随便一个常数，而是“当前策略从状态 \(s\) 继续生成时，平均能得到多少后续回报”。这个量叫**状态价值**。

课本用字母 \(V\) 表示 value；右上角的 \(\pi\) 表示“平均结果取决于当前使用的策略 \(\pi\)”。定义为：

\[
V^\pi(s)
=
\mathbb E_{\tau\sim\pi}\left[G_t\mid s_t=s\right].
\]

对 LLM：

- \(s_t\)：prompt 加已经生成的 token；
- \(V^\pi(s_t)\)：从这个前缀继续采样，最终平均能得到多少回报。

如果除了状态，还固定当前选择的动作，就得到**动作价值**。课本约定用字母 \(Q\) 表示它；\(Q\) 只是历史沿用的记号，不是另一个缩写。定义为：

\[
Q^\pi(s,a)
=
\mathbb E_{\tau\sim\pi}\left[G_t\mid s_t=s,a_t=a\right].
\]

动作价值减去状态价值，才正式叫**优势**。既然英文是 advantage，就用字母 \(A\)：

\[
A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s).
\]

现在这个术语不再突兀：**Advantage 就是某动作相对当前状态正常水平的超额表现。**

### 数值例子：同样答对，更新强度不同

假设 value 预测：

\[
V(s_{\text{easy}})=0.9,\qquad V(s_{\text{hard}})=0.05.
\]

两次回答都得 \(G=1\)，则：

\[
\widehat A_{\text{easy}}=1-0.9=0.1,
\qquad
\widehat A_{\text{hard}}=1-0.05=0.95.
\]

难题中的成功轨迹得到更强的强化，正是因为它更“出乎预期”。

## 3.4 value 不知道答案：把它变成一个监督学习问题

真实的 \(V^\pi\) 是未知期望。我们用一个带参数的预测器 \(V_\phi(s)\) 逼近它。

最直接的训练标签，是 rollout 已经发生后算出的实际回报 \(G_t\)。

下面需要第一次给一个 loss 命名：\(\mathcal L\) 表示“训练时要最小化的损失”，下标 \(V\) 表示这是 value 的回归损失。因此 \(\mathcal L_V(\phi)\) 读作“参数为 \(\phi\) 的 value loss”：

\[
\mathcal L_V(\phi)
=
\mathbb E_t\left[
\left(V_\phi(s_t)-G_t\right)^2
\right].
\]

这就是普通回归：

- 输入：当前前缀 \(s_t\)；
- 标签：从这里往后实际拿到的 \(G_t\)；
- 输出：预测的期望回报。

到这里才需要两个角色名：

- **Actor**：产生动作的策略 \(\pi_\theta(a\mid s)\)；
- **Critic**：评估状态的预测器 \(V_\phi(s)\)。

所谓 **Actor–Critic** 没有多出一套神秘算法，它只是：

1. critic 学习“正常水平”；
2. actor 根据“实际表现减正常水平”更新。

二者的 loss 不同，参数也可以不同。下面的
\(\mathcal L_{\text{actor}}\) 和 \(\mathcal L_{\text{critic}}\)
只是两个 loss 的名称，下标说明它分别训练哪一个角色。
\(\widehat V_t^{\text{target}}\) 暂时表示“critic 应该拟合的 value
训练目标”；它的具体构造会从下一节开始逐步推出：

\[
\mathcal L_{\text{actor}}
=
-\mathbb E_t[\widehat A_t\log\pi_\theta(a_t\mid s_t)],
\]

\[
\mathcal L_{\text{critic}}
=
\mathbb E_t[(V_\phi(s_t)-\widehat V_t^{\text{target}})^2].
\]

## 3.5 不等整条轨迹结束：先推出一步预测误差

用完整 \(G_t\) 训练 value 是无偏的，但长序列里方差大，而且必须等终局。我们希望只看一步：

> 当前状态的价值，应该等于这一步奖励，加上下一个状态的价值。

这就是 Bellman 一致性：

\[
V^\pi(s_t)
=
\mathbb E\left[r_t+\gamma V^\pi(s_{t+1})\mid s_t\right].
\]

把期望换成这次真实转移，并把真实 value 换成预测器，就得到一个只向前看一步的训练目标。

把它记作 \(\widehat V_t^{(1)}\)：帽子表示估计目标，上标 \((1)\) 表示只使用一步真实转移，下标 \(t\) 表示它属于状态 \(s_t\)：

\[
\widehat V_t^{(1)}
=
r_t+\gamma V_\phi(s_{t+1}).
\]

接着计算“这次一步目标减去更新前的当前预测”：

\[
\underbrace{
r_t+\gamma V_\phi(s_{t+1})-V_\phi(s_t)
}_{\text{走完一步后，价值预期需要修正多少}}.
\]

> 走完这一步以后，新的信息让我们把最终回报预期上调了多少或下调了多少。

现在含义已经明确，再给它命名：这个量叫
**temporal-difference error，TD 误差**。课本用希腊字母
\(\delta\)（delta，常用来表示“差”），所以下标为第 \(t\) 步的 TD 误差写作：

\[
\delta_t
\equiv
r_t+\gamma V_\phi(s_{t+1})-V_\phi(s_t).
\]

`temporal difference` 指相邻时间步价值预测之间的差，不是另一种 reward。

### 数值例子：一个 token 如何获得局部信号

设 \(\gamma=1\)，生成某 token 前：

\[
V(s_t)=0.45.
\]

生成后还未结束，\(r_t=0\)，但新前缀更像正确解，value 变为：

\[
V(s_{t+1})=0.60.
\]

于是：

\[
\delta_t=0+0.60-0.45=0.15.
\]

这个 token 得到正信号，不是因为它立刻拿到了 reward，而是因为它让“最终成功概率”的预测上升了。

若最后输出错误，终止状态的后续价值记为 0，且 \(r_T=0\)。假设结束前 \(V(s_T)=0.55\)，则：

\[
\delta_T=0+0-0.55=-0.55.
\]

终局失败会把先前过于乐观的预测拉回去。

## 3.6 一步很稳但有偏，整段很准但很抖

一步目标依赖 \(V_\phi(s_{t+1})\)。预测器尚不准确时，它会把自己的误差带回训练，这叫 **bootstrapping（自举）**。

另一个极端是一直看到终局。它叫 Monte Carlo target，缩写 MC。
把第 \(t\) 步的这个估计目标记作
\(\widehat V_t^{(\text{MC})}\)：帽子表示估计，括号中的 MC 说明它一直使用真实 reward 到终局：

\[
\widehat V_t^{(\text{MC})}
=
r_t+\gamma r_{t+1}+\cdots+\gamma^{T-t}r_T.
\]

它不依赖未来 value 预测，偏差小，但不同 rollout 的结果波动大。

中间方案是 \(n\)-step target：先使用 \(n\) 步真实 reward，再从第
\(t+n\) 个状态的 value 预测自举。相应目标记作
\(\widehat V_t^{(n)}\)：

\[
\widehat V_t^{(n)}
=
\sum_{l=0}^{n-1}\gamma^l r_{t+l}
+\gamma^nV_\phi(s_{t+n}).
\]

| 目标 | 看多远 | 主要优点 | 主要代价 |
|---|---:|---|---|
| 1-step | 1 步 | 方差较小、更新及时 | 更依赖 critic，偏差可能大 |
| \(n\)-step | \(n\) 步 | 可调折中 | 多一个尺度选择 |
| Monte Carlo | 到终局 | 不从未来 value 自举 | 长轨迹方差大 |

这一步先建立“不同视野长度”的概念，下一节才组合它们。

## 3.7 GAE：把不同视野的 TD 信息做指数加权

我们已经有每一步的预测修正 \(\delta_t\)。可以把未来若干步的修正累加。下面
\(\widehat A_t^{(1)}\)、\(\widehat A_t^{(2)}\) 和
\(\widehat A_t^{(3)}\) 的上标表示分别累计 1、2、3 个 TD 修正项：

\[
\widehat A_t^{(1)}=\delta_t,
\]

\[
\widehat A_t^{(2)}=\delta_t+\gamma\delta_{t+1},
\]

\[
\widehat A_t^{(3)}=\delta_t+\gamma\delta_{t+1}
+\gamma^2\delta_{t+2}.
\]

希望近处权重大、远处逐渐衰减，就得到 GAE。
\(\lambda\in[0,1]\) 是人为选择的衰减系数；每向未来多走一步，权重就再乘一次
\(\gamma\lambda\)。因此：

\[
\widehat A_t^{\text{GAE}(\gamma,\lambda)}
=
\sum_{l=0}^{T-t}
(\gamma\lambda)^l\delta_{t+l}.
\]

把公式写成代码之前，先处理一条轨迹的边界。对第 \(t\) 步，先定义：

- `terminated[t] = 0`：执行 \(a_t\) 后轨迹还没有真正结束，\(s_{t+1}\) 仍属于同一条轨迹；
- `terminated[t] = 1`：执行 \(a_t\) 后到达真正的终止状态，例如生成 EOS、任务成功或明确失败。

代码需要一个开关，决定能不能继续使用下一时刻的信息。这里把它直接命名为
`continue_mask`：

\[
\texttt{continue\_mask}_t=1-\texttt{terminated}_t.
\]

| 当前这一步之后 | `terminated[t]` | `continue_mask` | 递推行为 |
|---|---:|---:|---|
| 同一条轨迹还在继续 | 0 | 1 | 保留 \(V(s_{t+1})\) 和后续 GAE |
| 轨迹在这里真正结束 | 1 | 0 | 同时切断二者 |

它叫 `mask`，只是因为乘以 0 会屏蔽不该跨越边界传播的信息；它不是一个新的强化学习量。于是反向递推可以写成：

```python
gae = 0.0
for t in reversed(range(T)):
    continue_mask = 1.0 - terminated[t]

    delta = (
        reward[t]
        + gamma * value[t + 1] * continue_mask
        - value[t]
    )
    gae = (
        delta
        + gamma * lam * continue_mask * gae
    )
    advantage[t] = gae

value_target = advantage + value[:-1]
```

代入两个取值就能看清它的作用：

- **尚未结束**：`continue_mask = 1`，所以
  \(\delta_t=r_t+\gamma V(s_{t+1})-V(s_t)\)，后续的 GAE 也会继续向前传播；
- **真正终止**：`continue_mask = 0`，所以
  \(\delta_t=r_t-V(s_t)\)，并且 \(\widehat A_t=\delta_t\)。下一条轨迹的信息不会串进来。

\(\lambda\) 控制“相信一步 critic”还是“更多相信真实后续”：

- \(\lambda=0\)：退化为一步 TD，通常方差低、偏差高；
- \(\lambda\to1\)：接近长视野回报，通常偏差低、方差高。

还要区分**真正终止**和**达到最大长度而被截断**。真正终止时没有合法的后续，必须令
`continue_mask = 0`。达到最大长度只是采样器停止收集；任务本身可能仍能继续。如果保留了最后一个真实观测并希望 critic 估计其后续，就应继续 bootstrap，而不能因为“长度到了”自动当成真正终止。工程代码通常分别保存 `terminated` 和 `truncated`，避免一个含义模糊的 `done` 把两种情况混在一起。补齐 batch 的 `padding mask` 又是另一件事，它只负责排除填充 token。

## 3.8 只有终局 reward 时，GAE 到底做了什么

数学推理常见：

\[
r_t=0\quad(t<T),\qquad r_T=R.
\]

这不意味着所有 token 天生拥有精确的过程监督。GAE 的局部差异来自 critic：

- 某个前缀让成功概率预测上升，附近 \(\delta_t\) 为正；
- 某个前缀暴露错误，让预测下降，附近 \(\delta_t\) 为负；
- 终局 reward 再修正整条预测链。

所以必须说清边界：

1. GAE 能把**学到的前缀价值变化**传播成 token 级权重；
2. 它不会凭空知道哪一步逻辑是对的；
3. critic 不准时，细粒度 credit assignment 也不准；
4. 如果不训练 critic，而只把同一序列 reward 广播给所有 completion token，那仍是序列级监督。

第二周会介绍另一条路线：对同一 prompt 采样一组回答，用组内均值代替 learned value baseline。那是 GRPO/GR-REINFORCE 的出发点，不要提前与本章混为一谈。

### 第三章检查点

读完应能不看公式回答：

1. baseline 为什么能产生负向更新，却不改变期望梯度？
2. value 是什么量的条件期望？
3. critic 的训练标签从哪里来？
4. TD error 为什么可以读成“预期修正”？
5. GAE 中 \(\lambda\) 在折中什么？
6. 终局 reward 为什么不等于真实过程监督？

# 第四章：同一批 rollout 为什么不能随便多训

第三章改善了每个动作的权重，却仍有一个数据问题。

第二章和第三章的推导都默认：

\[
a_t\sim\pi_\theta(\cdot\mid s_t).
\]

但工程中，rollout 很贵。我们通常先冻结一份策略采样，再对这批数据做多个 minibatch、多个 epoch。第一次更新后，当前策略已经变了，数据却仍来自旧策略。问题就变成：

> 怎样有限度地复用旧数据，同时不让一次更新把策略推得太远？

这才是 PPO 要回答的问题。

## 4.1 三份模型先分清

LLM PPO 常同时出现三份概率，不分清就一定会把 KL 和 ratio 写混：

| 名称 | 记号 | 是否更新 | 用途 |
|---|---|---:|---|
| old policy | \(\pi_{\text{old}}\) | 一轮 rollout 内冻结 | 产生数据，提供旧 log-prob |
| current policy | \(\pi_\theta\) | 是 | 本轮正在优化 |
| reference policy | \(\pi_{\text{ref}}\) | 通常冻结 | 约束模型别偏离 SFT 起点太远 |

`old` 是短期快照，每轮会刷新；`reference` 是长期行为锚点。二者偶尔参数相同，也不代表概念相同。

## 4.2 从换分布估计推出概率比

想计算当前策略下某个量 \(f(a)\) 的期望，却只有旧策略样本：

\[
\begin{aligned}
\mathbb E_{a\sim\pi_\theta}[f(a)]
&=
\sum_a\pi_\theta(a\mid s)f(a)\\
&=
\sum_a\pi_{\text{old}}(a\mid s)
\frac{\pi_\theta(a\mid s)}
{\pi_{\text{old}}(a\mid s)}
f(a).
\end{aligned}
\]

沿用 PPO 文献的习惯，把概率比记作 \(r_t(\theta)\)。这里的字母
\(r\) 表示 ratio，**不是前文的环境 reward \(r_t\)**；代码里建议直接命名为
`ratio_t`，避免混淆。定义为：

\[
r_t(\theta)
=
\frac{\pi_\theta(a_t\mid s_t)}
{\pi_{\text{old}}(a_t\mid s_t)}
=
\exp\left(
\log\pi_\theta(a_t\mid s_t)
-\log\pi_{\text{old}}(a_t\mid s_t)
\right).
\]

于是旧样本上的策略目标可写成一个“替代目标”。
\(\mathcal L\) 表示优化目标，下标 `surrogate` 表示它是在旧样本上替代当前策略真实目标的近似名称：

\[
\mathcal L_{\text{surrogate}}(\theta)
=
\mathbb E_t[r_t(\theta)\widehat A_t].
\]

直觉：

- \(r_t=1\)：当前策略对该 token 的概率没变；
- \(r_t=1.2\)：概率变为旧策略的 1.2 倍；
- \(r_t=0.7\)：概率变为旧策略的 0.7 倍。

> 限制：这个比值修正了给定旧状态 \(s_t\) 时的动作概率，没有完整修正更新后策略访问到的状态分布。策略变化太大时，旧前缀本身也会失去代表性。因此还需要限制步长。

## 4.3 为什么直接优化概率比会失控

若 \(\widehat A_t>0\)，最大化 \(r_t\widehat A_t\) 会持续增大 \(r_t\)；若 \(\widehat A_t<0\)，会持续减小 \(r_t\)。在同一批数据上训练很多次，模型可能极端放大少量偶然成功样本。

PPO 的核心是：超出一个局部区间后，不再从这批旧样本获得额外“便宜收益”。

把这个带截断的目标命名为
\(\mathcal L_{\text{clip}}\)：\(\mathcal L\) 表示目标，`clip` 表示使用截断规则。
\(\operatorname{clip}(r,1-\epsilon,1+\epsilon)\) 的意思是把 \(r\)
限制到区间内：低于下界就取下界，高于上界就取上界；外层
\(\min\) 再从原目标和截断目标中取较小者。公式是：

\[
\mathcal L_{\text{clip}}(\theta)
=
\mathbb E_t\left[
\min\left(
r_t\widehat A_t,\;
\operatorname{clip}(r_t,1-\epsilon,1+\epsilon)\widehat A_t
\right)
\right].
\]

训练代码通常最小化负号：

\[
\mathcal L_{\text{actor}}=-\mathcal L_{\text{clip}}.
\]

## 4.4 分正负两种情况读 clip

设 \(\epsilon=0.2\)。

### 好动作：\(\widehat A=2\)

若 \(r=1.35\)：

\[
r\widehat A=2.7,\qquad
\operatorname{clip}(r,0.8,1.2)\widehat A=2.4.
\]

取较小值 2.4。概率已经提高很多后，继续提高不会改善 clipped objective。

### 坏动作：\(\widehat A=-2\)

若 \(r=0.70\)：

\[
r\widehat A=-1.4,\qquad
\operatorname{clip}(r,0.8,1.2)\widehat A=-1.6.
\]

取较小值 \(-1.6\)。概率已经降低很多后，继续降低也不会得到额外好处。

| advantage | 策略想做什么 | 哪一侧被截断 |
|---|---|---|
| \(\widehat A>0\) | 提高该动作概率 | \(r>1+\epsilon\) |
| \(\widehat A<0\) | 降低该动作概率 | \(r<1-\epsilon\) |

PPO 不是把所有 ratio 都硬裁成区间，而是用 `min` 构造悲观目标。区间另一侧仍会保留惩罚性梯度。

## 4.5 clip 不等于真正的距离保证

clip 只作用于样本中的动作概率比，不保证整个词表分布的 KL 一定小。因此至少监控 KL divergence。

课本把它写成 \(D_{\mathrm{KL}}(P\|Q)\)：\(D\) 表示 divergence，
下标 KL 是 Kullback–Leibler，竖线两边的顺序有意义。下面第一个式子比较
old policy 与 current policy：

\[
D_{\mathrm{KL}}
\left(
\pi_{\text{old}}(\cdot\mid s)
\;\|\;
\pi_\theta(\cdot\mid s)
\right).
\]

它回答：“本轮 current 离 rollout 快照走了多远？”

另一个常见 KL 是：

\[
D_{\mathrm{KL}}
\left(
\pi_\theta(\cdot\mid s)
\;\|\;
\pi_{\text{ref}}(\cdot\mid s)
\right).
\]

它回答：“训练后的模型离长期 reference 走了多远？”

| KL | 角色 | 典型用途 |
|---|---|---|
| old ↔ current | 优化稳定性 | early stop、调小 epoch/学习率 |
| current ↔ reference | 行为锚定 | 奖励塑形或额外正则 |

不要只写“监控 KL”而不说明是哪两份策略。

<details>
<summary>可选：TRPO 与 natural gradient 在这条故事中的位置</summary>

TRPO 直接提出一个带 KL 约束的优化问题，在局部二阶近似下得到 natural-gradient 风格的方向。PPO 用一阶优化和 clipping/KL penalty 给出更容易实现的近似。对本周主线，只需记住：TRPO 更显式地约束分布距离，PPO 更工程化，但 clip 不是数学上的硬信赖域。

</details>

### 第四章检查点

1. 为什么第一次更新后 rollout 就变成旧数据？
2. old policy 与 reference policy 的生命周期有什么不同？
3. ratio 从哪个换分布等式得到？
4. 为什么正 advantage 和负 advantage 的截断方向相反？
5. PPO clip 为什么不能代替 KL 监控？

# 第五章：把 value、GAE、PPO 接成一轮可执行训练

现在所有组件都有了来由。本章只做一件事：明确数据在什么时候由谁产生，以及哪些量必须冻结。

## 5.1 一轮训练的因果顺序

### 步骤一：冻结 rollout 快照

箭头 \(\leftarrow\) 表示“把右边当前策略的参数复制给左边的 old
policy”，不是求导，也不是让两份模型以后始终同步：

\[
\pi_{\text{old}}\leftarrow\pi_\theta.
\]

记录每个采样 completion token 的：

- `token_id`
- `old_logp`
- `value`
- `completion_mask`

### 步骤二：生成并打分

对 prompt \(x_i\)：

\[
y_i\sim\pi_{\text{old}}(\cdot\mid x_i),
\qquad
R_i=\text{reward}(x_i,y_i).
\]

reward 可以来自人工、reward model 或 verifier。它负责评价结果，不负责预测未来。

### 步骤三：构造逐步 reward

终局任务最简单的版本：

\[
r_{i,t}=
\begin{cases}
0,&t<T_i,\\
R_i,&t=T_i.
\end{cases}
\]

若使用 reference KL 约束，可选两条实现路径之一：

**路径 A：KL 进入 shaped reward**

\(k_{i,t}\) 表示第 \(i\) 条回答、第 \(t\) 个 token 相对 reference
policy 的偏离代价；\(\beta\) 控制这项代价的权重。于是任务 reward
减去偏离代价后得到 shaped reward：

\[
r^{\text{shaped}}_{i,t}
=
r^{\text{task}}_{i,t}
-\beta\,k_{i,t}.
\]

随后用 shaped reward 计算 GAE。

**路径 B：KL 作为单独 loss**

下面公式中的三个 \(\mathcal L\) 分别表示 actor loss、value loss 和 KL loss；
\(c_v\) 是 value loss 的系数，\(\beta\) 是 KL loss 的系数。它们只负责调节三部分在总 loss 中的相对尺度：

\[
\mathcal L
=
\mathcal L_{\text{actor}}
+c_v\mathcal L_{\text{value}}
+\beta\mathcal L_{\text{KL}}.
\]

两条路径的尺度和梯度语义不同。除非你明确推导过，不要同时加两次 KL。

### 步骤四：冻结 advantage 与 old log-prob

用 rollout 阶段保存的 value 和 reward 计算两个量：

- \(\widehat A_t\)：给 actor 使用的优势估计；
- \(\widehat R_t\)：给 critic 回归的 value target。这里的帽子表示估计目标；它不是环境给出的原始 reward \(R_i\)。

\[
\widehat A_t=\operatorname{GAE}(r_t,V_{\text{old}}(s_t)),
\qquad
\widehat R_t=\widehat A_t+V_{\text{old}}(s_t).
\]

进入 PPO epoch 后，`advantage`、`value_target`、`old_logp` 都作为常量使用，不能随着每个 minibatch 重新反向传播。

### 步骤五：有限复用 rollout

```python
for epoch in range(K):
    for mb in minibatches(rollout):
        new_logp = policy.logprob(mb.tokens)
        ratio = exp(new_logp - mb.old_logp)

        unclipped = ratio * mb.advantage
        clipped = clamp(ratio, 1-eps, 1+eps) * mb.advantage
        actor_loss = -masked_mean(min(unclipped, clipped), mb.mask)

        new_value = critic(mb.states)
        value_loss = masked_mean(
            (new_value - mb.value_target) ** 2,
            mb.mask,
        )

        loss = actor_loss + value_coef * value_loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

### 步骤六：丢弃旧数据，重新采样

完成 \(K\) 个 epoch 后，这批 rollout 的复用到此为止。下一轮必须：

\[
\pi_{\text{old}}\leftarrow\pi_\theta
\]

并生成新数据。PPO 是**受限制的短期复用**，不是把 rollout 变成永久离线数据集。

## 5.2 五个模型头分别回答什么

| 组件 | 输入 | 输出 | 训练/冻结 | 回答的问题 |
|---|---|---|---|---|
| current policy | 前缀 | 下个 token 分布 | 训练 | 接下来生成什么？ |
| old policy | 前缀 | 旧 token 概率 | 一轮冻结 | 数据由谁生成？ |
| reference policy | 前缀 | 参考 token 概率 | 长期冻结 | 偏离起点多远？ |
| value/critic | 前缀 | 期望未来回报 | 训练 | 从这里通常能得多少分？ |
| reward/verifier | 完整输入输出 | 标量或规则结果 | 通常固定 | 这份结果有多好？ |

reward model 和 value model 都输出标量，但条件、目标和生命周期不同，不能互换。

## 5.3 贯穿例子：乘法题的一轮 PPO

prompt：`23 × 17 = ?`

old policy 生成两条：

1. `391`，verifier 给 \(R=1\)；
2. `381`，verifier 给 \(R=0\)。

critic 认为生成首位数字前成功率约为 0.35：

- 正确轨迹最终会产生正的 value 修正；
- 错误轨迹终局会产生负的 value 修正；
- GAE 把这些修正按距离传播回各 token；
- PPO 用 old/new ratio 限制同一批样本的复用强度；
- reference KL 防止模型为了这类题而整体语言分布漂移过远。

这条链里没有哪个组件能独自完成全部工作：

\[
\text{verifier 定义结果}
\rightarrow
\text{critic 估计预期}
\rightarrow
\text{GAE 构造相对权重}
\rightarrow
\text{PPO 限制更新}
\rightarrow
\text{新 policy 产生下一轮数据}.
\]

## 5.4 训练日志应该对应哪一段机制

| 现象 | 先看 | 机制解释 |
|---|---|---|
| reward 不动，entropy 很快塌缩 | 采样多样性、学习率 | 还没探索到成功样本就变确定 |
| clip fraction 长期很高 | epoch、学习率、\(\epsilon\) | 旧数据被推得过远 |
| old-current KL 暴涨 | 更新步数、batch | rollout 很快失效 |
| current-reference KL 暴涨 | \(\beta\)、reward 漏洞 | 策略远离行为锚点 |
| value loss 下降但 reward 不升 | value target、策略梯度 | critic 会拟合数据不代表 actor 变好 |
| explained variance 很差 | mask、终止、自举 | critic 没学会回报结构 |
| 长回答占优 | token sum/mean、长度奖励 | loss 聚合方式引入长度偏差 |

# 第六章：把整周压成一张依赖图

## 6.1 每个概念是为了解决哪个前一个问题

| 已有方法 | 暴露的问题 | 下一概念 |
|---|---|---|
| 轨迹 reward | 不可直接对离散采样求导 | log-derivative trick |
| REINFORCE | 绝对回报噪声大 | baseline |
| baseline | 理想参照未知 | value/critic |
| Monte Carlo value | 长轨迹方差大、反馈晚 | Bellman 自举与 TD |
| 1-step TD | critic 偏差重 | n-step 与 GAE |
| on-policy actor–critic | rollout 贵，只训一次浪费 | importance ratio |
| 直接复用旧数据 | ratio 失控、状态分布漂移 | PPO clip / KL |
| PPO 任务优化 | 可能远离 SFT 行为 | reference KL |

如果一个术语不能落回这张表中的“问题”，说明它还没有真正理解。

## 6.2 六个必须会手算的量

给定：

\[
\log\pi_\theta=-1.2,\quad
\log\pi_{\text{old}}=-1.4,\quad
\widehat A=-0.5,\quad
\epsilon=0.2,
\]

应能算出：

1. \(r=\exp(-1.2+1.4)\)；
2. unclipped objective \(r\widehat A\)；
3. clipped ratio；
4. clipped objective；
5. 两者的 `min`；
6. 最小化代码中的 actor loss 符号。

再给：

\[
r_t=0,\quad \gamma=1,\quad
V(s_t)=0.3,\quad V(s_{t+1})=0.5,
\]

应能算出 \(\delta_t=0.2\)，并解释它为什么不是环境 reward。

## 6.3 本周完成标准

- [ ] 能从期望回报独立推到 REINFORCE；
- [ ] 能证明 state-only baseline 不改变期望梯度；
- [ ] 能用中文区分 return、value、Q、advantage、TD error；
- [ ] 能从一步预测误差推出 GAE 递推；
- [ ] 能解释 old/current/reference 三份策略；
- [ ] 能分 advantage 正负读懂 PPO clip；
- [ ] 能画出 rollout → reward → GAE → PPO → resample；
- [ ] 能说明终局 reward 下 token credit 的能力边界；
- [ ] 能指出 KL 是哪两份模型之间的 KL；
- [ ] 能从日志定位是采样、critic、PPO 还是 reward 出问题。

# 附录：官方课件与对应视频

正文已经自包含。下面只用于复习原始讲授和核对公式。

| 本文位置 | 官方课件 | 对应视频 |
|---|---|---|
| 第一章：MDP、轨迹、回报 | [CS285 L4 · RL Basics](https://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-4.pdf) | [CS285 Fall 2023 · Lecture 4](https://www.youtube.com/playlist?list=PL_iWQOsE6TfVYGEGiAOMaOzzv41Jfm_Ps&index=4) |
| 第二章：REINFORCE | [CS285 L5 · Policy Gradients](https://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-5.pdf) | [CS285 Fall 2023 · Lecture 5](https://www.youtube.com/playlist?list=PL_iWQOsE6TfVYGEGiAOMaOzzv41Jfm_Ps&index=5) |
| 第三章：value、TD、GAE | [CS285 L6 · Actor–Critic](https://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-6.pdf) | [CS285 Fall 2023 · Lecture 6](https://www.youtube.com/playlist?list=PL_iWQOsE6TfVYGEGiAOMaOzzv41Jfm_Ps&index=6) |
| 第四章：换分布估计 | [CS285 L9 · Advanced Policy Gradients I](https://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-9.pdf) | [CS285 Fall 2023 · Lecture 9](https://www.youtube.com/playlist?list=PL_iWQOsE6TfVYGEGiAOMaOzzv41Jfm_Ps&index=9) |
| 第四章：约束、PPO、TRPO | [CS285 L10 · Advanced Policy Gradients II](https://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-10.pdf) | [课程页中的最新录像入口](https://rail.eecs.berkeley.edu/deeprlcourse/) |

下一周不再重复 PPO 推导，而是回答一个新问题：**reward 从哪里来，以及偏好数据、自动 verifier 和组内比较分别适合什么场景。**
