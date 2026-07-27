import fs from "node:fs/promises";
import path from "node:path";

const projectDir = path.resolve(import.meta.dirname, "..");
const docsDir = path.join(projectDir, "docs");

const pages = [
  {
    file: "week1.html",
    required: [
      "从回答得分到稳定更新",
      "第一章：模型怎样自己生成答案并从得分中学习",
      "从一条 LLM 回答如何生成和评分开始",
      "后面反复使用的强化学习基础，全部在第一周成块讲完",
      "具体问题 → 数值例子 → 引入记号 → 推导公式 → 最后命名 → 放回代码",
      "先看一轮训练实际发生了什么",
      "一次生成中，模型究竟做了哪些选择",
      "一整条回答出现的概率从哪里来",
      "训练到底要让哪个数字变大",
      "为什么这不是普通的有标签监督学习",
      "第二章：最终得分怎样改变已生成 token 的概率",
      "不对 token 求导，怎样对产生它的概率求导",
      "把概率梯度改写成 log 概率梯度",
      "log-derivative trick（对数导数技巧）",
      "名称到这里才有用",
      "对当前 LLM 任务，不需要先背一个抽象六元组",
      "这一步不再需要额外的初始状态分布或环境转移符号",
      "第三章：一份答案的分数，怎样变成每个 token 的训练信号",
      "为什么直接使用整份答案的回报太粗",
      "怎样估计“从这个前缀继续，通常能拿多少分”",
      "让当前模型从这里独立生成 10 次",
      "Value 网络实际怎样学",
      "帽子表示这是由本次样本估计出来的量",
      "生成一个片段后，最终得分预期改变了多少",
      "怎样把终局修正传回更早的 token",
      "最终应该使用哪个数作为自己的更新权重",
      "0.20+0.30-0.80=-0.30",
      "这不是凭空出现的新目标",
      "直到这里，再给这条规则命名",
      "代码怎样在一条回答结束时停止递推",
      "第四章：为什么同一批回答训练几次后会失效",
      "第一次参数更新后，什么东西已经变了",
      "怎样用旧回答估计当前模型下的结果",
      "它与直接按 current policy 计算的结果相同",
      "为什么按这个概率比反复训练会失控",
      "怎样停止从旧样本获得额外的便宜收益",
      "到这里，才给这套方法正式命名",
      "限制 sampled token 仍不等于限制整个模型",
      "continue_mask = 1.0 - terminated[t]",
      "本章只需要带走五句话",
    ],
    ordered: [
      "第一章：模型怎样自己生成答案并从得分中学习",
      "第二章：最终得分怎样改变已生成 token 的概率",
      "第三章：一份答案的分数，怎样变成每个 token 的训练信号",
      "第四章：为什么同一批回答训练几次后会失效",
      "第五章：把 value、GAE、PPO 接成一轮可执行训练",
    ],
    forbidden: [
      "第一章：先把 LLM 后训练写成一个 MDP",
      "第二章：REINFORCE —— 让高回报轨迹更可能出现",
      "关键技巧：log-derivative trick",
      "由链式法则得到 log-derivative trick",
      "从 LLM 的 MDP 建模开始",
      "\\mathcal M=(\\mathcal S,\\mathcal A, P, r, \\rho_0, \\gamma)",
      "\\rho_0(s_1)",
      "全篇符号约定：先知道字母是谁，再读公式",
      "假设 value 预测",
      "critic_backbone",
      "第三章：用一条 LLM 回答算出 Value、TD 与 GAE",
      "想计算当前策略下某个量",
      "f(a)",
      "从换分布估计推出概率比",
      "三份模型先分清",
      "一条回答怎样逐步算出 TD",
      "GAE 只做一件事：控制终局修正向前传多远",
      "有两个极端：",
      "最后再看代码：",
      "## 3.11",
      "nonterminal = 1.0 - done[t]",
      "\\mathcal L_{\\text{PG}}",
    ],
    localLinks: ["index.html", "week2.html"],
  },
  {
    file: "week2.html",
    required: [
      "反馈怎样变成训练信号",
      "第二周：没有唯一标准答案时，训练信号从哪里来",
      "先只看反馈，不看算法名",
      "supervised fine-tuning（监督微调，SFT）",
      "reinforcement learning from human feedback（RLHF）",
      "固定偏好对能否直接训练语言模型",
      "偏离参考模型的惩罚强度",
      "direct preference optimization（直接偏好优化，DPO）",
      "有了自动判题以后，回答应该由谁产生",
      "同一道题采样多次后，怎样构造相对分数",
      "先让每批新回答只更新一次",
      "同一批分组回答怎样有限复用",
      "第一章：先分清三种反馈，不要先背算法",
      "第一周已经集中讲完本课程反复使用的 RL 地基",
    ],
    ordered: [
      "第一章：先分清三种反馈，不要先背算法",
      "第二章：偏好如何变成一个可学习的标量",
      "第三章：固定偏好对能否直接训练语言模型",
      "第四章：有了自动判题以后，回答应该由谁产生",
      "第五章：同一道题采样多次后，怎样构造相对分数",
      "第六章：按数据和目标选择方法",
    ],
    forbidden: [
      "第二周：reward 从哪里来——偏好学习、DPO、RLHF 与可验证强化学习",
      "一张因果地图",
      "DPO 是怎样从 KL 正则化目标推出的",
      "从同题比较推出 GR-REINFORCE 与 GRPO",
      "先做最简单的在线更新：GR-REINFORCE",
      "rollout 很贵：再从一步更新走到 GRPO",
    ],
    localLinks: ["index.html", "week1.html", "week3.html"],
  },
  {
    file: "week3.html",
    required: [
      "第三周：把已经推导过的训练规则实现成可靠代码",
      "本周不再引入新的 RL 数学",
      "进入代码前，先把本周依赖的四条接口一次性固定",
      "第一章：先写 tensor 契约，再写一行 loss",
      "哪些 token 位置才应该参与训练",
      "completion mask（回答掩码）",
      "有效 token 的 loss 应该怎样平均",
      "masked mean（掩码平均）",
      "先用一个只检查格式复制的最小任务",
      "smoke test（冒烟测试）",
      "token_mask = completion_mask[:, 1:] * attention_mask[:, 1:]",
      "第八章：用不变量定位故障",
    ],
    ordered: [
      "第一章：先写 tensor 契约，再写一行 loss",
      "第二章：rollout 是一份有生命周期的数据快照",
      "第三章：先把 verifier 做成可信软件",
      "第四章：把 token 概率与 reference KL 算对",
      "第五章：先实现一次更新的 GR-REINFORCE",
      "第六章：确认基线正确后，再加入 GRPO",
      "第七章：按风险排序实现，而不是一次跑完整训练",
      "第八章：用不变量定位故障",
    ],
    forbidden: [
      "第三周：把 LLM 强化学习实现成一条可验证的数据流水线",
      "completion mask 也必须跟着目标右移",
      "先固定 masked mean 的定义",
      "从 format-copy smoke task 开始",
    ],
    localLinks: ["index.html", "week2.html", "week4.html"],
  },
  {
    file: "week4.html",
    required: [
      "第四周：怎样证明训练真的有效，再迁移到会调用工具的模型",
      "本周不再增加新的训练算法",
      "第一章：先定义什么证据能支持“训练有效”",
      "一次回答成功与多次尝试成功是两个问题",
      "条共有多少种选法",
      "为什么最终结果必须重复不同的随机运行",
      "random seed（随机种子）",
      "最终得分是否混入了额外的过程分",
      "reward shaping（奖励塑形）",
      "表示同一批 rollout 被优化的 epoch 数",
      "中间过程满足检查点得到的部分分",
      "第四章：训练分数变高，模型却可能学错目标",
      "现在才给它命名：",
      "怎样在训练前主动攻击判题程序",
      "第六章：从单轮回答扩展到会调用工具的模型",
      "Agent（智能体）",
      "名称没有改变数学结构",
      "最终成功怎样归因到前面的工具动作",
    ],
    ordered: [
      "第一章：先定义什么证据能支持“训练有效”",
      "第二章：在 RL 之前建立可信基线",
      "第三章：用受控消融解释训练机制",
      "第四章：训练分数变高，模型却可能学错目标",
      "第五章：从曲线反推问题发生在哪一层",
      "第六章：从单轮回答扩展到会调用工具的模型",
      "第七章：一页实验报告模板",
      "第八章：整套课程的最终依赖链",
    ],
    forbidden: [
      "第四周：从“reward 涨了”到可信实验，再迁移到 Agent RL",
      "pass@1 与 pass@k 回答不同问题",
      "seed 的职责",
      "消融四：reward shaping",
      "第四章：系统化识别 reward hacking",
      "reward hacking 是规格与行为的错位",
      "红队 verifier",
      "credit assignment 再次出现",
      "第六章：从单轮回答扩展到会调用工具的 Agent",
      "先从熟悉的 LLM MDP 逐项替换",
      "用一个客服 Agent 例子走完整条链",
      "Agent verifier 应验证状态变化，而非自我声明",
      "Agent 实验指标",
    ],
    localLinks: ["index.html", "week3.html"],
  },
];

const failures = [];

for (const page of pages) {
  const html = await fs.readFile(path.join(docsDir, page.file), "utf8");

  for (const text of page.required) {
    if (!html.includes(text)) failures.push(`${page.file}: missing ${text}`);
  }
  for (const text of page.forbidden) {
    if (html.includes(text)) failures.push(`${page.file}: stale text ${text}`);
  }
  let previousIndex = -1;
  for (const milestone of page.ordered ?? []) {
    const index = html.indexOf(milestone);
    if (index === -1) {
      failures.push(`${page.file}: missing ordered milestone ${milestone}`);
    } else if (index <= previousIndex) {
      failures.push(`${page.file}: milestone out of order ${milestone}`);
    }
    previousIndex = index;
  }
  for (const href of page.localLinks) {
    if (!html.includes(`href="${href}"`)) failures.push(`${page.file}: missing link ${href}`);
  }

  if (!html.includes("@media(prefers-color-scheme:dark)")) {
    failures.push(`${page.file}: missing automatic dark mode`);
  }
  if (!html.includes("max-height:calc(100vh - 96px)") || !html.includes("overflow-y:auto")) {
    failures.push(`${page.file}: table of contents is not independently scrollable`);
  }
  if (html.includes("GUIDEBLOCKMATH") || html.includes("GUIDEINLINEMATH")) {
    failures.push(`${page.file}: contains unresolved math placeholder`);
  }
  if (html.includes("katex-error")) {
    failures.push(`${page.file}: contains KaTeX render error`);
  }
  if (html.includes("\uFFFD")) {
    failures.push(`${page.file}: contains replacement character`);
  }
  if (page.file === "week1.html") {
    const firstChapterStart = html.indexOf("<h1>第一章：");
    const secondChapterStart = html.indexOf("<h1>第二章：", firstChapterStart);
    const firstChapterHtml = html.slice(firstChapterStart, secondChapterStart);
    const firstChapterSections = firstChapterHtml.match(/<h2>1\.[1-9]/g) ?? [];
    if (firstChapterSections.length !== 5) {
      failures.push(`week1.html: expected 5 first-chapter sections, got ${firstChapterSections.length}`);
    }
  }
}

const indexHtml = await fs.readFile(path.join(docsDir, "index.html"), "utf8");
for (const href of ["week1.html", "week2.html", "week3.html", "week4.html"]) {
  if (!indexHtml.includes(`href="${href}"`)) failures.push(`index.html: missing ${href}`);
}
if (!indexHtml.includes("NEW COURSEWARE · 2026-07")) {
  failures.push("index.html: missing new-courseware marker");
}
if (!indexHtml.includes("@media(prefers-color-scheme:dark)")) {
  failures.push("index.html: missing automatic dark mode");
}

if (failures.length) {
  console.error(failures.join("\n"));
  process.exitCode = 1;
} else {
  console.log("Validated index + week1..week4: content, math, navigation, encoding, dark mode");
}
