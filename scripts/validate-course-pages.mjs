import fs from "node:fs/promises";
import path from "node:path";

const projectDir = path.resolve(import.meta.dirname, "..");
const docsDir = path.join(projectDir, "docs");

const pages = [
  {
    file: "week1.html",
    required: [
      "第一章：模型怎样自己生成答案并从得分中学习",
      "从一条 LLM 回答如何生成和评分开始",
      "先看一轮训练实际发生了什么",
      "一次生成中，模型究竟做了哪些选择",
      "一整条回答出现的概率从哪里来",
      "训练到底要让哪个数字变大",
      "为什么这不是普通的有标签监督学习",
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
      "第四章：同一批 rollout 为什么不能随便多训",
      "continue_mask = 1.0 - terminated[t]",
      "本章只需要带走五句话",
    ],
    forbidden: [
      "第一章：先把 LLM 后训练写成一个 MDP",
      "从 LLM 的 MDP 建模开始",
      "\\mathcal M=(\\mathcal S,\\mathcal A, P, r, \\rho_0, \\gamma)",
      "\\rho_0(s_1)",
      "全篇符号约定：先知道字母是谁，再读公式",
      "假设 value 预测",
      "critic_backbone",
      "第三章：用一条 LLM 回答算出 Value、TD 与 GAE",
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
      "第一章：先分清三种反馈，不要先背算法",
      "第三章：DPO 是怎样从 KL 正则化目标推出的",
      "第五章：从同题比较推出 GR-REINFORCE 与 GRPO",
    ],
    forbidden: [],
    localLinks: ["index.html", "week1.html", "week3.html"],
  },
  {
    file: "week3.html",
    required: [
      "第一章：先写 tensor 契约，再写一行 loss",
      "token_mask = completion_mask[:, 1:] * attention_mask[:, 1:]",
      "第八章：用不变量定位故障",
    ],
    forbidden: [],
    localLinks: ["index.html", "week2.html", "week4.html"],
  },
  {
    file: "week4.html",
    required: [
      "第一章：先定义什么证据能支持“训练有效”",
      "第四章：系统化识别 reward hacking",
      "第六章：从单轮回答扩展到会调用工具的 Agent",
    ],
    forbidden: [],
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
