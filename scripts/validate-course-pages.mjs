import fs from "node:fs/promises";
import path from "node:path";

const projectDir = path.resolve(import.meta.dirname, "..");
const docsDir = path.join(projectDir, "docs");

const pages = [
  {
    file: "week1.html",
    required: [
      "第三章：从“整份答案得几分”到“这一步比预期好多少”",
      "本章开头暂时不再给这条规则起新的公式名",
      "表示它是用样本算出的估计值",
      "先构造“正常水平”作为参照",
      "第四章：同一批 rollout 为什么不能随便多训",
    ],
    forbidden: [
      "从绝对回报到 Advantage —— 建立相对表现的参照系",
      "第二章已经得到一个能训练的规则",
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
  if (html.includes("GUIDEBLOCKMATH") || html.includes("GUIDEINLINEMATH")) {
    failures.push(`${page.file}: contains unresolved math placeholder`);
  }
  if (html.includes("katex-error")) {
    failures.push(`${page.file}: contains KaTeX render error`);
  }
  if (html.includes("\uFFFD")) {
    failures.push(`${page.file}: contains replacement character`);
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
