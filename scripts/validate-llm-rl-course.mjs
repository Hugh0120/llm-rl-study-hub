import fs from "node:fs/promises";
import path from "node:path";

const projectDir = path.resolve(import.meta.dirname, "..");
const roots = [path.join(projectDir, "docs")];

const chapters = [
  ["00-roadmap", "这套教程最后要完成什么", "全课程只有一条问题链"],
  ["01-llm-decision-process", "本章符号说明", "本章新增概念"],
  ["02-score-to-gradient", "本章符号说明", "本章新增概念"],
  ["03-value-td-gae", "本章符号说明", "从正常水平得到 advantage"],
  ["04-ppo-and-kl", "本章符号说明", "三个策略先分清"],
  ["05-feedback-and-preference", "本章符号说明", "先按反馈接口"],
  ["06-group-relative-grpo", "本章符号说明", "从题目难度得到 group baseline"],
  ["07-implementation", "本章符号说明", "先固定一条 rollout record"],
  ["08-evaluation", "本章符号说明", "先写清楚你要证明的命题"],
  ["09-tool-agent", "本章符号说明", "整门课程的闭环"],
];

const failures = [];

for (const root of roots) {
  const rootLabel = path.basename(root);
  const indexHtml = await fs.readFile(path.join(root, "index.html"), "utf8");
  const courseIndex = await fs.readFile(
    path.join(root, "course", "index.html"),
    "utf8"
  );

  for (const [slug] of chapters) {
    if (!indexHtml.includes(`course/${slug}.html`)) {
      failures.push(`${rootLabel}/index.html missing ${slug}`);
    }
    if (!courseIndex.includes(`${slug}.html`)) {
      failures.push(`${rootLabel}/course/index.html missing ${slug}`);
    }
  }

  for (const marker of [
    "从一个回答得分",
    "一条问题链",
    "旧版四周讲义归档",
    "@media (prefers-color-scheme: dark)",
  ]) {
    if (!indexHtml.includes(marker)) {
      failures.push(`${rootLabel}/index.html missing marker: ${marker}`);
    }
  }

  for (const [index, [slug, markerA, markerB]] of chapters.entries()) {
    const filename = path.join(root, "course", `${slug}.html`);
    const html = await fs.readFile(filename, "utf8");

    const required = [
      markerA,
      markerB,
      "chapter-toc",
      "本章目录",
      "chapter-nav",
      "page-progress",
      "@media (prefers-color-scheme: dark)",
      "../vendor/katex.min.css",
    ];
    for (const marker of required) {
      if (!html.includes(marker)) {
        failures.push(`${rootLabel}/course/${slug}.html missing: ${marker}`);
      }
    }

    if (index > 0) {
      const previous = chapters[index - 1][0];
      if (!html.includes(`href="${previous}.html"`)) {
        failures.push(`${rootLabel}/course/${slug}.html missing previous link`);
      }
    }
    if (index < chapters.length - 1) {
      const next = chapters[index + 1][0];
      if (!html.includes(`href="${next}.html"`)) {
        failures.push(`${rootLabel}/course/${slug}.html missing next link`);
      }
    }

    for (const bad of [
      "COURSEBLOCKMATH",
      "COURSEINLINEMATH",
      "katex-error",
      "\uFFFD",
    ]) {
      if (html.includes(bad)) {
        failures.push(`${rootLabel}/course/${slug}.html contains ${bad}`);
      }
    }

    const ids = [...html.matchAll(/\sid="([^"]+)"/g)].map(
      (match) => match[1]
    );
    const duplicateIds = ids.filter(
      (id, idIndex) => ids.indexOf(id) !== idIndex
    );
    if (duplicateIds.length > 0) {
      failures.push(
        `${rootLabel}/course/${slug}.html duplicate ids: ${[
          ...new Set(duplicateIds),
        ].join(", ")}`
      );
    }

    const hrefs = [...html.matchAll(/\shref="([^"]+)"/g)].map(
      (match) => match[1]
    );
    for (const href of hrefs) {
      if (/^(https?:|mailto:)/.test(href)) continue;

      const [relativePath, fragment] = href.split("#");
      if (!relativePath && fragment && !ids.includes(fragment)) {
        failures.push(
          `${rootLabel}/course/${slug}.html broken anchor: #${fragment}`
        );
        continue;
      }

      if (relativePath) {
        const target = path.resolve(path.dirname(filename), relativePath);
        try {
          await fs.access(target);
        } catch {
          failures.push(
            `${rootLabel}/course/${slug}.html broken local link: ${href}`
          );
        }
      }
    }
  }
}

const sourceChecks = [
  [
    "02-score-to-gradient.md",
    [
      ["log-derivative trick", "## 2.6"],
      ["REINFORCE", "## 2.10"],
    ],
  ],
  [
    "03-value-td-gae.md",
    [
      ["advantage", "## 3.7"],
      ["TD residual", "## 3.8"],
      ["GAE", "## 3.10"],
      ["Actor–Critic", "## 3.11"],
    ],
  ],
  [
    "04-ppo-and-kl.md",
    [
      ["概率比", "## 4.6"],
      ["PPO clipping", "## 4.9"],
      ["两种 KL", "## 4.11"],
    ],
  ],
  [
    "06-group-relative-grpo.md",
    [
      ["group baseline", "## 6.5"],
      ["GR-REINFORCE", "## 6.9"],
      ["GRPO", "## 6.11"],
    ],
  ],
  [
    "09-tool-agent.md",
    [["Agent", "## 9.3"]],
  ],
];

for (const [file, checks] of sourceChecks) {
  const source = await fs.readFile(
    path.join(projectDir, "content", "course", file),
    "utf8"
  );
  for (const [term, intro] of checks) {
    const introIndex = source.indexOf(intro);
    const contentAfterIntro = source.indexOf(term, introIndex);
    if (introIndex === -1 || contentAfterIntro === -1) {
      failures.push(`${file}: missing ordered introduction ${intro} → ${term}`);
    }
  }
}

if (failures.length > 0) {
  console.error(failures.join("\n"));
  process.exitCode = 1;
} else {
  console.log(
    "Validated new 10-page course: indexes, navigation, math, encoding, dark mode, concept order"
  );
}
