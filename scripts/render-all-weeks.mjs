import fs from "node:fs/promises";
import path from "node:path";
import katex from "katex";

const markedModule = process.env.MARKED_MODULE;
if (!markedModule) throw new Error("MARKED_MODULE is required");
const { marked } = await import(markedModule);

const projectDir = path.resolve(import.meta.dirname, "..");
const pages = [
  {
    slug: "week1",
    prefixReplacementSource: path.join(projectDir, "content", "week1-opening-ch1.md"),
    prefixReplaceTo: "# 第二章",
    replacementSource: path.join(projectDir, "content", "week1-ch3-onward.md"),
    replaceFrom: "\n# 第三章",
    subsectionReplacementSource: path.join(projectDir, "content", "week1-ch2-step3.md"),
    subsectionReplaceFrom: "### 第三步：把“整条轨迹的 log 概率”拆成每一步",
    subsectionReplaceTo: "## 2.2",
    sectionReplacements: [
      {
        source: path.join(projectDir, "content", "week1-ch3.md"),
        from: "# 第三章",
        to: "# 第四章",
      },
      {
        source: path.join(projectDir, "content", "week1-ch4.md"),
        from: "# 第四章",
        to: "# 第五章",
      },
    ],
    textReplacements: [
      {
        from: "# 第二章：REINFORCE —— 让高回报轨迹更可能出现",
        to: "# 第二章：最终得分怎样改变已生成 token 的概率",
      },
      {
        from: "## 2.1 关键技巧：log-derivative trick",
        to: "## 2.1 不对 token 求导，怎样对产生它的概率求导",
      },
      {
        from: "### 第二步：由链式法则得到 log-derivative trick",
        to: "### 第二步：把概率梯度改写成 log 概率梯度",
      },
      {
        from: "这不是一个额外假设，只是 \\(\\nabla\\log p=(1/p)\\nabla p\\) 的移项。将它代回上一步：",
        to: "这不是一个额外假设，只是 \\(\\nabla\\log p=(1/p)\\nabla p\\) 的移项。\n这条把概率梯度改写成 log 概率梯度的恒等变换，常叫 **log-derivative trick（对数导数技巧）**。将它代回上一步：",
      },
      {
        from: "## 2.2 一条 rollout 如何变成可训练的 loss",
        to: "这套“用采样回报加权已采样动作的 log-prob 梯度”的方法叫 **REINFORCE**。名称到这里才有用：它只是上面整条推导的简称，不是一条额外假设。\n\n## 2.2 一条 rollout 如何变成可训练的 loss",
      },
    ],
    termOrderRules: [
      {
        term: "log-derivative trick",
        introduction: "log-derivative trick（对数导数技巧）",
      },
      {
        term: "REINFORCE",
        introduction: "的方法叫 **REINFORCE**",
      },
      {
        term: "baseline",
        introduction: "叫 **baseline（基线）**",
      },
      {
        term: "TD 误差",
        introduction: "**temporal-difference error（TD 误差）**",
      },
      {
        term: "GAE",
        introduction: "简称 **GAE**",
      },
      {
        term: "PPO",
        introduction: "**PPO（Proximal Policy Optimization）**",
      },
    ],
    source: path.resolve(projectDir, "..", "第一周-CS285-策略优化与PPO-中文精读.md"),
    label: "WEEK 01 · SELF-CONTAINED EDITION",
    title: "从回答得分到稳定更新",
    description: "从一条 LLM 回答如何生成和评分开始，逐步建立按结果更新 token 概率、估计相对表现并限制更新幅度的完整训练逻辑。",
    meta: "面向算法工程师的第一周自包含教材：建立后续课程反复使用的强化学习地基。",
    officialUrl: "https://rail.eecs.berkeley.edu/deeprlcourse/",
    officialLabel: "CS285 官网 ↗",
    previous: null,
    next: "week2.html",
    note: "按章节顺序阅读：第一章从一次 LLM 生成建立状态、动作、回答概率和平均回报；第二章直接接着解决最终分数如何改变 token 概率。",
  },
  {
    slug: "week2",
    source: path.join(projectDir, "content", "week2.md"),
    label: "WEEK 02 · SELF-CONTAINED EDITION",
    title: "反馈怎样变成训练信号",
    description: "从示范、两两偏好和自动判题三种反馈出发，逐步判断何时模仿、何时学习评分器、何时让当前模型在线探索。",
    meta: "第二周自包含教材：从反馈接口出发推导固定数据训练、学习评分器与在线探索。",
    officialUrl: "https://cs224r.stanford.edu/",
    officialLabel: "CS224R 官网 ↗",
    previous: "week1.html",
    next: "week3.html",
    note: "先判断手中反馈能回答“应该输出什么”“两者谁更好”还是“结果是否满足规则”，再进入对应训练方法。",
    termOrderRules: [
      {
        term: "SFT",
        introduction: "supervised fine-tuning（监督微调，SFT）",
      },
      {
        term: "RLVR",
        introduction: "reinforcement learning with verifiable rewards（RLVR）",
      },
      {
        term: "reward model",
        introduction: "reward model（奖励模型）",
      },
      {
        term: "RLHF",
        introduction: "reinforcement learning from human feedback（RLHF）",
      },
      {
        term: "DPO",
        introduction: "direct preference optimization（直接偏好优化，DPO）",
      },
      {
        term: "GR-REINFORCE",
        introduction: "group-relative REINFORCE（GR-REINFORCE）",
      },
      {
        term: "GRPO",
        introduction: "group relative policy optimization（GRPO）",
      },
    ],
  },
  {
    slug: "week3",
    source: path.join(projectDir, "content", "week3.md"),
    label: "WEEK 03 · IMPLEMENTATION EDITION",
    title: "把训练规则实现成可靠代码",
    description: "先固定输入、分组、评分和模型快照四条接口，再把已经推导过的训练规则落成张量、测试与可诊断循环。",
    meta: "第三周工程教材：从数据接口和张量契约到训练实现与故障诊断。",
    officialUrl: "https://github.com/berkeleydeeprlcourse/homework_spring2026",
    officialLabel: "Starter Code ↗",
    previous: "week2.html",
    next: "week4.html",
    note: "按纯函数、无训练 rollout、单次 backward、十步 smoke test 的顺序实现；简单任务没跑通前不要启动昂贵实验。",
  },
  {
    slug: "week4",
    source: path.join(projectDir, "content", "week4.md"),
    label: "WEEK 04 · EXPERIMENT EDITION",
    title: "证明训练有效并迁移到工具交互",
    description: "把一次成功训练升级为可信实验：建立对照、估计不确定性、定位评分漏洞，再迁移到多步工具环境。",
    meta: "第四周实验教材：可信比较、受控消融、评分漏洞审计与工具交互迁移。",
    officialUrl: "https://rdi.berkeley.edu/adv-llm-agents/sp25",
    officialLabel: "Agent RL 课程 ↗",
    previous: "week3.html",
    next: null,
    note: "每个结论都要同时给出任务指标、机制指标、失败样例和成本；训练 reward 只能作为证据之一。",
    termOrderRules: [
      {
        term: "pass@1",
        introduction: "**pass@1**",
      },
      {
        term: "seed",
        introduction: "random seed（随机种子）",
      },
      {
        term: "reward shaping",
        introduction: "reward shaping（奖励塑形）",
      },
      {
        term: "reward hacking",
        introduction: "reward hacking（奖励投机）",
      },
      {
        term: "red teaming",
        introduction: "red teaming（红队测试）",
      },
      {
        term: "credit assignment",
        introduction: "credit assignment（信用分配）",
      },
      {
        term: "Agent",
        introduction: "Agent（智能体）",
      },
    ],
  },
];

const officialLinks = new Map([
  ["cs285-spring2026-materials/slides/lec-4.pdf", "https://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-4.pdf"],
  ["cs285-spring2026-materials/slides/lec-5.pdf", "https://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-5.pdf"],
  ["cs285-spring2026-materials/slides/lec-6.pdf", "https://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-6.pdf"],
  ["cs285-spring2026-materials/slides/lec-9.pdf", "https://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-9.pdf"],
  ["cs285-spring2026-materials/slides/lec-10.pdf", "https://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-10.pdf"],
]);

function renderMarkdown(source) {
  let markdown = source;
  for (const [local, remote] of officialLinks) {
    markdown = markdown.replaceAll(`](${local})`, `](${remote})`);
  }

  const displayMath = [];
  markdown = markdown.replace(/\\\[[\s\S]*?\\\]/g, (formula) => {
    const index = displayMath.push(formula) - 1;
    return `\n\nGUIDEBLOCKMATH${index}END\n\n`;
  });

  const inlineMath = [];
  markdown = markdown.replace(/\\\([\s\S]*?\\\)/g, (formula) => {
    const index = inlineMath.push(formula) - 1;
    return `GUIDEINLINEMATH${index}END`;
  });

  let body = marked.parse(markdown, { gfm: true });
  body = body.replace(/<p>GUIDEBLOCKMATH(\d+)END<\/p>/g, (_, index) => {
    const latex = displayMath[Number(index)].slice(2, -2).trim();
    return `<div class="math-block">${katex.renderToString(latex, {
      displayMode: true,
      throwOnError: false,
    })}</div>`;
  });
  body = body.replace(/GUIDEINLINEMATH(\d+)END/g, (_, index) => {
    const latex = inlineMath[Number(index)].slice(2, -2).trim();
    return `<span class="math-inline">${katex.renderToString(latex, {
      displayMode: false,
      throwOnError: false,
    })}</span>`;
  });
  return body;
}

function navigation(page) {
  const links = [];
  if (page.previous) links.push(`<a href="${page.previous}">← 上一周</a>`);
  if (page.next) links.push(`<a href="${page.next}">下一周 →</a>`);
  return links.join("");
}

function createHtml(page, body) {
  return `<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <meta name="description" content="${page.meta}">
  <meta name="theme-color" content="#f7f6f0" media="(prefers-color-scheme: light)">
  <meta name="theme-color" content="#0d110e" media="(prefers-color-scheme: dark)">
  <title>${page.title}｜LLM / RL Study Hub</title>
  <link rel="stylesheet" href="vendor/katex.min.css">
  <style>
    :root{color-scheme:light;--paper:#f7f6f0;--ink:#111612;--muted:#657067;--line:#d7dacf;--deep:#173b35;--acid:#d8ff47;--code:#151a16;--link:#173b35;--soft:#e7eadf;--soft-code:#e8ebe2;--raised:#fff;--hero:#173b35;--on-acid:#111612}
    *{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--paper);color:var(--ink);font:17px/1.78 Arial,"Microsoft YaHei",sans-serif}
    a{color:var(--link);text-decoration-thickness:1px;text-underline-offset:3px}.top{position:sticky;top:0;z-index:10;display:flex;align-items:center;gap:18px;padding:13px max(22px,5vw);background:#101511f2;color:#f7f6f0;border-bottom:1px solid #354238;backdrop-filter:blur(10px)}
    .top a{color:inherit;text-decoration:none}.brand{font-size:12px;font-weight:900;letter-spacing:.14em;margin-right:auto}.top .primary{background:var(--acid);color:var(--on-acid);padding:7px 10px;font-weight:800}
    .week-nav{display:flex;gap:12px}.week-nav a{font-size:13px;color:#cbd4cb}.progress{position:fixed;left:0;top:0;height:3px;background:var(--acid);z-index:20;width:0}
    .hero{padding:84px max(24px,calc((100vw - 1040px)/2)) 66px;background:var(--hero);color:#f7f6f0}.hero small{color:var(--acid);font-weight:800;letter-spacing:.13em}
    .hero h1{font-size:clamp(46px,7vw,90px);line-height:.95;letter-spacing:-.06em;max-width:940px;margin:22px 0 28px}.hero p{max-width:760px;color:#d5dfd5;font-size:19px}
    .layout{display:grid;grid-template-columns:minmax(0,760px) 230px;gap:56px;max-width:1080px;margin:0 auto;padding:70px 24px 120px}article{min-width:0}
    article>h1:first-child,article>blockquote:first-of-type{display:none}h1,h2,h3{line-height:1.2;letter-spacing:-.035em;scroll-margin-top:75px}
    h1{font-size:42px;margin:72px 0 20px;border-top:1px solid var(--line);padding-top:52px}h2{font-size:28px;margin:46px 0 15px}h3{font-size:21px;margin:34px 0 12px}p{margin:13px 0}hr{border:0;border-top:1px solid var(--line);margin:55px 0}
    blockquote{margin:22px 0;padding:18px 22px;background:var(--soft);border-left:4px solid var(--deep);color:var(--muted)}blockquote p{margin:5px 0}
    table{border-collapse:collapse;width:100%;font-size:15px;margin:24px 0;display:block;overflow:auto}th,td{border:1px solid var(--line);padding:10px 12px;text-align:left;vertical-align:top}th{background:var(--soft)}
    code{font-family:"Cascadia Code",Consolas,monospace;font-size:.9em;background:var(--soft-code);padding:2px 5px;border-radius:3px}pre{background:var(--code);color:#ecf3e9;padding:20px;overflow:auto;border-radius:5px;line-height:1.55}pre code{background:none;padding:0}
    .math-block{overflow-x:auto;margin:24px 0;padding:18px 12px;background:var(--raised);border:1px solid var(--line);text-align:center}.math-inline{white-space:nowrap}
    details{margin:25px 0;padding:16px 18px;border:1px solid var(--line);background:var(--raised)}summary{cursor:pointer}
    aside{position:sticky;top:76px;align-self:start;height:max-content;max-height:calc(100vh - 96px);overflow-y:auto;overscroll-behavior:contain;scrollbar-gutter:stable;border-left:1px solid var(--line);padding:0 10px 18px 20px;scrollbar-width:thin}
    aside strong{position:sticky;top:0;z-index:1;display:block;font-size:11px;letter-spacing:.12em;margin:0 0 11px;padding:8px 0 10px;background:var(--paper);color:var(--muted)}
    aside a{display:block;font-size:13px;line-height:1.35;margin:0 0 10px;text-decoration:none;color:var(--muted)}aside a:hover,aside a:focus-visible{color:var(--link);text-decoration:underline}
    .note{background:#111612;color:#e6eee5;padding:28px;margin-top:58px}.note b{color:var(--acid)}.bottom-nav{display:flex;justify-content:space-between;gap:20px;margin-top:32px}.bottom-nav a{font-weight:800}
    footer{background:#101511;color:#aeb7ae;padding:24px max(24px,5vw);font-size:12px}
    @media(prefers-color-scheme:dark){:root{color-scheme:dark;--paper:#0d110e;--ink:#e5ece6;--muted:#aab5ab;--line:#303a32;--deep:#276052;--code:#080b09;--link:#8ed6ba;--soft:#171e19;--soft-code:#1b231d;--raised:#121713;--hero:#12372f}.math-block,.note,pre{box-shadow:inset 0 0 0 1px #202a22}}
    @media(max-width:860px){.layout{grid-template-columns:1fr}.layout aside{display:none}.top .hide-mobile,.top .primary{display:none}.hero{padding-top:64px}body{font-size:16px}.math-block{text-align:left}.week-nav{gap:8px}.week-nav a{font-size:12px}}
  </style>
</head>
<body>
  <div class="progress" id="progress"></div>
  <nav class="top">
    <a class="brand" href="index.html">LLM / RL STUDY HUB</a>
    <div class="week-nav">${navigation(page)}</div>
    <a class="hide-mobile" href="#content">中文教材</a>
    <a class="primary" href="${page.officialUrl}" target="_blank" rel="noreferrer">${page.officialLabel}</a>
  </nav>
  <header class="hero"><small>${page.label}</small><h1>${page.title}</h1><p>${page.description}</p></header>
  <main class="layout" id="content">
    <article>${body}
      <div class="note"><b>学习建议：</b>${page.note}</div>
      <div class="bottom-nav">${navigation(page)}</div>
    </article>
    <aside id="toc"><strong>本页目录</strong></aside>
  </main>
  <footer>LLM / RL STUDY HUB · ${page.label} · 公式已预渲染</footer>
  <script>
    const headings=[...document.querySelectorAll('article h1,article h2')];
    const toc=document.getElementById('toc');
    headings.forEach((h,i)=>{h.id='section-'+i;const a=document.createElement('a');a.href='#'+h.id;a.textContent=h.textContent;a.style.paddingLeft=h.tagName==='H2'?'12px':'0';toc.appendChild(a)});
    addEventListener('scroll',()=>{const d=document.documentElement;const range=d.scrollHeight-d.clientHeight;document.getElementById('progress').style.width=(range>0?d.scrollTop/range*100:0)+'%'},{passive:true});
  </script>
</body>
</html>`;
}

const katexDist = path.join(projectDir, "node_modules", "katex", "dist");
await Promise.all([
  fs.mkdir(path.join(projectDir, "public", "vendor"), { recursive: true }),
  fs.mkdir(path.join(projectDir, "docs", "vendor"), { recursive: true }),
]);
await Promise.all([
  fs.copyFile(path.join(katexDist, "katex.min.css"), path.join(projectDir, "public", "vendor", "katex.min.css")),
  fs.copyFile(path.join(katexDist, "katex.min.css"), path.join(projectDir, "docs", "vendor", "katex.min.css")),
  fs.cp(path.join(katexDist, "fonts"), path.join(projectDir, "public", "vendor", "fonts"), { recursive: true }),
  fs.cp(path.join(katexDist, "fonts"), path.join(projectDir, "docs", "vendor", "fonts"), { recursive: true }),
]);

for (const page of pages) {
  let source = await fs.readFile(page.source, "utf8");
  if (page.prefixReplacementSource) {
    const prefixReplacement = await fs.readFile(page.prefixReplacementSource, "utf8");
    const prefixEnd = source.indexOf(page.prefixReplaceTo);
    if (prefixEnd < 0) {
      throw new Error(`Cannot find prefix boundary ${page.prefixReplaceTo} in ${page.source}`);
    }
    source = `${prefixReplacement.trim()}\n\n${source.slice(prefixEnd).trimStart()}`;
  }
  if (page.subsectionReplacementSource) {
    const subsectionReplacement = await fs.readFile(page.subsectionReplacementSource, "utf8");
    const subsectionStart = source.indexOf(page.subsectionReplaceFrom);
    const subsectionEnd = source.indexOf(page.subsectionReplaceTo, subsectionStart);
    if (subsectionStart < 0 || subsectionEnd < 0) {
      throw new Error(
        `Cannot find subsection boundaries ${page.subsectionReplaceFrom}..${page.subsectionReplaceTo}`,
      );
    }
    source = [
      source.slice(0, subsectionStart).trimEnd(),
      subsectionReplacement.trim(),
      source.slice(subsectionEnd).trimStart(),
    ].join("\n\n");
  }
  if (page.replacementSource) {
    const replacement = await fs.readFile(page.replacementSource, "utf8");
    const boundary = source.indexOf(page.replaceFrom);
    if (boundary < 0) {
      throw new Error(`Cannot find replacement boundary ${page.replaceFrom} in ${page.source}`);
    }
    source = `${source.slice(0, boundary).trimEnd()}\n\n${replacement.trimStart()}`;
  }
  for (const sectionReplacementConfig of page.sectionReplacements ?? []) {
    const sectionReplacement = await fs.readFile(
      sectionReplacementConfig.source,
      "utf8",
    );
    const sectionStart = source.indexOf(sectionReplacementConfig.from);
    const sectionEnd = source.indexOf(sectionReplacementConfig.to, sectionStart);
    if (sectionStart < 0 || sectionEnd < 0) {
      throw new Error(
        `Cannot find section boundaries ${sectionReplacementConfig.from}..${sectionReplacementConfig.to}`,
      );
    }
    source = [
      source.slice(0, sectionStart).trimEnd(),
      sectionReplacement.trim(),
      source.slice(sectionEnd).trimStart(),
    ].join("\n\n");
  }
  for (const textReplacement of page.textReplacements ?? []) {
    if (!source.includes(textReplacement.from)) {
      throw new Error(
        `Cannot find text replacement target ${textReplacement.from} in ${page.source}`,
      );
    }
    source = source.replace(textReplacement.from, textReplacement.to);
  }
  for (const rule of page.termOrderRules ?? []) {
    const firstTermIndex = source.indexOf(rule.term);
    const introductionIndex = source.indexOf(rule.introduction);
    if (introductionIndex < 0) {
      throw new Error(`Missing introduction for ${rule.term} in ${page.slug}`);
    }
    if (firstTermIndex < introductionIndex) {
      throw new Error(
        `${rule.term} appears before its introduction in ${page.slug}`,
      );
    }
  }
  const body = renderMarkdown(source);
  const html = createHtml(page, body);
  await Promise.all([
    fs.writeFile(path.join(projectDir, "public", `${page.slug}.html`), html, "utf8"),
    fs.writeFile(path.join(projectDir, "docs", `${page.slug}.html`), html, "utf8"),
  ]);
  console.log(`Rendered ${page.slug}: ${html.length} bytes`);
}
