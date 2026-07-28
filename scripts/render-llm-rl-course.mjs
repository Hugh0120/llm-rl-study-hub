import fs from "node:fs/promises";
import path from "node:path";
import katex from "katex";

const markedModule = process.env.MARKED_MODULE;
if (!markedModule) {
  throw new Error("MARKED_MODULE is required and must point to marked.esm.js");
}
const { marked } = await import(markedModule);

const projectDir = path.resolve(import.meta.dirname, "..");
const sourceDir = path.join(projectDir, "content", "course");
const outputRoots = [path.join(projectDir, "docs")];

const pages = [
  {
    slug: "00-roadmap",
    source: "00-roadmap.md",
    number: "00",
    short: "课程路线",
    title: "从回答得分到可靠训练系统",
    summary: "先看全局问题链、符号表、章节依赖和贯穿案例，再开始推导。",
    time: "35 MIN",
    topics: ["全局路线", "符号表", "学习方法"],
  },
  {
    slug: "01-llm-decision-process",
    source: "01-llm-decision-process.md",
    number: "01",
    short: "生成即决策",
    title: "语言模型到底在做什么选择",
    summary: "从一次自回归回答建立状态、动作、轨迹概率、reward 与训练目标。",
    time: "70 MIN",
    topics: ["MDP 映射", "轨迹概率", "期望回报"],
  },
  {
    slug: "02-score-to-gradient",
    source: "02-score-to-gradient.md",
    number: "02",
    short: "分数到梯度",
    title: "最终分数怎样改变 token 概率",
    summary: "逐行推导 log-derivative、policy gradient 与 REINFORCE，不对离散 token 求导。",
    time: "90 MIN",
    topics: ["Policy Gradient", "REINFORCE", "Reward-to-go"],
  },
  {
    slug: "03-value-td-gae",
    source: "03-value-td-gae.md",
    number: "03",
    short: "相对表现",
    title: "怎样判断每一步做得好不好",
    summary: "由正常水平依次推出 value、advantage、TD、GAE，最后才命名 Actor–Critic。",
    time: "120 MIN",
    topics: ["Value", "TD", "GAE"],
  },
  {
    slug: "04-ppo-and-kl",
    source: "04-ppo-and-kl.md",
    number: "04",
    short: "受控复用",
    title: "rollout 很贵，怎样安全地多用几次",
    summary: "从旧数据问题推出 importance ratio、PPO clipping，并分清 old 与 reference。",
    time: "110 MIN",
    topics: ["Importance Ratio", "PPO", "KL"],
  },
  {
    slug: "05-feedback-and-preference",
    source: "05-feedback-and-preference.md",
    number: "05",
    short: "反馈来源",
    title: "模型的分数从哪里来",
    summary: "先区分示范、偏好与可验证结果，再引出 SFT、RLHF、DPO 与 RLVR。",
    time: "105 MIN",
    topics: ["SFT", "RLHF / DPO", "RLVR"],
  },
  {
    slug: "06-group-relative-grpo",
    source: "06-group-relative-grpo.md",
    number: "06",
    short: "同题多采样",
    title: "能不能不用 critic",
    summary: "由同题组均值构造相对 advantage，再按是否复用数据得到 GR-REINFORCE 与 GRPO。",
    time: "100 MIN",
    topics: ["Group Baseline", "GR-REINFORCE", "GRPO"],
  },
  {
    slug: "07-implementation",
    source: "07-implementation.md",
    number: "07",
    short: "工程落地",
    title: "把公式变成不会错位的训练代码",
    summary: "固定 tensor contract、mask、log-prob 生命周期、verifier 分层与 smoke tests。",
    time: "120 MIN",
    topics: ["Tensor Contract", "Lifecycle", "Debugging"],
  },
  {
    slug: "08-evaluation",
    source: "08-evaluation.md",
    number: "08",
    short: "可信评估",
    title: "怎样证明模型真的变好了",
    summary: "将 reward 曲线升级为 held-out、配对比较、不确定性、消融与漏洞审计。",
    time: "85 MIN",
    topics: ["pass@k", "Ablation", "Reward Hacking"],
  },
  {
    slug: "09-tool-agent",
    source: "09-tool-agent.md",
    number: "09",
    short: "工具 Agent",
    title: "从单轮回答走向外部环境",
    summary: "逐项替换状态、动作、转移与 reward，加入真实状态核验、成本和安全边界。",
    time: "95 MIN",
    topics: ["Tool Use", "Environment", "Safety"],
  },
];

function escapeHtml(value) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function stripHtml(value) {
  return value
    .replace(/<[^>]+>/g, "")
    .replaceAll("&amp;", "&")
    .replaceAll("&lt;", "<")
    .replaceAll("&gt;", ">")
    .replace(/\s+/g, " ")
    .trim();
}

function slugFromHeading(text, fallback) {
  const number = text.match(/^(\d+(?:\.\d+)*)/);
  if (number) return `section-${number[1].replaceAll(".", "-")}`;
  const ascii = text
    .toLowerCase()
    .replace(/[^a-z0-9\u4e00-\u9fff]+/g, "-")
    .replace(/^-|-$/g, "");
  return ascii || fallback;
}

function renderMarkdown(source) {
  const displayMath = [];
  let markdown = source.replace(/\\\[[\s\S]*?\\\]/g, (formula) => {
    const index = displayMath.push(formula) - 1;
    return `\n\nCOURSEBLOCKMATH${index}END\n\n`;
  });

  const inlineMath = [];
  markdown = markdown.replace(/\\\([\s\S]*?\\\)/g, (formula) => {
    const index = inlineMath.push(formula) - 1;
    return `COURSEINLINEMATH${index}END`;
  });

  let body = marked.parse(markdown, { gfm: true });

  body = body.replace(/<p>COURSEBLOCKMATH(\d+)END<\/p>/g, (_, index) => {
    const latex = displayMath[Number(index)].slice(2, -2).trim();
    return `<div class="math-block">${katex.renderToString(latex, {
      displayMode: true,
      throwOnError: false,
      strict: false,
    })}</div>`;
  });

  body = body.replace(/COURSEINLINEMATH(\d+)END/g, (_, index) => {
    const latex = inlineMath[Number(index)].slice(2, -2).trim();
    return `<span class="math-inline">${katex.renderToString(latex, {
      displayMode: false,
      throwOnError: false,
      strict: false,
    })}</span>`;
  });

  const headings = [];
  const usedIds = new Map();
  body = body.replace(/<h([1-4])>([\s\S]*?)<\/h\1>/g, (_, level, inner) => {
    const text = stripHtml(inner);
    const base = slugFromHeading(text, `heading-${headings.length + 1}`);
    const count = usedIds.get(base) ?? 0;
    usedIds.set(base, count + 1);
    const id = count === 0 ? base : `${base}-${count + 1}`;
    headings.push({ level: Number(level), text, id });
    return `<h${level} id="${id}">${inner}<a class="heading-anchor" href="#${id}" aria-label="链接到本节">#</a></h${level}>`;
  });

  body = body.replace(
    /<a href="(https?:\/\/[^"]+)">/g,
    '<a href="$1" target="_blank" rel="noreferrer">'
  );

  return { body, headings };
}

function chapterHref(page) {
  return `${page.slug}.html`;
}

function chapterNavigation(index, location = "course") {
  const previous = index > 0 ? pages[index - 1] : null;
  const next = index < pages.length - 1 ? pages[index + 1] : null;
  const indexHref = location === "course" ? "index.html" : "course/index.html";
  return `<nav class="chapter-nav" aria-label="章节导航">
      <div class="chapter-nav__left">
        <a href="${indexHref}" class="nav-index">课程目录</a>
        ${previous ? `<a href="${chapterHref(previous)}">← ${previous.number} ${escapeHtml(previous.short)}</a>` : "<span></span>"}
      </div>
      <div class="chapter-nav__right">
        ${next ? `<a class="nav-next" href="${chapterHref(next)}">${next.number} ${escapeHtml(next.short)} →</a>` : `<a class="nav-next" href="${indexHref}">回到课程目录 ↑</a>`}
      </div>
    </nav>`;
}

function tocHtml(headings) {
  const sections = headings.filter((heading) => heading.level === 2);
  return `
    <details class="chapter-toc" open>
      <summary>本章目录 <span>展开 / 收起</span></summary>
      <nav class="toc-links" aria-label="本章目录">
        ${sections
          .map(
            (heading) =>
              `<a href="#${heading.id}">${escapeHtml(heading.text)}</a>`
          )
          .join("\n")}
      </nav>
    </details>`;
}

const courseCss = `
  :root {
    color-scheme: light;
    --paper: #f7f5ee;
    --surface: #fffef9;
    --soft: #ecefe5;
    --ink: #161a17;
    --muted: #5d675f;
    --line: #cfd5ca;
    --deep: #102c27;
    --accent: #cfff45;
    --link: #126450;
    --code: #101612;
    --code-ink: #eff8ef;
    --shadow: rgba(22, 30, 24, .08);
  }
  * { box-sizing: border-box; }
  html { scroll-behavior: smooth; scroll-padding-top: 20px; }
  body {
    margin: 0;
    background: var(--paper);
    color: var(--ink);
    font: 18px/1.82 Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", "Microsoft YaHei", sans-serif;
    text-rendering: optimizeLegibility;
  }
  a { color: var(--link); text-decoration-thickness: 1px; text-underline-offset: 3px; }
  a:hover { text-decoration-thickness: 2px; }
  .course-strip {
    border-bottom: 1px solid var(--line);
    background: var(--deep);
    color: #f4f7f1;
  }
  .course-strip__inner {
    max-width: 1100px;
    margin: 0 auto;
    padding: 13px 28px;
    display: flex;
    justify-content: space-between;
    gap: 20px;
    align-items: center;
    font-size: 13px;
    font-weight: 750;
    letter-spacing: .04em;
  }
  .course-strip a { color: var(--accent); text-decoration: none; }
  .doc-shell {
    width: min(100% - 40px, 980px);
    margin: 0 auto;
    padding: 30px 0 88px;
  }
  .chapter-nav {
    min-height: 55px;
    display: flex;
    justify-content: space-between;
    gap: 24px;
    align-items: center;
    border-bottom: 1px solid var(--line);
    margin-bottom: 54px;
    font-size: 14px;
    font-weight: 760;
  }
  .chapter-nav__left, .chapter-nav__right { display: flex; gap: 20px; align-items: center; }
  .chapter-nav a { text-decoration: none; }
  .nav-index {
    color: var(--ink) !important;
    border-right: 1px solid var(--line);
    padding-right: 20px;
  }
  .nav-next {
    display: inline-flex;
    align-items: center;
    min-height: 34px;
    padding: 0 12px;
    color: var(--deep) !important;
    background: var(--accent);
  }
  .chapter-kicker {
    color: var(--link);
    font-size: 12px;
    font-weight: 850;
    letter-spacing: .16em;
    text-transform: uppercase;
    margin-bottom: 18px;
  }
  article > h1 {
    max-width: 900px;
    margin: 0 0 32px;
    font-size: clamp(42px, 5.8vw, 64px);
    line-height: 1.05;
    letter-spacing: -.055em;
  }
  h1, h2, h3, h4 { font-weight: 830; }
  h2 {
    margin: 76px 0 24px;
    padding-top: 14px;
    border-top: 1px solid var(--line);
    font-size: clamp(29px, 4vw, 42px);
    line-height: 1.17;
    letter-spacing: -.035em;
  }
  h3 {
    margin: 46px 0 16px;
    font-size: clamp(23px, 3vw, 30px);
    line-height: 1.28;
    letter-spacing: -.025em;
  }
  h4 { margin-top: 32px; font-size: 20px; }
  .heading-anchor {
    margin-left: 10px;
    opacity: 0;
    color: var(--muted);
    text-decoration: none;
    font-size: .62em;
  }
  h1:hover .heading-anchor, h2:hover .heading-anchor, h3:hover .heading-anchor, h4:hover .heading-anchor { opacity: .7; }
  p { margin: 18px 0; }
  strong { font-weight: 800; }
  blockquote {
    margin: 28px 0;
    padding: 20px 24px;
    border: 1px solid var(--line);
    border-left: 6px solid var(--accent);
    background: var(--surface);
    font-size: 20px;
  }
  blockquote p { margin: 0; }
  ul, ol { padding-left: 1.45em; }
  li { margin: 8px 0; padding-left: 5px; }
  table {
    width: 100%;
    border-collapse: collapse;
    margin: 26px 0 34px;
    background: var(--surface);
    font-size: 15.5px;
    line-height: 1.55;
  }
  th, td { border: 1px solid var(--line); padding: 12px 14px; text-align: left; vertical-align: top; }
  th { background: var(--soft); font-weight: 800; }
  code {
    padding: .12em .36em;
    border-radius: 4px;
    background: var(--soft);
    color: var(--ink);
    font: .88em/1.5 "SFMono-Regular", Consolas, "Liberation Mono", monospace;
  }
  pre {
    overflow: auto;
    margin: 26px 0 34px;
    padding: 22px 24px;
    border: 1px solid #27322b;
    border-radius: 6px;
    background: var(--code);
    color: var(--code-ink);
    font-size: 14.5px;
    line-height: 1.7;
  }
  pre code { padding: 0; background: transparent; color: inherit; }
  .math-block {
    overflow-x: auto;
    margin: 28px 0 34px;
    padding: 24px 20px;
    border: 1px solid var(--line);
    background: var(--surface);
    text-align: center;
  }
  .math-inline { white-space: nowrap; }
  .chapter-toc {
    margin: 38px 0 62px;
    border: 1px solid var(--line);
    background: var(--surface);
  }
  .chapter-toc summary {
    cursor: pointer;
    list-style: none;
    padding: 17px 20px;
    display: flex;
    justify-content: space-between;
    font-weight: 820;
    background: var(--soft);
  }
  .chapter-toc summary::-webkit-details-marker { display: none; }
  .chapter-toc summary span { color: var(--muted); font-size: 12px; letter-spacing: .06em; }
  .toc-links {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 0 22px;
    padding: 12px 20px 18px;
    max-height: 420px;
    overflow-y: auto;
  }
  .toc-links a {
    padding: 8px 0;
    border-bottom: 1px solid var(--line);
    text-decoration: none;
    font-size: 14px;
    line-height: 1.45;
  }
  .chapter-end { margin-top: 80px; border-top: 1px solid var(--line); }
  .page-progress {
    display: flex;
    align-items: center;
    gap: 14px;
    color: var(--muted);
    font-size: 13px;
    margin: -28px 0 36px;
  }
  .page-progress__bar {
    height: 5px;
    flex: 1;
    background: var(--line);
  }
  .page-progress__fill { height: 100%; background: var(--link); }
  .course-footer {
    border-top: 1px solid var(--line);
    padding: 24px;
    color: var(--muted);
    text-align: center;
    font-size: 12px;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      color-scheme: dark;
      --paper: #0d120f;
      --surface: #131a15;
      --soft: #19231c;
      --ink: #e9efe9;
      --muted: #a3aea5;
      --line: #303b33;
      --deep: #07100c;
      --accent: #cfff45;
      --link: #86d6b7;
      --code: #080d0a;
      --code-ink: #eaf4eb;
      --shadow: rgba(0, 0, 0, .25);
    }
    .nav-next { color: #101510 !important; }
  }
  @media (max-width: 720px) {
    body { font-size: 16.5px; line-height: 1.75; }
    .doc-shell { width: min(100% - 28px, 980px); padding-top: 18px; }
    .course-strip__inner { padding: 11px 16px; }
    .chapter-nav { align-items: stretch; flex-direction: column; padding-bottom: 14px; margin-bottom: 38px; }
    .chapter-nav__left, .chapter-nav__right { justify-content: space-between; }
    article > h1 { font-size: clamp(35px, 9.7vw, 42px); line-height: 1.08; }
    h2 { margin-top: 60px; }
    .toc-links { grid-template-columns: 1fr; max-height: 360px; }
    .math-block { padding: 18px 10px; }
    table { display: block; overflow-x: auto; white-space: normal; }
    .heading-anchor { display: none; }
  }
`;

function chapterDocument(page, index, rendered) {
  const progress = Math.round(((index + 1) / pages.length) * 100);
  return `<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="description" content="${escapeHtml(page.summary)}">
  <meta name="theme-color" content="#f7f5ee" media="(prefers-color-scheme: light)">
  <meta name="theme-color" content="#0d120f" media="(prefers-color-scheme: dark)">
  <title>${escapeHtml(page.title)}｜大模型强化学习教程</title>
  <link rel="stylesheet" href="../vendor/katex.min.css">
  <style>${courseCss}</style>
</head>
<body>
  <header class="course-strip">
    <div class="course-strip__inner">
      <span>LLM RL · 连续教程</span>
      <a href="index.html">10 页完整问题链</a>
    </div>
  </header>
  <main class="doc-shell">
    ${chapterNavigation(index)}
    <div class="chapter-kicker">CHAPTER ${page.number} · ${escapeHtml(page.short)} · ${page.time}</div>
    <div class="page-progress">
      <span>${index + 1} / ${pages.length}</span>
      <div class="page-progress__bar"><div class="page-progress__fill" style="width:${progress}%"></div></div>
      <span>${progress}%</span>
    </div>
    <article>
      ${rendered.body.replace(
        /(<\/h1>)/,
        `$1${tocHtml(rendered.headings)}`
      )}
    </article>
    <div class="chapter-end">${chapterNavigation(index)}</div>
  </main>
  <footer class="course-footer">从反馈接口到可靠训练系统 · 旧版四周讲义仍保留在站点归档中</footer>
</body>
</html>`;
}

const indexCss = `
  :root {
    color-scheme: light;
    --paper: #f6f4ec;
    --ink: #121713;
    --muted: #5e685f;
    --line: #cbd1c6;
    --deep: #0d2923;
    --accent: #cfff45;
    --card: #fffef9;
    --link: #125e4c;
  }
  * { box-sizing: border-box; }
  html { scroll-behavior: smooth; }
  body { margin: 0; background: var(--paper); color: var(--ink); font: 16px/1.65 Inter, system-ui, "Segoe UI", "Microsoft YaHei", sans-serif; }
  a { color: inherit; }
  .hero { background: var(--deep); color: #f5f8f4; padding: 72px max(24px, 8vw) 82px; }
  .hero__inner { max-width: 1160px; margin: auto; }
  .eyebrow { color: var(--accent); font-size: 12px; font-weight: 850; letter-spacing: .16em; }
  h1 { max-width: 1030px; margin: 26px 0; font-size: clamp(51px, 8.7vw, 116px); line-height: .95; letter-spacing: -.068em; }
  h1 em { color: var(--accent); font-style: normal; }
  .lead { max-width: 760px; color: #d5ddd7; font-size: clamp(18px, 2vw, 22px); }
  .actions { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 32px; }
  .button { padding: 12px 17px; border: 1px solid var(--accent); background: var(--accent); color: #111711; text-decoration: none; font-weight: 820; }
  .button.alt { background: transparent; color: #f5f8f4; }
  main { max-width: 1160px; margin: auto; padding: 70px 24px 100px; }
  .section-head { display: grid; grid-template-columns: .9fr 1.1fr; gap: 50px; align-items: end; margin-bottom: 36px; }
  h2 { margin: 0; font-size: clamp(37px, 5vw, 66px); line-height: 1; letter-spacing: -.05em; }
  .section-head p { margin: 0; color: var(--muted); font-size: 18px; }
  .chain {
    display: flex;
    gap: 0;
    overflow-x: auto;
    padding-bottom: 8px;
    margin-bottom: 64px;
  }
  .chain span { min-width: 135px; border: 1px solid var(--line); border-right: 0; padding: 12px; background: var(--card); font-size: 12px; font-weight: 800; text-align: center; }
  .chain span:last-child { border-right: 1px solid var(--line); }
  .chapter-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; }
  .chapter-card {
    min-height: 305px;
    display: flex;
    flex-direction: column;
    padding: 26px;
    border: 1px solid var(--line);
    background: var(--card);
    text-decoration: none;
    transition: transform .18s, box-shadow .18s, border-color .18s;
  }
  .chapter-card:hover { transform: translateY(-4px); border-color: var(--link); box-shadow: 0 14px 32px rgba(18, 30, 22, .12); }
  .chapter-card__meta { display: flex; justify-content: space-between; color: var(--muted); font-size: 11px; font-weight: 850; letter-spacing: .12em; }
  .chapter-card h3 { margin: 48px 0 13px; font-size: clamp(27px, 3vw, 39px); line-height: 1.04; letter-spacing: -.045em; }
  .chapter-card p { margin: 0; color: var(--muted); }
  .topics { margin: 18px 0 26px; color: var(--link); font-size: 12px; font-weight: 800; }
  .enter { display: flex; justify-content: space-between; margin-top: auto; padding-top: 14px; border-top: 1px solid var(--line); font-weight: 820; }
  .principles { margin-top: 76px; display: grid; grid-template-columns: repeat(4, 1fr); border: 1px solid var(--line); }
  .principle { padding: 24px; border-right: 1px solid var(--line); }
  .principle:last-child { border-right: 0; }
  .principle strong { display: block; margin-bottom: 8px; font-size: 20px; }
  .principle p { margin: 0; color: var(--muted); font-size: 14px; }
  .archive { margin-top: 64px; padding: 28px; border: 1px solid var(--line); background: var(--card); }
  .archive h3 { margin: 0 0 8px; font-size: 24px; }
  .archive p { color: var(--muted); }
  .archive-links { display: flex; flex-wrap: wrap; gap: 10px; }
  .archive-links a { color: var(--link); font-weight: 760; }
  footer { padding: 24px; border-top: 1px solid var(--line); color: var(--muted); text-align: center; font-size: 12px; }
  @media (prefers-color-scheme: dark) {
    :root { color-scheme: dark; --paper: #0d120f; --ink: #e9efe9; --muted: #a4afa6; --line: #303b33; --deep: #07100c; --card: #131a15; --link: #88d4b7; }
  }
  @media (max-width: 760px) {
    .hero { padding-top: 52px; }
    .section-head { grid-template-columns: 1fr; gap: 18px; }
    .chapter-grid { grid-template-columns: 1fr; }
    .principles { grid-template-columns: 1fr; }
    .principle { border-right: 0; border-bottom: 1px solid var(--line); }
    .principle:last-child { border-bottom: 0; }
  }
`;

function courseCards(prefix = "") {
  return pages
    .map(
      (page) => `<a class="chapter-card" href="${prefix}${page.slug}.html">
          <div class="chapter-card__meta"><span>CHAPTER ${page.number}</span><span>${page.time}</span></div>
          <h3>${escapeHtml(page.title)}</h3>
          <p>${escapeHtml(page.summary)}</p>
          <div class="topics">${page.topics.map(escapeHtml).join(" · ")}</div>
          <div class="enter"><span>${page.number === "00" ? "先看课程路线" : "进入本章"}</span><span>↗</span></div>
        </a>`
    )
    .join("\n");
}

function indexDocument(nested = false) {
  const prefix = nested ? "" : "course/";
  const archivePrefix = nested ? "../" : "";
  return `<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="description" content="面向有大模型基础、强化学习基础较少的算法工程师：从一次回答得分连续推导到 REINFORCE、GAE、PPO、DPO、GRPO、可靠实现与 Agent。">
  <meta name="theme-color" content="#0d2923" media="(prefers-color-scheme: light)">
  <meta name="theme-color" content="#07100c" media="(prefers-color-scheme: dark)">
  <title>大模型强化学习｜从回答得分到可靠训练系统</title>
  <style>${indexCss}</style>
</head>
<body>
  <header class="hero">
    <div class="hero__inner">
      <div class="eyebrow">LLM REINFORCEMENT LEARNING · CONTINUOUS COURSE</div>
      <h1>从一个回答得分，<br>走到<em>可靠训练系统</em>。</h1>
      <p class="lead">面向已经熟悉 Transformer 与大模型训练、但没有系统强化学习基础的算法工程师。全课沿同一个数学问答案例前进；每章先交代上一章留下的问题，再引入唯一需要的新概念。</p>
      <div class="actions">
        <a class="button" href="${prefix}00-roadmap.html">先看完整路线 ↗</a>
        <a class="button alt" href="${prefix}01-llm-decision-process.html">直接开始第 1 章 →</a>
        <a class="button alt" href="https://rail.eecs.berkeley.edu/deeprlcourse/" target="_blank" rel="noreferrer">CS285 官方课程 ↗</a>
      </div>
    </div>
  </header>
  <main>
    <section class="section-head">
      <h2>一条问题链，<br>十页读完。</h2>
      <p>不按算法名堆章节。每一页只解决上一页留下的一个障碍：先理解模型怎样做选择，再解决梯度、方差、旧数据、反馈、组内比较、实现、评估，最后扩展到工具环境。</p>
    </section>
    <div class="chain" aria-label="课程问题链">
      <span>生成即决策</span><span>分数到梯度</span><span>相对表现</span><span>受控复用</span><span>反馈来源</span><span>同题多采样</span><span>可靠实现</span><span>可信评估</span><span>工具 Agent</span>
    </div>
    <section class="chapter-grid">
      ${courseCards(prefix)}
    </section>
    <section class="principles">
      <div class="principle"><strong>先问题</strong><p>每章先写清上一章已解决什么、当前还缺什么。</p></div>
      <div class="principle"><strong>后命名</strong><p>先用具体案例建立直觉，再给符号、公式和算法简称。</p></div>
      <div class="principle"><strong>同一案例</strong><p>始终围绕 23×17 的回答、判分和更新，避免上下文切换。</p></div>
      <div class="principle"><strong>落到代码</strong><p>推导最终连接 tensor、mask、生命周期、测试与评估。</p></div>
    </section>
    <section class="archive">
      <h3>旧版四周讲义归档</h3>
      <p>旧页面不再作为主课程入口，但仍保留，便于对照与查找原先的周计划。</p>
      <div class="archive-links">
        <a href="${archivePrefix}week1.html">Week 1</a>
        <a href="${archivePrefix}week2.html">Week 2</a>
        <a href="${archivePrefix}week3.html">Week 3</a>
        <a href="${archivePrefix}week4.html">Week 4</a>
      </div>
    </section>
  </main>
  <footer>大模型强化学习：从回答得分到可靠训练系统 · GitHub Pages</footer>
</body>
</html>`;
}

for (const root of outputRoots) {
  const courseDir = path.join(root, "course");
  await fs.mkdir(courseDir, { recursive: true });

  for (const [index, page] of pages.entries()) {
    const source = await fs.readFile(path.join(sourceDir, page.source), "utf8");
    const rendered = renderMarkdown(source);
    await fs.writeFile(
      path.join(courseDir, `${page.slug}.html`),
      chapterDocument(page, index, rendered),
      "utf8"
    );
  }

  await fs.writeFile(path.join(root, "index.html"), indexDocument(false), "utf8");
  await fs.writeFile(
    path.join(courseDir, "index.html"),
    indexDocument(true),
    "utf8"
  );
}

console.log(
  `Rendered ${pages.length} chapters + course indexes into docs/`
);
