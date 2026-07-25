import fs from "node:fs/promises";
import path from "node:path";

const markedModule = process.env.MARKED_MODULE;
if (!markedModule) throw new Error("MARKED_MODULE is required");
const { marked } = await import(markedModule);

const projectDir = path.resolve(import.meta.dirname, "..");
const sourcePath = path.resolve(projectDir, "..", "第一周-CS285-策略优化与PPO-中文精读.md");
const source = await fs.readFile(sourcePath, "utf8");

const officialLinks = new Map([
  ["cs285-spring2026-materials/slides/lec-4.pdf", "https://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-4.pdf"],
  ["cs285-spring2026-materials/slides/lec-5.pdf", "https://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-5.pdf"],
  ["cs285-spring2026-materials/slides/lec-6.pdf", "https://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-6.pdf"],
  ["cs285-spring2026-materials/slides/lec-9.pdf", "https://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-9.pdf"],
  ["cs285-spring2026-materials/slides/lec-10.pdf", "https://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-10.pdf"],
]);

let markdown = source;
for (const [local, remote] of officialLinks) markdown = markdown.replaceAll(`](${local})`, `](${remote})`);

const displayMath = [];
markdown = markdown.replace(/\\\[[\s\S]*?\\\]/g, (formula) => {
  const index = displayMath.push(formula) - 1;
  return `\n\nWEEKONEBLOCKMATH${index}END\n\n`;
});

const inlineMath = [];
markdown = markdown.replace(/\\\([\s\S]*?\\\)/g, (formula) => {
  const index = inlineMath.push(formula) - 1;
  return `WEEKONEINLINEMATH${index}END`;
});

let body = await marked.parse(markdown, { gfm: true });
body = body.replace(
  /<p>WEEKONEBLOCKMATH(\d+)END<\/p>/g,
  (_, index) => `<div class="math-block">${displayMath[Number(index)]}</div>`,
);
body = body.replace(
  /WEEKONEINLINEMATH(\d+)END/g,
  (_, index) => `<span class="math-inline">${inlineMath[Number(index)]}</span>`,
);

const html = `<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <meta name="description" content="面向算法工程师的 CS285 第一周中文精读：从策略梯度、Advantage、GAE 到 PPO。">
  <title>第一周｜从策略梯度到 PPO 的中文精读</title>
  <script>
    window.MathJax = {
      tex: { inlineMath: [['\\\\(', '\\\\)']], displayMath: [['\\\\[', '\\\\]']] },
      chtml: { scale: 0.96 },
      options: { skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code'] }
    };
  </script>
  <script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
  <style>
    :root{--paper:#f7f6f0;--ink:#111612;--muted:#657067;--line:#d7dacf;--deep:#173b35;--acid:#d8ff47;--code:#151a16}
    *{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--paper);color:var(--ink);font:17px/1.78 Arial,"Microsoft YaHei",sans-serif}
    a{color:var(--deep);text-decoration-thickness:1px;text-underline-offset:3px}.top{position:sticky;top:0;z-index:10;display:flex;align-items:center;gap:18px;padding:13px max(22px,5vw);background:#101511f2;color:#f7f6f0;border-bottom:1px solid #354238;backdrop-filter:blur(10px)}
    .top a{color:inherit;text-decoration:none}.brand{font-size:12px;font-weight:900;letter-spacing:.14em;margin-right:auto}.top .primary{background:var(--acid);color:var(--ink);padding:7px 10px;font-weight:800}
    .progress{position:fixed;left:0;top:0;height:3px;background:var(--acid);z-index:20;width:0}.hero{padding:84px max(24px,calc((100vw - 1040px)/2)) 66px;background:var(--deep);color:#f7f6f0}
    .hero small{color:var(--acid);font-weight:800;letter-spacing:.13em}.hero h1{font-size:clamp(46px,7vw,90px);line-height:.95;letter-spacing:-.06em;max-width:920px;margin:22px 0 28px}.hero p{max-width:720px;color:#d5dfd5;font-size:19px}
    .layout{display:grid;grid-template-columns:minmax(0,760px) 230px;gap:56px;max-width:1080px;margin:0 auto;padding:70px 24px 120px}
    article{min-width:0}article>h1:first-child,article>blockquote:first-of-type{display:none}h1,h2,h3{line-height:1.2;letter-spacing:-.035em;scroll-margin-top:75px}h1{font-size:42px;margin:72px 0 20px;border-top:1px solid var(--line);padding-top:52px}h2{font-size:28px;margin:46px 0 15px}h3{font-size:21px;margin:34px 0 12px}p{margin:13px 0}hr{border:0;border-top:1px solid var(--line);margin:55px 0}
    blockquote{margin:22px 0;padding:18px 22px;background:#e7eadf;border-left:4px solid var(--deep);color:#39443b}blockquote p{margin:5px 0}
    table{border-collapse:collapse;width:100%;font-size:15px;margin:24px 0;display:block;overflow:auto}th,td{border:1px solid var(--line);padding:10px 12px;text-align:left;vertical-align:top}th{background:#e7eadf}
    code{font-family:"Cascadia Code",Consolas,monospace;font-size:.9em;background:#e8ebe2;padding:2px 5px;border-radius:3px}pre{background:var(--code);color:#ecf3e9;padding:20px;overflow:auto;border-radius:5px;line-height:1.55}pre code{background:none;padding:0}
    .math-block{overflow-x:auto;margin:24px 0;padding:18px 12px;background:#fff;border:1px solid var(--line);text-align:center}.math-inline{white-space:nowrap}
    aside{position:sticky;top:90px;height:max-content;border-left:1px solid var(--line);padding-left:20px}aside strong{display:block;font-size:11px;letter-spacing:.12em;margin-bottom:11px;color:var(--muted)}aside a{display:block;font-size:13px;line-height:1.35;margin:0 0 10px;text-decoration:none;color:var(--muted)}aside a:hover{color:var(--deep)}
    .note{background:#111612;color:#e6eee5;padding:28px;margin-top:58px}.note b{color:var(--acid)}footer{background:#101511;color:#aeb7ae;padding:24px max(24px,5vw);font-size:12px}
    @media(max-width:860px){.layout{grid-template-columns:1fr}.layout aside{display:none}.top .hide-mobile{display:none}.hero{padding-top:64px}body{font-size:16px}.math-block{text-align:left}}
  </style>
</head>
<body>
  <div class="progress" id="progress"></div>
  <nav class="top"><a class="brand" href="/">LLM / RL STUDY HUB</a><a class="hide-mobile" href="#content">中文精读</a><a class="primary" href="https://rail.eecs.berkeley.edu/deeprlcourse/" target="_blank" rel="noreferrer">CS285 官网 ↗</a></nav>
  <header class="hero"><small>WEEK 01 · CHINESE STUDY EDITION</small><h1>从策略梯度到 PPO</h1><p>把五份 CS285 课件重组为一条完整问题链：怎样利用带奖励的回答更新语言模型，同时避免一次更新让策略跑偏？</p></header>
  <main class="layout" id="content"><article>${body}<div class="note"><b>阅读建议：</b>先连续读完第一至第四章，再回到公式推导；每章结束后用自己的 LLM 任务代入 state、action、reward 和 advantage。</div></article><aside id="toc"><strong>本页目录</strong></aside></main>
  <footer>LLM / RL STUDY HUB · 第一周中文精读 · 公式由 MathJax 渲染</footer>
  <script>
    const headings=[...document.querySelectorAll('article h1,article h2')];
    const toc=document.getElementById('toc');
    headings.forEach((h,i)=>{h.id='section-'+i;const a=document.createElement('a');a.href='#'+h.id;a.textContent=h.textContent;a.style.paddingLeft=h.tagName==='H2'?'12px':'0';toc.appendChild(a)});
    addEventListener('scroll',()=>{const d=document.documentElement;document.getElementById('progress').style.width=(d.scrollTop/(d.scrollHeight-d.clientHeight)*100)+'%'},{passive:true});
  </script>
</body>
</html>`;

await Promise.all([
  fs.writeFile(path.join(projectDir, "public", "week1.html"), html, "utf8"),
  fs.writeFile(path.join(projectDir, "docs", "week1.html"), html, "utf8"),
]);
console.log(`Rendered ${html.length} bytes to public/week1.html and docs/week1.html`);
