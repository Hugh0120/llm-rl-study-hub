"use client";

import { useEffect, useMemo, useState } from "react";

type Resource = { label: string; href: string; videoLabel?: string; videoHref?: string };
type Week = {
  number: string;
  eyebrow: string;
  title: string;
  outcome: string;
  hours: string;
  guideHref: string;
  resources: Resource[];
  checklist: string[];
  deliverable: string;
};

const weeks: Week[] = [
  {
    number: "01",
    eyebrow: "基础压缩",
    title: "策略梯度、Advantage 与 PPO",
    outcome: "建立从 token 概率到 reward-weighted update 的直觉。",
    hours: "12–15h",
    guideHref: "/week1.html",
    resources: [
      { label: "CS285 · RL Basics", href: "https://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-4.pdf", videoLabel: "Fall 2023 · Lecture 4: Introduction to RL", videoHref: "https://www.youtube.com/playlist?list=PL_iWQOsE6TfVYGEGiAOMaOzzv41Jfm_Ps&index=4" },
      { label: "CS285 · Policy Gradients", href: "https://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-5.pdf", videoLabel: "Fall 2023 · Lecture 5: Policy Gradients", videoHref: "https://www.youtube.com/playlist?list=PL_iWQOsE6TfVYGEGiAOMaOzzv41Jfm_Ps&index=5" },
      { label: "CS285 · Actor-Critic", href: "https://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-6.pdf", videoLabel: "Fall 2023 · Lecture 6: Actor-Critic", videoHref: "https://www.youtube.com/playlist?list=PL_iWQOsE6TfVYGEGiAOMaOzzv41Jfm_Ps&index=6" },
      { label: "CS285 · Advanced Policy Gradients I", href: "https://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-9.pdf", videoLabel: "Fall 2023 · Lecture 9: Advanced Policy Gradients", videoHref: "https://www.youtube.com/playlist?list=PL_iWQOsE6TfVYGEGiAOMaOzzv41Jfm_Ps&index=9" },
      { label: "CS285 · Advanced Policy Gradients II", href: "https://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-10.pdf", videoLabel: "Fall 2023 · Lecture 9（最接近；课程版本拆分不同）", videoHref: "https://www.youtube.com/playlist?list=PL_iWQOsE6TfVYGEGiAOMaOzzv41Jfm_Ps&index=9" },
    ],
    checklist: [
      "把 prompt + 已生成 token 映射为 state，把下一个 token 映射为 action。",
      "理解 REINFORCE：高回报 rollout 的 log-prob 获得更强更新。",
      "能解释 baseline/value、advantage、PPO clipping 与 KL penalty 各自的作用。",
    ],
    deliverable: "画出：sample → reward → advantage → policy update。",
  },
  {
    number: "02",
    eyebrow: "LLM 后训练核心",
    title: "RLHF、可验证奖励与 GRPO",
    outcome: "理解偏好奖励与结果验证器分别如何驱动训练。",
    hours: "12–15h",
    guideHref: "/week2.html",
    resources: [
      { label: "CS224R · Reward Learning", href: "https://cs224r.stanford.edu/spring_2025/slides/08_cs224r_reward_learning_2025.pdf", videoLabel: "Spring 2025 · Lecture 8: Reward Learning", videoHref: "https://www.youtube.com/playlist?list=PLoROMvodv4rPwxE0ONYRa_itZFdaKCylL&index=8" },
      { label: "CS224R · RLHF / Preference Optimization", href: "https://cs224r.stanford.edu/spring_2025/slides/09_cs224r-2025-rlhf.pdf", videoLabel: "Spring 2025 · Lecture 9: RL for LLMs — Preference Optimization", videoHref: "https://www.youtube.com/playlist?list=PLoROMvodv4rPwxE0ONYRa_itZFdaKCylL&index=9" },
      { label: "CS224R · RL for Reasoning", href: "https://cs224r.stanford.edu/spring_2025/slides/10_cs224r-rl_for_reasoning_lecture.pdf", videoLabel: "Spring 2025 · Lecture 10: RL for LLMs — Reasoning", videoHref: "https://www.youtube.com/playlist?list=PLoROMvodv4rPwxE0ONYRa_itZFdaKCylL&index=10" },
      { label: "CS285 · LLM RL", href: "https://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-14.pdf", videoLabel: "替代录像：CS224R L9/L10（CS285 同版录像未公开）", videoHref: "https://www.youtube.com/playlist?list=PLoROMvodv4rPwxE0ONYRa_itZFdaKCylL&index=9" },
      { label: "CS285 · Homework 4（先读第 1–3 节）", href: "https://rail.eecs.berkeley.edu/deeprlcourse/static/homeworks/hw4.pdf" },
    ],
    checklist: [
      "区分 SFT、DPO、RLHF + PPO、GRPO 的训练信号和采样成本。",
      "画出 SFT → reward model → PPO，以及 prompt → group rollouts → verifier → GRPO。",
      "为数学或 JSON 任务写一个 0/1 verifier，并列出 3 种 reward hacking。",
    ],
    deliverable: "向同事讲清楚：DPO 为什么不等于 GRPO。",
  },
  {
    number: "03",
    eyebrow: "动手训练",
    title: "先跑通 format-copy",
    outcome: "用简单可验证任务验证完整训练管线，而不是盲目追求难题分数。",
    hours: "12–15h",
    guideHref: "/week3.html",
    resources: [
      { label: "CS285 · Homework 4（读第 4–8 节）", href: "https://rail.eecs.berkeley.edu/deeprlcourse/static/homeworks/hw4.pdf" },
      { label: "官方 starter code", href: "https://github.com/berkeleydeeprlcourse/homework_spring2026" },
      { label: "CS285 · LLM RL 回看", href: "https://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-14.pdf", videoLabel: "替代录像：CS224R L9 · Preference Optimization", videoHref: "https://www.youtube.com/playlist?list=PLoROMvodv4rPwxE0ONYRa_itZFdaKCylL&index=9" },
    ],
    checklist: [
      "先跑最小 smoke test，确认 tokenizer、rollout、reward 与日志闭环。",
      "完成 format-copy + GRPO，再完成 format-copy + GR-REINFORCE。",
      "每次记录 seed、配置、reward、KL、response length 和 10 条生成样例。",
    ],
    deliverable: "一页对比：两种算法的曲线、样例与训练稳定性。",
  },
  {
    number: "04",
    eyebrow: "实验与延伸",
    title: "math-hard、消融与 Agent RL",
    outcome: "把一次训练变成可复盘、可沟通的算法实验。",
    hours: "12–15h",
    guideHref: "/week4.html",
    resources: [
      { label: "CS285 · Homework 4：math-hard / diagnostics", href: "https://rail.eecs.berkeley.edu/deeprlcourse/static/homeworks/hw4.pdf" },
      { label: "Berkeley · Advanced LLM Agents", href: "https://rdi.berkeley.edu/adv-llm-agents/sp25" },
      { label: "视频：LLM Agents Lecture 1 · Inference-time reasoning", href: "https://www.youtube.com/live/g0Dwtf3BH-0" },
      { label: "CS285 课程主页", href: "https://rail.eecs.berkeley.edu/deeprlcourse/" },
    ],
    checklist: [
      "训练前人工抽查 20 条回答，确认 verifier 和奖励可信。",
      "完成 math-hard 的 GRPO 与 GR-REINFORCE 基线。",
      "只做一个消融：group size、KL 系数、格式奖励或长度限制，四选一。",
    ],
    deliverable: "一页报告：结果、成本、失败样例、reward hacking 风险与下一步。",
  },
];

const concepts = [
  ["SFT", "模仿高质量示范；是后训练的冷启动。"],
  ["PPO", "稳定的在线策略更新；用 clipping/KL 约束偏移。"],
  ["DPO", "直接从偏好对学习；不需要在线 rollout。"],
  ["GRPO", "同题多次采样后按组内相对奖励更新；适合可验证推理。"],
];

export function StudyHub() {
  const [done, setDone] = useState<Record<string, boolean>>({});
  const [active, setActive] = useState(0);

  useEffect(() => {
    const saved = window.localStorage.getItem("llm-rl-study-progress");
    if (saved) setDone(JSON.parse(saved));
  }, []);

  useEffect(() => {
    window.localStorage.setItem("llm-rl-study-progress", JSON.stringify(done));
  }, [done]);

  const total = weeks.reduce((sum, week) => sum + week.checklist.length, 0);
  const completed = Object.values(done).filter(Boolean).length;
  const progress = useMemo(() => Math.round((completed / total) * 100), [completed, total]);
  const week = weeks[active];

  function toggle(key: string) {
    setDone((previous) => ({ ...previous, [key]: !previous[key] }));
  }

  return (
    <main>
      <section className="hero">
        <div className="hero-grid" aria-hidden="true" />
        <nav className="nav"><span className="brand">LLM / RL</span><a href="#guides">四周教材</a><a href="#roadmap">配套资料</a><a href="#glossary">概念速查</a></nav>
        <div className="hero-inner">
          <p className="kicker">FOR ALGORITHM ENGINEERS · 4 WEEKS</p>
          <h1>把大模型<br /><em>后训练</em>做成能力。</h1>
          <p className="hero-copy">一条删去传统 RL 冗余内容的四周路径：从策略梯度直达 PPO、RLHF、GRPO 与可验证推理训练。</p>
          <div className="hero-actions"><a className="button primary" href="/week1.html">从第 1 周开始 <span>↗</span></a><a className="button ghost" href="#guides">打开四周教材目录 ↓</a></div>
        </div>
        <aside className="progress-card" aria-label="学习进度">
          <div><p className="label">YOUR PROGRESS</p><strong>{progress}%</strong></div>
          <div className="bar"><span style={{ width: `${progress}%` }} /></div>
          <p>{completed} / {total} 项已完成 · 自动保存在此设备</p>
        </aside>
      </section>

      <section id="guides" className="guide-index section">
        <div className="section-heading"><div><p className="kicker">START HERE</p><h2>四份中文教材，<br />按顺序直接学。</h2></div><p>每一周都是可独立阅读的完整章节，公式、符号、推导和工程上下文已串联好；Slides 与视频放在后面作为补充。</p></div>
        <div className="guide-grid">
          {weeks.map((item, index) => (
            <a className="guide-card" href={item.guideHref} key={item.number}>
              <div className="guide-card-top"><span>WEEK {item.number}</span><span>{item.hours}</span></div>
              <h3>{item.title}</h3>
              <p>{item.outcome}</p>
              <div className="guide-topics">
                {index === 0 && "MDP · REINFORCE · GAE · PPO"}
                {index === 1 && "RLHF · DPO · Verifier · RLVR · GRPO"}
                {index === 2 && "Tensor · Mask · Rollout · KL · Smoke test"}
                {index === 3 && "Ablation · Reward hacking · Agent RL"}
              </div>
              <strong>进入本周教材 <span>↗</span></strong>
            </a>
          ))}
        </div>
      </section>

      <section id="roadmap" className="roadmap section">
        <div className="section-heading"><div><p className="kicker">OPTIONAL MATERIALS</p><h2>需要时再查<br />Slides 与视频。</h2></div><p>先读上面的中文教材；遇到需要听讲解或核对原始定义的地方，再按周查看对应的官方课件与录像。</p></div>
        <div className="week-tabs" role="tablist" aria-label="学习周次">{weeks.map((item, index) => <button key={item.number} role="tab" aria-selected={active === index} className={active === index ? "active" : ""} onClick={() => setActive(index)}><span>{item.number}</span>{item.eyebrow}</button>)}</div>
        <article className="week-panel">
          <div className="week-main"><p className="kicker">WEEK {week.number} · {week.hours}</p><h3>{week.title}</h3><p className="outcome">{week.outcome}</p><a className="button primary" href={week.guideHref}>阅读本周中文教材 <span>↗</span></a><div className="resources"><p className="label">OPTIONAL · SLIDES ↔ MATCHED VIDEO</p>{week.resources.map((resource) => <div className="resource-pair" key={resource.href}><a href={resource.href} target="_blank" rel="noreferrer" className="resource-link"><span>课件</span>{resource.label}</a>{resource.videoHref && <a href={resource.videoHref} target="_blank" rel="noreferrer" className="video-link"><span>视频</span>{resource.videoLabel}</a>}</div>)}</div></div>
          <div className="week-work"><p className="label">THIS WEEK, DO THIS</p><ul>{week.checklist.map((task, index) => { const key = `${active}-${index}`; return <li key={key}><button className={done[key] ? "check done" : "check"} aria-label={`标记任务 ${index + 1} 完成`} onClick={() => toggle(key)}>{done[key] ? "✓" : ""}</button><span>{task}</span></li>; })}</ul><div className="deliverable"><p className="label">STOP WHEN YOU HAVE</p><strong>{week.deliverable}</strong></div></div>
        </article>
      </section>

      <section id="glossary" className="glossary section"><div className="section-heading"><div><p className="kicker">MINIMUM VOCABULARY</p><h2>先掌握这四个。</h2></div><p>不要把缩写当知识。每一个概念都要能映射到数据、loss 与训练曲线。</p></div><div className="concept-grid">{concepts.map(([name, description], index) => <article key={name} className="concept"><span>0{index + 1}</span><h3>{name}</h3><p>{description}</p></article>)}</div></section>

      <section className="finish"><p className="kicker">AFTER WEEK 4</p><h2>下一步：把 verifier<br />换成你的业务环境。</h2><p>数学、JSON、代码单测、工具调用都可以成为可验证奖励任务。先定义状态、动作、奖励、投机方式和评估集，再开始训练。</p><a className="button primary" href="https://rdi.berkeley.edu/adv-llm-agents/sp25" target="_blank" rel="noreferrer">进入 Agent RL 课程 ↗</a></section>

      <footer><span>LLM / RL STUDY HUB</span><span>官方资源链接 · 学习进度仅存储在本地浏览器</span></footer>
    </main>
  );
}
